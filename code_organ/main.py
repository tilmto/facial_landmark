"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
"""

import tensorflow as tf
import numpy as np
import os
import csv
import shutil
#import matplotlib.pyplot as plt
from model_ResNet import ResNet152
import copy
import compute_error
import random
import time
import Gen_ICME_img

MAX_GLOBAL_EP = 500
RANDOM_coord_NUM=5
load_model=None
POINT_NUMS=106

def model_size():
    params = tf.trainable_variables()
    size = 0
    for x in params:
        print(x.name)
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        print(x.get_shape)
        print(sz)
        size += sz
    return size

class ACNet(object):
    def __init__(self, scope, session):
        global coord_NUM,POINT_NUMS
        self.session = session
        self.weight_landmarks = 5
        self.state_tensor = tf.placeholder(tf.float32, [None, H, W,3], name='images')
        self.landmarks = tf.placeholder(tf.float32, [None, POINT_NUMS,2], name='landmarks')
        self.is_training=tf.placeholder(tf.bool, name='is_training')
        self.mask = tf.placeholder(tf.float32,[None,106,2],name='mask')
        self._build_net()

        self.loss_detection = tf.reduce_mean(tf.square((self.model.landmark_output*self.mask - self.landmarks*self.mask)))

        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss_detection)
        self.optimizer_precise = tf.train.AdamOptimizer(3e-5).minimize(self.loss_detection)
            
    def _build_net(self):
        global coord_NUM, TRAIN_SUB_NUM, POINT_NUMS

        self.model=ResNet152(self.state_tensor, self.landmarks, POINT_NUMS, self.is_training)


class Worker(object):
    def __init__(self, name, session):
        self.name = name
        self.session=session
        self.AC = ACNet(name, session)

def build_worker(session,is_training,model_name='my_model_best'):
    worker=Worker('worker0', session)
    saver = tf.train.Saver(max_to_keep=50)
    if not is_training:
        if flags.load_model:
            session.run(tf.global_variables_initializer())
            saver.restore(session, model_name)
            print('Loaded model from', model_name)
        else:
            print('Created and initialized fresh model. Size:', model_size())
            session.run(tf.global_variables_initializer())
        return session,worker,worker.AC,saver
    # Create worker
    if flags.load_model:
        session.run(tf.global_variables_initializer())
        saver.restore(session, model_name)
        print('Loaded model from', model_name)
    else:
        print('Created and initialized fresh model. Size:', model_size())
        #exit(0)
        session.run(tf.global_variables_initializer())
    return session,worker,worker.AC,saver


tf.app.flags.DEFINE_string('model', '1', '1:Hourglass\n2:ResNet101(Regression)\n3:ResNet101(Heatmap)')
tf.app.flags.DEFINE_integer('batch_size',20, 'Input the batch size')
tf.app.flags.DEFINE_boolean('is_training',True,'Training mode or not')
tf.app.flags.DEFINE_boolean('load_model',False,'Load model or not')
flags = tf.app.flags.FLAGS


if __name__ == '__main__':
    ############parameters############
    model_name='my_model_best'
    is_training=flags.is_training
    H,W=(256,256)
    ############training & testing process############

    train_logs = open('train_logs.txt','w')

    Gen_ICME_img.BATCH_SIZE=flags.batch_size
    Gen_ICME_img.train_id={}
    Gen_ICME_img.test_id={}
    Gen_ICME_img.test_tensor=[]
    Gen_ICME_img.test_labels=[]

    best_test_all={}
    for num in range(POINT_NUMS):
        best_test_all[num]=10000.0    
    save_idx=0

    config=tf.ConfigProto(inter_op_parallelism_threads=6,intra_op_parallelism_threads=6)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=config) as session:
        session,worker,GLOBAL_AC,saver=build_worker(session,is_training,model_name)
        #print('output layer name:',GLOBAL_AC.model.landmark_output1.name)
        if is_training:
            print('do train!!!!!!')
        else:
            print('do test!!!!!!')

        best_test_error=10000000.0

        for ep_idx in range(MAX_GLOBAL_EP if is_training else 1):
            avg_pre_a_loss=0.0
            batch_interval=0
            Gen_ICME_img.END_OF_ONE_EPOCH=False
            while Gen_ICME_img.END_OF_ONE_EPOCH==False:
                #Below: ICME Data
                state_tensor,coord_labels,mask=Gen_ICME_img.get_ICME_img(H,W,use_random_sample=False if ep_idx==0 else True)                  
                feed_dict_a={
                    GLOBAL_AC.state_tensor: state_tensor,#[neg_size*batch_size,pointnums,2]
                    GLOBAL_AC.landmarks: coord_labels,                                             
                    GLOBAL_AC.is_training:True,
                    GLOBAL_AC.mask:mask                        
                }
                if is_training and ep_idx:
                    if best_test_error>=0.01500:
                        _,loss=session.run([GLOBAL_AC.optimizer,GLOBAL_AC.loss_detection],feed_dict=feed_dict_a)
                    else:
                        _,loss=session.run([GLOBAL_AC.optimizer_precise,GLOBAL_AC.loss_detection],feed_dict=feed_dict_a)
                    avg_pre_a_loss+=loss

                batch_interval+=1

            if not ep_idx:
                print('Read all images.')
                continue
            else:
                print('avg_pre_a_loss: ',avg_pre_a_loss/batch_interval)

            #################test on ibug#################
            error_array_acc=np.zeros((106,))
            test_instance_num=0
            failure_rate=0.0
            test_error_avg=0.0
            while True:
                img_list,state_tensor,coord_labels=Gen_ICME_img.get_validate_img()
                feed_dict_a={
                    GLOBAL_AC.state_tensor: state_tensor,#[neg_size*batch_size,pointnums,2]
                    GLOBAL_AC.is_training:False,                        
                }
                if int(state_tensor.shape[0])==0:
                    break
                #print(state_tensor.shape)   
                landmark_output=session.run(GLOBAL_AC.model.landmark_output,feed_dict=feed_dict_a)

                #landmark_output = landmark_output*128+127.5

                for ii in range(len(img_list)):
                    _,under_threshold,_,error,error_array=compute_error.compute_error_all_point(landmark_output[ii],coord_labels[ii],0.08,H)
                    error_array_acc+=error_array
                    # cut_img=img_list[ii]
                    # show_img=copy.deepcopy(cut_img)
                    # for test_point in range(coord_labels.shape[1]):
                    #     cv.circle(show_img,(int(coord_labels[ii][test_point][0]),int(coord_labels[ii][test_point][1])),2,(0,0,255),thickness=-1)
                    #     cv.circle(show_img,(int(landmark_output[ii][test_point][0]),int(landmark_output[ii][test_point][1])),2,(0,255,0),thickness=-1)
                    # cv.imshow('show_img',show_img)
                    # cv.waitKey(1000)
                    if not under_threshold:
                        failure_rate+=1.0
                    test_error_avg+=error                        
                    test_instance_num+=1

            test_error_avg/=test_instance_num
            failure_rate/=test_instance_num
            error_array_acc/=test_instance_num

            test_error_acc_list=error_array_acc.tolist()
            update_nums=0
            str_w=''
            for num in range(len(test_error_acc_list)):
                test_error_single=test_error_acc_list[num]
                if best_test_all[num]>test_error_single:
                    update_nums+=1
            if update_nums>int(POINT_NUMS/10):
                for num in range(len(test_error_acc_list)):
                    test_error_single=test_error_acc_list[num]
                    if best_test_all[num]>test_error_single:
                        best_test_all[num]=test_error_single
                        str_w+=str(num)+'\n'
                if is_training:
                    save_as='./my_model_temp'+'_'+str(save_idx)#_at_'+str(ep_idx)
                    saver.save(session, save_as)
                    f_write=open('model_log_'+str(save_idx)+'.txt','w')
                    f_write.write(str_w)
                    f_write.close()
                save_idx+=1
                if save_idx>=10:
                    save_idx=0
                #print(best_test_all.values())
                #print(type(best_test_all.values()))
                best_value_ind=0.0
                for test_error_single in best_test_all.values():
                    best_value_ind+=test_error_single
                best_value_ind/=len(best_test_all)
                
            if best_test_error>test_error_avg:
                best_test_epx=ep_idx
                best_test_error=test_error_avg
                if is_training:
                    save_as='./my_model_best'#_at_'+str(ep_idx)
                    saver.save(session, save_as)
      
            if not is_training:
                with open('error_log.csv','w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Point','Avg_Error'])
                    for j in range(106):
                        writer.writerow([j+1,error_array_acc[j]]) 

            print('epoch: '+ str(ep_idx)+' results on validation, test_error_avg:',test_error_avg,'failure_rate:',failure_rate)
            print('best_epoch: ',best_test_epx,' best_error_avg: ',best_test_error,' best_value_ind:',best_value_ind)

            train_logs.write('epoch: '+str(ep_idx)+'  avg_pre_a_loss: '+str(avg_pre_a_loss/batch_interval)+'\ntest_error_avg: '+str(test_error_avg)+'  failure_rate: '+str(failure_rate)+' best_error_avg: '+str(best_test_error)+' best_value_ind:'+str(best_value_ind)+'\n\n')

    train_logs.close()
