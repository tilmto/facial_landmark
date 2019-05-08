import tensorflow as tf
import numpy as np
import cv2 as cv
import sys
import copy
import os
import slice_image
import compute_error
import csv
import json

'''
tinput_args = sys.argv[1:]
ICME_Root =  tinput_args[0]
output_Root = tinput_args[1]
'''
ICME_Root = '../training_set'

BATCH_SIZE=15
H,W=(256,256)
POINT_NUMS=106
root_fold=''
END_OF_ONE_EPOCH=False


def walk_dir(dirname):
 for root,dirs,files in os.walk(dirname):
  for f in files:
   yield [root, f]

walk_obj_ICME=walk_dir(ICME_Root)
###########generating the dict for each model###########
start_idx=1
overall_best_idx=30
prefix='model_log_'
f_dict={}
pt_dict={}
for i in range(10):
    curr_idx=(start_idx+i)%10
    f_dict[curr_idx]=[]
    fname=prefix+str(curr_idx)+'.txt'
    try:
        file_r=open(fname,'r')
    except:
        continue
    while True:
        line=file_r.readline()
        if not line:
            break
        f_dict[curr_idx].append(int(line.strip()))
        if not int(line.strip()) in pt_dict:
            pt_dict[int(line.strip())]=[]
        pt_dict[int(line.strip())].append(curr_idx)
    file_r.close()
# print(f_dict)
# print(pt_dict)
print(len(pt_dict))
for key in f_dict.keys():
    list_cp= copy.deepcopy(f_dict[key])
    for item in f_dict[key]:
        if pt_dict[item][-1]!=key:
            list_cp.remove(item)
    f_dict[key]=list_cp
print(f_dict)
'''---"Here, parameter1 refers to the absolute path for the input file (.jpg) 
and parameter2 refers to the absolute path for the output file (.txt)."    '''

#print('Image origin scale:',tinput_img1.shape,'scale_factor:',scale_factor,'\n\n\n')

##########draw landmarks function###########
def draw_points(img,predict,ground_truth):
    for i in range(0,106):
        cv.circle(img,(int(predict[i][0]),int(predict[i][1])),1,(0,255,0))
        cv.circle(img,(int(ground_truth[i][0]),int(ground_truth[i][1])),1,(0,0,255))
    return img

def get_ground_truth_frame(fname):
    ground_truth_file=open(fname,'r')
    ground_truth_file.readline()
    ground_truth_frame=[]
    for i in range(106):
        line=ground_truth_file.readline()
        if not line:
            break
        ground_truth_frame.append([float(x) for x in line.strip().split()])
    ground_truth_file.close()
    return np.array(ground_truth_frame,np.float32)


def get_ICME_img(H=256,W=256):
    global BATCH_SIZE,walk_obj_ICME,ICME_Root,END_OF_ONE_EPOCH,root_fold
    fname_list=[]
    test_img_list=[]
    test_tensor=[]
    scale_factor_tensor=[]
    ground_truth=[]
    i=0
    while i<BATCH_SIZE:
        try:
            root_fold,fname=walk_obj_ICME.__next__()
        except StopIteration:
            END_OF_ONE_EPOCH=True
            break

        if (fname.endswith('.png') or fname.endswith('.jpg')) and (fname.find('IBUG')!=-1 or fname.find('ibug')!=-1):
            tinput_img1= cv.imread(root_fold+'/'+fname)
            fname_list.append(fname)

            ground_truth_frame=get_ground_truth_frame(root_fold+'/'+fname+'.txt')

            file_rect=open(root_fold+'/'+fname+'.rect','r')
            left,top,right,bottom=map(int,file_rect.readline().strip().split())
            file_rect.close()

            tinput_img,_,scale_factor = slice_image.slice_frame(tinput_img1,left,top,right,bottom,(tinput_img1.shape[1],tinput_img1.shape[0]),slicewh=(W,H))
            scale_factor_tensor.append(scale_factor)

            test_img_list.append(tinput_img)     

            s_input=(tinput_img-127.5)/128.0
            test_tensor.append(s_input)

            ground_truth_frame=(ground_truth_frame-np.array([left,top]))*scale_factor
            ground_truth.append(ground_truth_frame)

            i+=1

    return np.array(test_tensor),np.expand_dims(np.array(scale_factor_tensor),axis=1),fname_list,test_img_list,ground_truth




#########predict##########
if __name__ == '__main__':
    prefix='my_model_temp_'    
    config=tf.ConfigProto(inter_op_parallelism_threads=6,intra_op_parallelism_threads=6)
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        load_graph='my_model_best'+'.meta'#+str(overall_best_idx)+'.meta'
        saver = tf.train.import_meta_graph(load_graph)
        saver.restore(sess,'my_model_best' )
        graph = tf.get_default_graph()
        image = graph.get_tensor_by_name("images:0")#(None,H,W,3)
        flag = graph.get_tensor_by_name("is_training:0")#bool
        output = graph.get_tensor_by_name("ResNet/landmark:0")
        all_test_img_list=[]
        all_fname_list=[]
        all_ground_truth=[]
        ################traverse the testing folder################
        landmarks_array=np.zeros((0,POINT_NUMS,2),dtype=np.float32)
        END_OF_ONE_EPOCH=False
        walk_obj_ICME=walk_dir(ICME_Root)
        while not END_OF_ONE_EPOCH:
            test_tensor,scale_factor_tensor,fname_list,test_img_list,ground_truth=get_ICME_img()
            if len(fname_list)==0:
                break
            all_test_img_list.extend(test_img_list)
            all_fname_list.extend(fname_list)
            all_ground_truth.extend(ground_truth)

            feed_dict={
                image:test_tensor,
                flag:False,                        
            }
            landmarks = sess.run(output,feed_dict=feed_dict)#(BATCH_SIZE,POINT_NUMS,2)
            
            # print(landmarks_array.shape)
            # print(landmarks.shape)
            landmarks_array=np.concatenate((landmarks_array,landmarks),axis=0)
        sess.close()

        for i in range(10):
            with tf.Graph().as_default(), tf.Session(config=config) as sess:
                #break
                # if i==0:
                #     continue
                #if i>1:
                #break
                curr_idx=(start_idx+i)%10
                if curr_idx==overall_best_idx:
                    continue
                if not curr_idx in  f_dict or len(f_dict[curr_idx])==0:
                    continue
                load_graph=prefix+str(curr_idx)+'.meta'
                #try:
                saver = tf.train.import_meta_graph(load_graph)
                #except:
                #continue
                saver.restore(sess,prefix+str(curr_idx))
                graph = tf.get_default_graph()
                image = graph.get_tensor_by_name("images:0")#(None,H,W,3)
                flag = graph.get_tensor_by_name("is_training:0")#bool
                output = graph.get_tensor_by_name("ResNet/landmark:0")
                landmarks_temp_array=np.zeros((0,POINT_NUMS,2),dtype=np.float32)
                END_OF_ONE_EPOCH=False
                walk_obj_ICME=walk_dir(ICME_Root)
                while not END_OF_ONE_EPOCH:
                    test_tensor,scale_factor_tensor,fname_list,test_img_list,ground_truth=get_ICME_img()
                    if len(fname_list)==0:
                        break
                    feed_dict={
                        image:test_tensor,
                        flag:False,                        
                    }
                    landmarks = sess.run(output,feed_dict=feed_dict)#(BATCH_SIZE,POINT_NUMS,2)
                    
                    landmarks_temp_array=np.concatenate((landmarks_temp_array,landmarks),axis=0)
                landmarks_array[:,np.array(f_dict[curr_idx],dtype=np.int32),:]=copy.deepcopy(landmarks_temp_array[:,np.array(f_dict[curr_idx],dtype=np.int32),:])
                print(load_graph)
                print(np.array(f_dict[curr_idx],dtype=np.int32))
                sess.close()

        '''
        error_list = []
        for i in range(len(all_ground_truth)):
            _,under_threshold,_,error,error_array=compute_error.compute_error_all_point(landmarks_array[i],all_ground_truth[i],0.08,256)
            error_list.append(error)
        
        for i in range(len(all_test_img_list)):
            img = draw_points(all_test_img_list[i],landmarks_array[i],all_ground_truth[i])
            cv.imwrite('./valid_img/'+str(error_list[i])+'@'+all_fname_list[i],img)
        '''

    valid = {}
    for i in range(len(all_fname_list)):
        valid[all_fname_list[i]]=[landmarks_array[i].tolist(),all_ground_truth[i].tolist()]

    with open('fyg.json','w') as f:
        json.dump(valid,f)


    '''            
    for i in range(len(all_fname_list)):
        fname=all_fname_list[i]
        landmarks_ori=landmarks_array[i,:,:]
        #draw_points(all_test_img_list[i],landmarks_ori,fname)
        #########write into txt file###########
        fo = open(output_Root+'/'+fname+'.txt', "w")
        fo.write('106\n')
        for i in range(0,106):
            fo.write(str(landmarks_ori[i][0])+' '+str(landmarks_ori[i][1])+'\n')
        fo.close()
    '''






