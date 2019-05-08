import tensorflow as tf
import numpy as np
import cv2 as cv
import sys
import copy
import os

tinput_args = sys.argv[1:]
ICME_Root =  tinput_args[0]
output_Root = tinput_args[1]

BATCH_SIZE=1
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
start_idx=4
overall_best_idx=6
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
def draw_points(img,points,fname):
    global output_Root
    for i in range(0,106):
        cv.circle(img,(int(points[i][0]),int(points[i][1])),2,(0,0,255),thickness=-1)
    '''cv.imshow('image',img)
    cv.waitKey(0)'''
    cv.imwrite(output_Root+'/'+fname,img)

def get_ICME_img(H=256,W=256):
    global BATCH_SIZE,walk_obj_ICME,ICME_Root,END_OF_ONE_EPOCH,root_fold
    fname_list=[]
    test_img_list=[]
    test_tensor=[]
    scale_factor_tensor=[]
    i=0
    while i<BATCH_SIZE:
        try:
            root_fold,fname=walk_obj_ICME.__next__()
        except StopIteration:
            END_OF_ONE_EPOCH=True
            break
        tinput_img1= cv.imread(root_fold+'/'+fname)
        test_img_list.append(tinput_img1)
        fname_list.append(fname)
        ##########reshape input image###########
        tinput_img = cv.resize(tinput_img1,(W,H))#default interpolation
        scale_factor = [float(tinput_img1.shape[1])/float(tinput_img.shape[1]),float(tinput_img1.shape[0])/float(tinput_img.shape[0])]
        scale_factor_tensor.append(scale_factor)     
        #sketch_cut_img=sketch(cut_img)
        #cut_img=np.concatenate((cut_img,sketch_cut_img),axis=-1)
        s_input=(tinput_img-127.5)/128.0
        test_tensor.append(s_input)
        i+=1
        # show_img=copy.deepcopy(cut_img)
        # for test_point in range(ground_truth_frame.shape[0]):
        #     cv.circle(show_img,(int(test_labels[-1][test_point][0]),int(test_labels[-1][test_point][1])),2,(0,255,0),thickness=-1)
        # cv.imshow('show_img',show_img)
        # cv.waitKey(1000)
    #print(state_tensor[0].shape)
    return np.array(test_tensor),np.expand_dims(np.array(scale_factor_tensor),axis=1),fname_list,test_img_list




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
        ################traverse the testing folder################
        landmarks_array=np.zeros((0,POINT_NUMS,2),dtype=np.float32)
        END_OF_ONE_EPOCH=False
        walk_obj_ICME=walk_dir(ICME_Root)
        while not END_OF_ONE_EPOCH:
            test_tensor,scale_factor_tensor,fname_list,test_img_list=get_ICME_img()
            if len(fname_list)==0:
                break
            all_test_img_list.extend(test_img_list)
            all_fname_list.extend(fname_list)
            feed_dict={
                image:test_tensor,
                flag:False,                        
            }
            landmarks = sess.run(output,feed_dict=feed_dict)#(BATCH_SIZE,POINT_NUMS,2)
            landmarks = landmarks*scale_factor_tensor
            # print(landmarks_array.shape)
            # print(landmarks.shape)
            landmarks_array=np.concatenate((landmarks_array,landmarks),axis=0)
        sess.close()
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






