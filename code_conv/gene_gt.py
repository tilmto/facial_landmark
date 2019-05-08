import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import cv2 as cv
import os

def draw_points(img,ground_truth):
	for i in range(106):
		cv.circle(img,(int(ground_truth[i][0]),int(ground_truth[i][1])),1,(0,0,255))
	return img

def get_ground_truth_frame(fname):
	ground_truth_file = open(fname,'r')
	ground_truth_file.readline()
	ground_truth_frame = []
	for i in range(106):
		line = ground_truth_file.readline()
		if not line:
			break
		ground_truth_frame.append([float(x) for x in line.strip().split()])
	ground_truth_file.close()
	return np.array(ground_truth_frame,np.float32)

debug = False

fname_list = []
for root,dirs,files in os.walk('../side_face_extern'):
	for fname in files:
		if fname.endswith('.mat'):
			fname_list.append(os.path.join(root,fname))
fname_list = np.array(fname_list)

print('Total face num:',fname_list.shape[0])

config=tf.ConfigProto(inter_op_parallelism_threads=6,intra_op_parallelism_threads=6)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
	load_graph='my_model_best'+'.meta'
	saver = tf.train.import_meta_graph(load_graph)
	saver.restore(sess,'my_model_best' )
	graph = tf.get_default_graph()
	coord = graph.get_tensor_by_name('coord:0')
	output = graph.get_tensor_by_name('ConvertNet/output:0')

	batch_size = 20
	point = 0
	end_of_data = False

	img_list= []
	landmark_array = np.zeros((0,106,2))

	print('### do converting ###')
	while True:
		coord_list = []
		for i in range(batch_size):
			if point>=fname_list.shape[0]:
				end_of_data = True
				break
			else:
				coord_list.append(loadmat(fname_list[point])['pts_2d'])
				img_list.append(fname_list[point][:-8]+'.jpg')
				point += 1

		coord_list = (np.array(coord_list)-225)/225
		
		if coord_list.shape[0]!=0:
			feed_dict = {coord:coord_list}
			landmarks = sess.run(output,feed_dict=feed_dict)
			landmark_array=np.concatenate((landmark_array,landmarks),axis=0)

		if end_of_data:
			break

	for i in range(len(img_list)):
		with open(img_list[i]+'.txt','w') as f:
			f.write('106\n')
			for j in range(106):
				f.write(str(landmark_array[i][j][0])+' '+str(landmark_array[i][j][1])+'\n')

		if debug:
			ground_truth_frame = get_ground_truth_frame(img_list[i]+'.txt')
			img = cv.imread(img_list[i])
			img = draw_points(img,ground_truth_frame)
			cv.imwrite(str(i+1)+'.jpg',img)
			print('generate figure '+str(i+1))
			input()
