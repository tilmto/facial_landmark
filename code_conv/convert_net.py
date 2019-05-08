import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.io import loadmat
import numpy as np
import cv2 as cv
import os
import sys

debug = False

class ConvertNet:
	def __init__(self,coord,ground_truth,scope='ConvertNet'):
		self.coord = coord
		self.ground_truth = ground_truth
		self.build_model(scope)
		self.loss = tf.reduce_mean(tf.square(self.output-self.ground_truth))

		global_step = tf.get_variable("global_step",initializer=0,trainable=False)
		boundaries = [450,1800,4500,9000,18000,45000]
		learing_rates = [1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4]

		self.lr = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learing_rates)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=global_step)

	def build_model(self,scope):
		with tf.variable_scope(scope):
			x = tf.reshape(self.coord,[-1,136])

			x = slim.fully_connected(x,256,scope='fc_1',activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(1e-4))
			x = slim.fully_connected(x,256,scope='fc_2',activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(1e-4))
			x = slim.fully_connected(x,256,scope='fc_3',activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(1e-4))
			x = slim.fully_connected(x,212,scope='fc_4',activation_fn=None,weights_regularizer=slim.l2_regularizer(1e-4))

			self.output = tf.reshape(x,[-1,106,2],name='output')

			'''
			x,y = tf.unstack(self.coord,axis=-1)

			x = slim.fully_connected(x,50,scope='fc_x1',activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(1e-4))
			x = slim.fully_connected(x,80,scope='fc_x2',activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(1e-4))
			x = slim.fully_connected(x,106,scope='fc_x3',activation_fn=None,weights_regularizer=slim.l2_regularizer(1e-4))

			y = slim.fully_connected(y,50,scope='fc_y1',activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(1e-4))
			y = slim.fully_connected(y,80,scope='fc_y2',activation_fn=tf.nn.relu,weights_regularizer=slim.l2_regularizer(1e-4))
			y = slim.fully_connected(y,106,scope='fc_y3',activation_fn=None,weights_regularizer=slim.l2_regularizer(1e-4))

			self.output = tf.stack([x,y],axis=-1)
			'''

def draw_points(img,output,orig,ground_truth):
	for i in range(106):
		cv.circle(img,(int(output[i][0]),int(output[i][1])),1,(0,255,0))
		cv.circle(img,(int(ground_truth[i][0]),int(ground_truth[i][1])),1,(0,0,255))
	for i in range(68):
		cv.circle(img,(int(orig[i][0]),int(orig[i][1])),1,(255,0,0))
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


def get_face_range(ground_truth_list):
	face_range_list = []
	for ground_truth in ground_truth_list:
		face_range_list.append(np.sqrt((np.max(ground_truth[:,0])-np.min(ground_truth[:,0]))*(np.max(ground_truth[:,1])-np.min(ground_truth[:,1]))))
	return np.reshape(np.array(face_range_list),[ground_truth_list.shape[0],1])


train_data = []
test_data = []

for root,dirs,files in os.walk('../conv_train'):
	for fname in files:
		if fname.endswith('.jpg'):
			cd_fname = '../conv_train/'+fname[0:fname.find('.')]+'_pts.mat'
			gt_fname = '../training_set/'+fname+'.txt'
			train_data.append([cd_fname,gt_fname])

for root,dirs,files in os.walk('../conv_valid'):
	for fname in files:
		if fname.endswith('.jpg'):
			cd_fname = '../conv_valid/'+fname[0:fname.find('.')]+'_pts.mat'
			gt_fname = '../training_set/'+fname+'.txt'
			test_data.append([cd_fname,gt_fname])

train_data = np.array(train_data)
test_data = np.array(test_data)

if debug:
	for i in range(train_data.shape[0]):
		img_gt = cv.imread(train_data[i][1][:-4])
		#cv.imwrite('img_gt'+str(i)+'.jpg',img_gt)
		print(img_gt.shape)
		img_cd = cv.imread(train_data[i][0][:-8]+'.jpg')
		print(img_cd.shape)
		#cv.imwrite('img_cd'+str(i)+'.jpg',img_cd)
	sys.exit(0)


def gene_train():
	for i in range(train_data.shape[0]):
		yield train_data[i]

def gene_test():
	for i in range(test_data.shape[0]):
		yield test_data[i]

train_generator = gene_train()
test_generator = gene_test()

batch_size = 20

def get_train_data():
	global train_generator
	coord_list = []
	ground_truth_list = []
	cd_img_list = []
	gt_img_list = []
	end_of_epoch = False

	for i in range(batch_size):
		try:
			cd_path,gt_path = train_generator.__next__()
		except StopIteration:
			train_generator = gene_train()
			end_of_epoch = True
			return np.array(coord_list),np.array(ground_truth_list),cd_img_list,gt_img_list,end_of_epoch

		cd_img_list.append(cd_path[:-8]+'.jpg')
		gt_img_list.append(gt_path[:-4])

		coord = loadmat(cd_path)['pts_2d']
		coord = (np.array(coord)-225)/225
		ground_truth = get_ground_truth_frame(gt_path)

		coord_list.append(coord)
		ground_truth_list.append(ground_truth)

	return np.array(coord_list),np.array(ground_truth_list),cd_img_list,gt_img_list,end_of_epoch

def get_test_data():
	global test_generator
	coord_list = []
	ground_truth_list = []
	cd_img_list = []
	gt_img_list = []
	test_complete = False

	for i in range(batch_size):
		try:
			cd_path,gt_path = test_generator.__next__()
		except StopIteration:
			test_generator = gene_test()
			test_complete = True
			return np.array(coord_list),np.array(ground_truth_list),cd_img_list,gt_img_list,test_complete

		cd_img_list.append(cd_path[:-8]+'.jpg')
		gt_img_list.append(gt_path[:-4])

		coord = loadmat(cd_path)['pts_2d']
		coord = (np.array(coord)-225)/225
		ground_truth = get_ground_truth_frame(gt_path)

		coord_list.append(coord)
		ground_truth_list.append(ground_truth)

	return np.array(coord_list),np.array(ground_truth_list),cd_img_list,gt_img_list,test_complete


tf.app.flags.DEFINE_boolean('load_model',False,'Load model or not')
tf.app.flags.DEFINE_boolean('is_training',True,'Is training or not')
flags = tf.app.flags.FLAGS

if __name__ == '__main__':
	coord = tf.placeholder(tf.float32,[None,68,2],name='coord')
	ground_truth = tf.placeholder(tf.float32,[None,106,2],name='ground_truth')
	model = ConvertNet(coord,ground_truth)

	epoch_num = 10000
	model_name = './my_model_best'
	best_error_avg = 1e10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(max_to_keep=50)

		if flags.load_model:
			saver.restore(sess,model_name)
			print('Load model from '+model_name)

		if flags.is_training:
			print('### do train ###')
			for ep_idx in range(epoch_num):
				loss_avg = 0
				batch_train_num = 0
				while True:
					coord_list,ground_truth_list,cd_img_list,gt_img_list,end_of_epoch = get_train_data()
					
					if coord_list.shape[0]!=0:
						feed_dict = {coord:coord_list,ground_truth:ground_truth_list}

						_,loss = sess.run([model.optimizer,model.loss],feed_dict=feed_dict)

						loss_avg += loss
						batch_train_num += 1

					if end_of_epoch:
						break

				loss_avg = loss_avg/batch_train_num
				print('epoch:',ep_idx,' loss_avg:',loss_avg)

				test_error_avg = 0
				batch_test_num =0
				while True:
					test_coord,test_ground_truth,test_cd_img_list,test_gt_img_list,test_complete = get_test_data()
					if test_coord.shape[0]!=0:
						feed_dict={coord:test_coord,ground_truth:test_ground_truth}
						output = sess.run(model.output,feed_dict=feed_dict)
						test_error_avg += np.mean(np.sqrt(np.square(output[:,:,0]-test_ground_truth[:,:,0])+np.square(output[:,:,1]-test_ground_truth[:,:,1]))/get_face_range(test_ground_truth))
						batch_test_num += 1

					if test_complete:
						break

				test_error_avg = test_error_avg/batch_test_num
				print('test_error_avg',test_error_avg)

				if best_error_avg>test_error_avg:
					saver.save(sess,model_name)
					best_error_avg = test_error_avg

		else:
			saver.restore(sess,model_name)
			print('Load model from '+model_name)

			print('### do test ###')

			test_error_avg = 0
			batch_test_num = 0
			error_array_avg = np.zeros((106,))
			while True:
				test_coord,test_ground_truth,cd_img_list,gt_img_list,test_complete = get_test_data()
				if test_coord.shape[0]!=0:
					feed_dict={coord:test_coord,ground_truth:test_ground_truth}
					output = sess.run(model.output,feed_dict=feed_dict)
					test_error_avg += np.mean(np.sqrt(np.square(output[:,:,0]-test_ground_truth[:,:,0])+np.square(output[:,:,1]-test_ground_truth[:,:,1]))/get_face_range(test_ground_truth))
					error_array_avg += np.mean(np.sqrt(np.square(output[:,:,0]-test_ground_truth[:,:,0])+np.square(output[:,:,1]-test_ground_truth[:,:,1]))/get_face_range(test_ground_truth),axis=0)
					batch_test_num += 1

					img = cv.imread(cd_img_list[0])
					test_coord = test_coord*225+225
					img = draw_points(img,output[0],test_coord[0],test_ground_truth[0])
					cv.imwrite('../gt/'+cd_img_list[0].split('/')[-1],img)

				if test_complete:
					break

			test_error_avg = test_error_avg/batch_test_num
			error_array_avg = error_array_avg/batch_test_num
			print('test_error_avg:',test_error_avg)
			print('error distribution:\n',error_array_avg)