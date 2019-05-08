import tensorflow as tf
import tensorflow.contrib.slim as slim

class ResNet101_HM:
	def __init__(self,images,ground_truth,points_num=106,is_training=True,img_size=256,scope='ResNet101'):
		self.images = images
		self.ground_truth = ground_truth
		self.points_num = points_num
		self.img_size = img_size
		self.batch_size = images.shape[0].value

		self.build_model(scope,is_training)

		x = tf.nn.sigmoid(self.heatmap)

		for i in range(20):
			for j in range(106):
				tf.add_to_collection('gt_loc',x[i,tf.cast(ground_truth[i,j,0],tf.int32),tf.cast(ground_truth[i,j,1],tf.int32),j])

		self.loss_detection = tf.reduce_sum(tf.square(x))-tf.reduce_sum(2*tf.get_collection('gt_loc'))-1
		self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss_detection)


	def build_model(self,scope,is_training=True):
		x = self.images

		with tf.variable_scope(scope):
			x = slim.conv2d(x,32,[7,7],1,padding='SAME',activation_fn=None,scope='conv1')
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='bn1'))

			x = slim.pool(x,[3, 3],'MAX',stride=2,padding='SAME',scope='pool1')

			x = self.residual_module(x,64,2,is_training,'residual128_1')
			for i in range(2):
				x = self.residual_module(x,128,1,is_training,'residual1_'+str(i+2))

			for i in range(4):
				x = self.residual_module(x,256,1,is_training,'residual2_'+str(i+1))

			for i in range(23):
				x = self.residual_module(x,512,1,is_training,'residual3_'+str(i+1))

			for i in range(3):
				x = self.residual_module(x,1024,1,is_training,'residual4_'+str(i+1))

			x = slim.conv2d(x,self.points_num,[1,1],1,padding='SAME',activation_fn=None,scope='compress')

			x = tf.image.resize_images(x,[self.img_size,self.img_size],method=0)

			self.heatmap = x

			x_flat = tf.reshape(x,[-1,self.img_size*self.img_size,self.points_num])
			x_flat = tf.argmax(x_flat,1)

			landmark_x = tf.cast(tf.floor(tf.divide(x_flat,self.img_size)),tf.int64)
			landmark_x = tf.reshape(landmark_x,[-1,1,self.points_num])

			landmark_y = tf.mod(x_flat,self.img_size)
			landmark_y = tf.reshape(landmark_y,[-1,1,self.points_num])
			
			x_flat = tf.concat([landmark_x,landmark_y],1)

			self.landmark_output = tf.transpose(x_flat,[0,2,1])


	def residual_module(self,x,out_planes,stride=1,is_training=True,scope='residual'):
		in_planes = x.shape[-1].value
		orig_x = x

		with tf.variable_scope(scope):
			x = slim.conv2d(x,out_planes/4,[1,1],stride,padding='SAME',activation_fn=None,scope='rconv1')
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='rbn1'))
			x = slim.conv2d(x,out_planes/4,[3,3],1,padding='SAME',activation_fn=None,scope='rconv2')
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='rbn2'))
			x = slim.conv2d(x,out_planes,[1,1],1,padding='SAME',activation_fn=None,scope='rconv3')

			if in_planes != out_planes or stride != 1:
				orig_x = slim.conv2d(orig_x,out_planes,[1,1],stride,padding='SAME',activation_fn=None,scope='downsample')

			x = tf.nn.relu(self.batch_norm(x+orig_x,is_training,scope='rbn3'))

		return x


	def batch_norm(self,x,is_training=True,scope='bn',moving_decay=0.9,eps=1e-6):
	    with tf.variable_scope(scope):
	        gamma = tf.get_variable('gamma',x.shape[-1],initializer=tf.constant_initializer(1))
	        beta  = tf.get_variable('beta', x.shape[-1],initializer=tf.constant_initializer(0))

	        axes = list(range(len(x.shape)-1))
	        batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

	        ema = tf.train.ExponentialMovingAverage(moving_decay)

	        def mean_var_with_update():
	            ema_apply_op = ema.apply([batch_mean,batch_var])
	            with tf.control_dependencies([ema_apply_op]):
	                return tf.identity(batch_mean), tf.identity(batch_var)

	        mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
	                lambda:(ema.average(batch_mean),ema.average(batch_var)))

	        return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)

if __name__ == '__main__':
	images = tf.placeholder(tf.float32, [20,256,256,3], name='images')
	ground_truth = tf.placeholder(tf.float32,[20,106,2],name = 'ground_truth')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		resnet = ResNet101_HM(images,ground_truth)
		print(resnet.landmark_output.shape)




