import tensorflow as tf
import tensorflow.contrib.slim as slim

class ResNet101_HMN:
	def __init__(self,images,ground_truth,points_num=106,is_training=True,img_size=256,scope='ResNet101'):
		self.images = images
		self.ground_truth = ground_truth
		self.points_num = points_num
		self.img_size = img_size

		self.build_model(scope,is_training)

		self.loss_detection = tf.reduce_mean(tf.square((self.landmark_output - self.ground_truth)))
		
		global_step = tf.get_variable("global_step",initializer=0,trainable=False)
		#lr = tf.train.exponential_decay(learning_rate=0.001,global_step=global_step,decay_steps=1140,decay_rate=0.9)
		#lr = tf.train.piecewise_constant(global_step, boundaries=[22800,45600], values=[1e-4,5e-5,1e-5])
		#lr = tf.train.piecewise_constant(global_step, boundaries=[2850,57000,114000,171000], values=[1e-3,1e-4,3e-5,1e-5,1e-6])
		lr = 1e-4
		self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss_detection,global_step=global_step)


	def build_model(self,scope,is_training=True):
		x = self.images

		with tf.variable_scope(scope):
			x = slim.conv2d(x,16,[7,7],1,padding='SAME',activation_fn=None,scope='conv1')
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='bn1'))

			x = slim.pool(x,[3, 3],'MAX',stride=2,padding='SAME',scope='pool1')

			x = self.residual_module(x,32,2,is_training,'residual128_1')
			for i in range(2):
				x = self.residual_module(x,64,1,is_training,'residual1_'+str(i+2))

			for i in range(4):
				x = self.residual_module(x,128,1,is_training,'residual2_'+str(i+1))

			for i in range(23):
				x = self.residual_module(x,256,1,is_training,'residual3_'+str(i+1))

			for i in range(3):
				x = self.residual_module(x,512,1,is_training,'residual4_'+str(i+1))

			x = slim.conv2d(x,self.points_num,[1,1],1,padding='SAME',activation_fn=None,scope='compress')

			#x = tf.image.resize_images(x,[self.img_size,self.img_size],method=0)
			hm_size = x.shape[1].value
			x = tf.reshape(x,[-1,hm_size*hm_size,self.points_num])
			x = tf.nn.softmax(x,1)
			x = tf.transpose(x,[0,2,1])
			x = tf.reshape(x,[-1,self.points_num,hm_size,hm_size])

			h = tf.constant([[(2*i-(hm_size+1))/hm_size for i in range(1,hm_size+1)] for j in range(hm_size)])
			h = tf.reshape(h,[1,1,hm_size,hm_size])
			w = tf.transpose(h,[0,1,3,2])

			landmark_x = (tf.reduce_sum(x*h,[2,3])*self.img_size+self.img_size+1)/2
			landmark_x = tf.reshape(landmark_x,[-1,self.points_num,1])

			landmark_y = (tf.reduce_sum(x*w,[2,3])*self.img_size+self.img_size+1)/2
			landmark_y = tf.reshape(landmark_y,[-1,self.points_num,1])

			self.landmark_output = tf.concat([landmark_x,landmark_y],2)


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
	images = tf.placeholder(tf.float32, [10,256,256,3], name='images')
	ground_truth = tf.placeholder(tf.float32,[10,106,2],name = 'ground_truth')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		resnet = ResNet101_HM(images,ground_truth)
		print(resnet.landmark_output.shape)




