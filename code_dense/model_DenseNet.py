import tensorflow as tf
import tensorflow.contrib.slim as slim

class DenseNet:
	def __init__(self,images,ground_truth,points_num=106,is_training=True,img_size=256,scope='DenseNet'):
		self.images = images
		self.ground_truth = ground_truth
		self.points_num = points_num
		self.img_size = img_size

		self.filters = 32

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
			x = slim.conv2d(x,64,[7,7],2,padding='SAME',activation_fn=None,scope='conv1')
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='bn1'))
			x = slim.max_pool2d(x,[3, 3],stride=2,padding='SAME',scope='max_pool')

			x = self.dense_block(x,6,is_training=is_training,scope='dense_block_1')
			x = self.trans_layer(x,is_training=is_training,scope='transition_1')

			x = self.dense_block(x,12,is_training=is_training,scope='dense_block_2')
			x = self.trans_layer(x,is_training=is_training,scope='transition_2')

			x = self.dense_block(x,48,is_training=is_training,scope='dense_block_3')
			x = self.trans_layer(x,is_training=is_training,scope='transition_3')

			x = self.dense_block(x,32,is_training=is_training,scope='dense_block_4')

			x = tf.nn.relu(self.batch_norm(x,is_training,scope='bn2'))
			x = slim.conv2d(x,self.points_num,[1,1],2,padding='SAME',activation_fn=None,scope='channel_fusion')
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='bn3'))

			x = slim.conv2d(x,self.points_num,[4,2],2,padding='VALID',activation_fn=None,scope='compress')

			self.landmark_output = tf.reshape(x,[-1,self.points_num,2],name='landmark')
			

	def dense_block(self,x,layers,stride=1,is_training=True,scope='dense_block'):
		input = []
		input.append(x)

		with tf.variable_scope(scope):
			for i in range(layers):
				x = self.residual_layer(x,is_training=is_training,scope='residual_'+str(i+1))
				input.append(x)
				x = tf.concat(input,axis=3)
		return x


	def residual_layer(self,x,stride=1,is_training=True,scope='residual'):
		with tf.variable_scope(scope):
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='rbn1'))
			x = slim.conv2d(x,4*self.filters,[1,1],1,padding='SAME',activation_fn=None,scope='rconv1')
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='rbn2'))
			x = slim.conv2d(x,self.filters,[3,3],1,padding='SAME',activation_fn=None,scope='rconv2')
		return x


	def trans_layer(self,x,compress_rate=0.5,is_training=True,scope='transition'):
		with tf.variable_scope(scope):
			x = tf.nn.relu(self.batch_norm(x,is_training,scope='trans_bn'))
			x = slim.conv2d(x,int(compress_rate*x.shape[3].value),[1,1],1,padding='SAME',activation_fn=None,scope='trans_conv')
			x = slim.avg_pool2d(x,[2,2],stride=2,padding='SAME',scope='avg_pool')
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


def model_size():
    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size

if __name__ == '__main__':
	images = tf.placeholder(tf.float32, [10,256,256,3], name='images')
	ground_truth = tf.placeholder(tf.float32,[10,106,2],name = 'ground_truth')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		resnet = DenseNet(images,ground_truth)
		print(resnet.landmark_output.shape)
		print(model_size())




