import tensorflow as tf

'''
Training Stage:
    Generator Input Size: 75 * 75
    Generator Output Size: 600 * 600
Test Stage:
    Generator Input Size: None * None
    Generator Output Size: None*8 * None*8
'''
class Generator(object):
    def __init__(self, input_noise_size=100, output_size=(128, 128)):
        self.input_noise_size = input_noise_size
        self.output_size = output_size
        assert output_size[0]%16==0 and output_size[1]%16==0, 'Height and Width of output need to be multiples of 8'
        self.fc_size = output_size[0]//16 * output_size[1]//16
    
    def upsample(self, tensor, filter=64, kernel_size=(4,4), strides=(2,2)):
        tensor = tf.layers.conv2d_transpose(tensor, filters=filter, kernel_size=kernel_size, strides=strides, padding='same', activation=tf.nn.leaky_relu)
        return tensor
    
    def __call__(self, noise):
        with tf.variable_scope('generator', reuse=False):
            tensor = tf.layers.dense(noise, self.fc_size)
            tensor = tf.reshape(tensor, shape=(-1, self.output_size[0]//16, self.output_size[1]//16, 1))

            tensor = self.upsample(tensor, filter=1024)
            tensor = self.upsample(tensor, filter=512)
            tensor = self.upsample(tensor, filter=256)
            tensor = self.upsample(tensor, filter=128)
            tensor = tf.layers.conv2d(tensor, filters=3, kernel_size=(3,3), padding='same', activation=tf.nn.tanh)
            return tensor
    
    def vars(self):
        return tf.trainable_variables(scope='generator')

'''
VGG-16
'''
class Discriminator(object):
    def __init__(self):
        pass

    def __call__(self, image, reuse=tf.AUTO_REUSE):
        tensor = image
        base_channel = 8
        self.weights = []
        self.biases = []
        w_init = tf.random_normal_initializer(stddev=0.02)
        with tf.variable_scope('discriminator', reuse=reuse):
            tensor = tf.layers.conv2d(tensor, filters=base_channel, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.conv2d(tensor, filters=base_channel, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.max_pooling2d(tensor, pool_size=(2,2), strides=(2,2), padding='same')
            tensor = tf.layers.conv2d(tensor, filters=base_channel*2, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.conv2d(tensor, filters=base_channel*2, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.max_pooling2d(tensor, pool_size=(2,2), strides=(2,2), padding='same')
            tensor = tf.layers.conv2d(tensor, filters=base_channel*4, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.conv2d(tensor, filters=base_channel*4, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.conv2d(tensor, filters=base_channel*4, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.max_pooling2d(tensor, pool_size=(2,2), strides=(2,2), padding='same')
            tensor = tf.layers.conv2d(tensor, filters=base_channel*8, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.conv2d(tensor, filters=base_channel*8, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.conv2d(tensor, filters=base_channel*8, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.max_pooling2d(tensor, pool_size=(2,2), strides=(2,2), padding='same')
            tensor = tf.layers.conv2d(tensor, filters=base_channel*8, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.conv2d(tensor, filters=base_channel*8, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.conv2d(tensor, filters=base_channel*8, kernel_size=(3,3), padding='same', kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.max_pooling2d(tensor, pool_size=(2,2), strides=(2,2), padding='same')
            tensor = tf.layers.flatten(tensor)
            #prev = 16*base_channel*8
            # for i in range(3):
            #     weight = tf.get_variable('w{}'.format(i), shape=(prev, base_channel*64), dtype=tf.float32)
            #     bias = tf.get_variable('b{}'.format(i), shape=(base_channel*64,), dtype=tf.float32)
            #     prev = base_channel*64
            #     tensor = tf.nn.relu(tf.matmul(tensor, weight) + bias)
            #     self.weights.append(weight)
            #     self.biases.append(bias)
            tensor = tf.layers.dense(tensor, units=base_channel*64, kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.dense(tensor, units=base_channel*64, kernel_initializer=w_init, activation=tf.nn.relu)
            tensor = tf.layers.dense(tensor, units=base_channel*64, kernel_initializer=w_init, activation=tf.nn.relu)
            
            # Output
            tensor = tf.layers.dense(tensor, units=1, kernel_initializer=w_init, activation=None)
            # weight = tf.get_variable('w_out', shape=(prev, 3), dtype=tf.float32)
            # bias = tf.get_variable('b_out', shape=(3,), dtype=tf.float32)
            # tensor = tf.matmul(tensor, weight) + bias
            # self.weights.append(weight)
            # self.biases.append(bias)
            return tensor
    
    def clip_weights_op(self):
        clip_op = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.vars()]
        #clip_op.extend([p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.biases])
        return clip_op
        
    def vars(self):
        return tf.trainable_variables(scope='discriminator')