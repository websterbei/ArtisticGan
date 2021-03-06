from MODEL import Generator, Discriminator
from DataLoader import DataLoader
import tensorflow as tf
import numpy as np
import skimage.io as io
import os

noise_dim = 100
batch_size = 100

input_dir = './img_resized2'
output_dir = './img_output'
max_step = 50000

def create_placeholders():
    noise = tf.placeholder(tf.float32, shape=(None, noise_dim))
    input_image = tf.placeholder(tf.float32, shape=(None, 28, 28))
    return noise, input_image

def compute_loss(noise, input_image):
    generator = Generator()
    discriminator = Discriminator()

    generated_image = generator(noise)
    input_image_dis = discriminator(input_image, reuse=False)
    generated_image_dis = discriminator(generated_image, reuse=True)

    epsilon = tf.random_uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)
    mixed_image = input_image + epsilon * (generated_image - input_image)
    mixed_image_dis = discriminator(mixed_image, reuse=True)
    gradient_mixed_image_dis = tf.gradients(mixed_image_dis, [mixed_image])[0]
    gradient_mixed_image_dis = tf.sqrt(tf.reduce_sum(tf.square(gradient_mixed_image_dis), axis=[1,2]))
    gradient_penalty = tf.reduce_mean((gradient_mixed_image_dis - 1.) ** 2)

    #D_loss = tf.reduce_mean(generated_image_dis) - tf.reduce_mean(input_image_dis) + 10.0 * gradient_penalty
    #G_loss = -tf.reduce_mean(generated_image_dis)

    D_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(input_image_dis)) + tf.log(1-tf.nn.sigmoid(generated_image_dis)))
    G_loss = -tf.reduce_mean(tf.log(tf.nn.sigmoid(generated_image_dis)))

    return G_loss, D_loss, generator, discriminator, generated_image

def get_ops(geneator, discriminator):
    clip_op = discriminator.clip_weights_op()
    D_optim = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(D_loss, var_list=discriminator.vars())
    G_optim = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(G_loss, var_list=generator.vars())
    return G_optim, D_optim, clip_op

with tf.device('GPU:0'):
    noise, input_image = create_placeholders()
    G_loss, D_loss, generator, discriminator, generated_image = compute_loss(noise, input_image)
    G_optim, D_optim, clip_op = get_ops(generator, discriminator)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def train_generator(sess):
    z = np.random.uniform(0,1,(batch_size, noise_dim))
    image_batch = mnist.train.next_batch(batch_size, shuffle=True)[0].reshape((batch_size, 28, 28))
    #image_batch = tf.image.resize_images(image_batch, [64, 64]).eval()
    gl, _ = sess.run([G_loss, G_optim], feed_dict={noise:z, input_image:image_batch})
    return gl

def train_discriminator(sess):
    z = np.random.uniform(0,1,(batch_size, noise_dim))
    image_batch = mnist.train.next_batch(batch_size, shuffle=True)[0].reshape((batch_size, 28, 28))
    #image_batch = tf.image.resize_images(image_batch, [64, 64]).eval()
    dl, _= sess.run([D_loss, D_optim], feed_dict={noise:z, input_image:image_batch})
    return dl

def generate_image(sess):
    z = np.random.uniform(0,1,(1, noise_dim))
    g_image = np.clip(sess.run(generated_image, {noise:z})[0,:,:], -1, 1)
    io.imsave(os.path.join(output_dir, '{}.jpg'.format(i)), g_image)

def update_train_control_variables(g_loss, d_loss, k_tot):
    d_loss_mean = sum(d_loss)/len(d_loss)
    g_loss_mean = sum(g_loss)/len(g_loss)
    del d_loss
    del g_loss
    print("Discriminator Loss: {},    Generator Loss: {}".format(d_loss_mean, g_loss_mean))
    k_d = min(k_tot, max(1, int(d_loss_mean/(d_loss_mean + g_loss_mean)*k_tot)))
    k_g = min(k_tot, max(1, int(g_loss_mean/(d_loss_mean + g_loss_mean)*k_tot)))
    return k_g, k_d

# Training control variables
k_d = 5
k_g = 5
k_tot = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        dl = train_discriminator(sess)
        print("Discriminator Loss: {}".format(np.mean(dl)))
    for i in range(max_step):
        d_loss = []
        g_loss = []
        for _ in range(1):
            dl = train_discriminator(sess)
            d_loss.append(dl)
        for _ in range(10):
            gl = train_generator(sess)
            g_loss.append(gl)
        print("Discriminator Loss: {}, Generator Loss: {}".format(np.mean(d_loss), np.mean(g_loss)))
        del d_loss
        del g_loss
        if i%50==0:
            generate_image(sess)