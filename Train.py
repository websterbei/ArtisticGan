from MODEL import Generator, Discriminator
from DataLoader import DataLoader
import tensorflow as tf
import numpy as np
import skimage as im
import os

noise_dim = 16
output_dim = 128
batch_size = 5

input_dir = './img_resized'
#output_dir = '/media/webster/Data/img_output'
output_dir = './img_output'
max_step = 50000

def loss(logit, label):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)

with tf.device('GPU:0'):
    noise = tf.placeholder(tf.float32, shape=(None, noise_dim))
    input_image = tf.placeholder(tf.float32, shape=(None, output_dim, output_dim, 3))
    generator = Generator(input_noise_size=noise_dim, output_size=(output_dim, output_dim))
    generated_image = generator(noise)
    discriminator = Discriminator()

    input_image_dis = discriminator(input_image)
    generated_image_dis = discriminator(generated_image)
    #input_image_label = tf.ones_like(input_image_dis)
    #generated_image_label = tf.zeros_like(generated_image_dis)
    #D_loss = tf.reduce_mean(loss(input_image_dis, input_image_label) + loss(generated_image_dis, generated_image_label))
    #G_loss = tf.reduce_mean(loss(generated_image_dis, tf.ones_like(generated_image_dis)))
    D_loss = tf.reduce_mean(input_image_dis) - tf.reduce_mean(generated_image_dis)
    G_loss = tf.reduce_mean(generated_image_dis)

    generator_variables = tf.trainable_variables(scope='generator')
    discriminator_variables = tf.trainable_variables(scope='discriminator')

    clip_op = discriminator.clip_weights_op()
    D_optim = tf.train.RMSPropOptimizer(learning_rate=0.00002).minimize(D_loss, var_list=discriminator.vars())
    G_optim = tf.train.RMSPropOptimizer(learning_rate=0.00002).minimize(G_loss, var_list=generator.vars())


# Initialize data loader
data_loader = DataLoader(input_dir)
dataset = data_loader.initialize_dataset()
iterator = dataset.make_one_shot_iterator()
images = iterator.get_next()


k_d = 5
k_g = 5
k_tot = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(200):
        z = np.random.uniform(0,1,(batch_size, noise_dim))
        image_batch = sess.run(images)
        #dl, _, _= sess.run([D_loss, D_optim, clip_op], feed_dict={noise:z, input_image:image_batch})
        dl, _ = sess.run([D_loss, D_optim], feed_dict={noise:z, input_image:image_batch})
        print("Discriminator Loss: {}".format(dl))
    for i in range(max_step):
        d_loss = []
        g_loss = []
        for _ in range(k_d):
            z = np.random.uniform(0,1,(batch_size, noise_dim))
            image_batch = sess.run(images)
            #dl, _, _ = sess.run([D_loss, D_optim, clip_op], feed_dict={noise:z, input_image:image_batch})
            dl, _ = sess.run([D_loss, D_optim], feed_dict={noise:z, input_image:image_batch})
            d_loss.append(dl)
        for _ in range(k_g):
            z = np.random.uniform(0,1,(batch_size, noise_dim))
            image_batch = sess.run(images)
            gl, _ = sess.run([G_loss, G_optim], feed_dict={noise:z, input_image:image_batch})
            g_loss.append(gl)
        d_loss_mean = sum(d_loss)/len(d_loss)
        g_loss_mean = sum(g_loss)/len(g_loss)
        del d_loss
        del g_loss
        print("Discriminator Loss: {},    Generator Loss: {}".format(d_loss_mean, g_loss_mean))
        k_d = min(k_tot, max(1, int(d_loss_mean/(d_loss_mean + g_loss_mean)*k_tot)))
        k_g = min(k_tot, max(1, int(g_loss_mean/(d_loss_mean + g_loss_mean)*k_tot)))
        print("kd: {} kg:{}".format(k_d, k_g))
        if i%10==0:
            z = np.random.uniform(0,1,(batch_size, noise_dim))
            g_image = sess.run(generated_image, {noise:z})[0,:,:,:]
            im.io.imsave(os.path.join(output_dir, '{}.jpg'.format(i)), g_image)