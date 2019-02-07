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
output_dir = './img_output'
max_step = 50000

def create_placeholders():
    noise = tf.placeholder(tf.float32, shape=(None, noise_dim))
    input_image = tf.placeholder(tf.float32, shape=(None, output_dim, output_dim, 3))
    return noise, input_image

def compute_loss(noise, input_image):
    generator = Generator(input_noise_size=noise_dim, output_size=(output_dim, output_dim))
    discriminator = Discriminator()

    generated_image = generator(noise)
    input_image_dis = discriminator(input_image)
    generated_image_dis = discriminator(generated_image)
    D_loss = tf.reduce_mean(input_image_dis) - tf.reduce_mean(generated_image_dis)
    G_loss = tf.reduce_mean(generated_image_dis)
    return G_loss, D_loss, generator, discriminator, generated_image

def get_ops(geneator, discriminator):
    clip_op = discriminator.clip_weights_op()
    D_optim = tf.train.RMSPropOptimizer(learning_rate=0.00002).minimize(D_loss, var_list=discriminator.vars())
    G_optim = tf.train.RMSPropOptimizer(learning_rate=0.00002).minimize(G_loss, var_list=generator.vars())
    return G_optim, D_optim, clip_op

def get_data():
    data_loader = DataLoader(input_dir)
    dataset = data_loader.initialize_dataset(batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()
    return images


with tf.device('GPU:0'):
    noise, input_image = create_placeholders()
    G_loss, D_loss, generator, discriminator, generated_image = compute_loss(noise, input_image)
    G_optim, D_optim, clip_op = get_ops(generator, discriminator)
    images = get_data()

def train_generator(sess):
    z = np.random.uniform(0,1,(batch_size, noise_dim))
    image_batch = sess.run(images)
    gl, _ = sess.run([G_loss, G_optim], feed_dict={noise:z, input_image:image_batch})
    return gl

def train_discriminator(sess):
    z = np.random.uniform(0,1,(batch_size, noise_dim))
    image_batch = sess.run(images)
    dl, _, _= sess.run([D_loss, D_optim, clip_op], feed_dict={noise:z, input_image:image_batch})
    return dl

def generate_image(sess):
    z = np.random.uniform(0,1,(batch_size, noise_dim))
    g_image = sess.run(generated_image, {noise:z})[0,:,:,:]
    im.io.imsave(os.path.join(output_dir, '{}.jpg'.format(i)), g_image)

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
    for i in range(200):
        dl = train_discriminator(sess)
        print("Discriminator Loss: {}".format(dl))
    for i in range(max_step):
        d_loss = []
        g_loss = []
        for _ in range(k_d):
            dl = train_discriminator(sess)
            d_loss.append(dl)
        for _ in range(k_g):
            gl = train_generator(sess)
            g_loss.append(gl)
        k_g, k_d = update_train_control_variables(g_loss, d_loss, k_tot)
        if i%10==0:
            generate_image(sess)