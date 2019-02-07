import tensorflow as tf
import os

class DataLoader(object):
    def __init__(self, path):
        image_files = [os.path.join(path, x) for x in os.listdir(path) if x.endswith('.jpg')]
        self.images = tf.data.Dataset.from_tensor_slices(image_files)

    def initialize_dataset(self, batch_size=5):
        def input_parser(image_path):
            img_file = tf.read_file(image_path)
            return tf.image.convert_image_dtype(tf.image.decode_image(img_file, channels=3), dtype=tf.float32)
        return self.images.map(input_parser).repeat(100000).batch(batch_size)