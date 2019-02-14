import skimage as im
import os

input_dir = './img_resized'
output_dir = './img_resized2'

files = os.listdir(input_dir)
for i in files:
    img = im.io.imread(os.path.join(input_dir, i))
    img_resized = im.transform.resize(img, (117, 117), anti_aliasing=True)
    im.io.imsave(os.path.join(output_dir, i), img_resized)