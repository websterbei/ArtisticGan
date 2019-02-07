import skimage as im
import os

input_dir = './img'
output_dir = './img_resized'

files = os.listdir(input_dir)
for i in files:
    img = im.io.imread(os.path.join(input_dir, i))
    img_resized = im.transform.resize(img, (600, 600), anti_aliasing=True)
    im.io.imsave(os.path.join(output_dir, i), img_resized)