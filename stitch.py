from skimage.io import imread, imsave
import os
import numpy as np

input_dir = './img_output_Set1'

files = [os.path.join(input_dir, x) for x in sorted(os.listdir(input_dir), key=lambda k: int(k.split('.')[0])) if x.endswith('.jpg')]

rows = []
for i in range(20):
    row = np.concatenate([imread(x) for x in files[i*20:i*20+20]], axis=1)
    rows.append(row)

img = np.concatenate(rows, axis=0)
imsave('collage.png', img)