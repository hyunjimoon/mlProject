# Day_09_02_ImageProcessing_2.py

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import PIL              # 이미지. Pillow
from PIL import Image

def natural_key(string_):
    # 1, 12, 123
    return int(re.search(r'\d+', string_).group())

def norm_image(img):
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()

    scale = np.max([np.abs(np.percentile(img_y_np, 1.0))])
    np.abs(np.percentile(img_y_np, 99.0))

    img_y_np /= scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np+1.0) / 2.0
    img_y_np = (img_y_np*255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)
    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))
    img_nrm = img_ybr.convert('RGB')

    return img_nrm

def resize_image(img, size):
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size*n_x/n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size*n_y/n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    img_pad = Image.new('RGB', (size, size), (0, 0, 0))
    top_left = ((size-n_x_new)//2, (size-n_y_new)//2)
    img_pad.paste(img_res, top_left)

    return img_pad


TRAIN_DIR = 'CatDog/train'
TEST_DIR  = 'CatDog/test'
SAVE_DIR  = 'CatDog/save'

# print(os.path.join(TRAIN_DIR, 'cat*.jpg'))
# print(*glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')),
#       sep='\n')

train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')), key=natural_key)
train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')), key=natural_key)
train_all = train_cats + train_dogs
print(*train_cats, sep='\n')

test_all = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')), key=natural_key)

# ---------------------------- #

SIZE = 224

path = train_all[1]
img  = Image.open(path)

img_nrm = norm_image(img)
img_res = resize_image(img_nrm, SIZE)

# plt.figure(figsize=(8,4))
# plt.subplot(131)
# plt.title('Original')
# plt.imshow(img)
#
# plt.subplot(132)
# plt.title('Normalized')
# plt.imshow(img_nrm)
#
# plt.subplot(133)
# plt.title('Resized')
# plt.imshow(img_res)
#
# plt.tight_layout()
# plt.show()

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

for path in train_all[:10]:
    img = Image.open(path)

    img_nrm = norm_image(img)
    img_res = resize_image(img_nrm, SIZE)

    new_path = os.path.join(SAVE_DIR, os.path.basename(path))
    img_res.save(new_path)
