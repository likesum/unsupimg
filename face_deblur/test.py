#!/usr/bin/env python3
import importlib, argparse
import glob

import tensorflow as tf
import numpy as np
from utils import utils as ut
from utils import pix as net
from imageio import imsave, imread

import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='wts/blind', help='path to trained model')
parser.add_argument('--data', default='data/Test_data_Helen', help='path to data')
parser.add_argument('--outpath', default=None, help='where to save predictions')
opts = parser.parse_args()

if opts.path.endswith('.npz'):
    mfile = opts.path
else:
    wts = opts.path
    msave = ut.ckpter(wts + '/iter_*.model.npz')
    mfile = msave.latest

outpath = opts.outpath
if outpath is not None:
    if not os.path.exists(outpath):
        os.makedirs(outpath)

def csave(fn,img):
    img = np.maximum(0.,np.minimum(1.,img))
    img = np.uint8(img*255.)
    imsave(fn,img)

VLIST = opts.data
nms = glob.glob('%s/*_gt/*.png'%opts.data)
nms = [l.replace('.png', '') for l in nms]
BSZ = 80
    
#### Build Graph
names, blurs = [], []
for i in range(BSZ):
    nm = tf.placeholder(tf.string)
    blur = tf.read_file(nm)
    blur = tf.image.decode_png(blur, channels=3, dtype=tf.uint8)
    blur = tf.to_float(blur) / 255.
    names.append(nm)
    blurs.append(blur)
blurs = tf.stack(blurs, axis=0)

is_training = tf.placeholder_with_default(False, shape=[])
model = net.Net(is_training)
deblur = blurs + model.generate(blurs)

# Create session
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
sess.run(tf.global_variables_initializer())

# Load model
print("Restoring model from " + mfile )
ut.loadNet(mfile,model.weights,sess)
print("Done!")

loss, avgpsnr = 0., 0.
for c, nm in enumerate(nms):
    gt = np.float32(imread(nm+'.png')) / 255.0
    nm = nm.replace('_gt', '_blur')

    fd, k = {}, 0
    for i in range(1, 11):
        for j in range(13, 29, 2):
            fd[names[k]] = '%s_ker%02d_blur_k%d.png' % (nm, i, j)
            k = k + 1

    noisy, preds = sess.run([blurs, deblur], feed_dict=fd)

    mse = np.mean((gt[np.newaxis]-preds)**2, axis=(1,2,3))
    psnr = np.mean(-10.*np.log10(mse))
    loss = loss + np.mean(mse)
    avgpsnr = avgpsnr + psnr

    if c % 10 == 0:
        print('Finish %d/%d images' % (c, len(nms)))

    if outpath is not None:
        k = 0
        for i in range(1, 11):
            for j in range(13, 29, 2):
                outnm = '%s_ker%02d_blur_k%d.png' % (os.path.basename(nm), i, j)
                csave('%s/%s'%(outpath,outnm), preds[k])
                k = k + 1

mse = loss / len(nms)
psnr = avgpsnr / len(nms)
print("MSE: %.4f"%mse)
print("PSNR: %.2f"%psnr)











