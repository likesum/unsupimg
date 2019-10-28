#!/usr/bin/env python3

import os
import argparse
from imageio import imread, imsave

from utils import proc
import utils.utils as ut
from utils import pix_stack as net

import numpy as np
import tensorflow as tf

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', default=10, type=int, help='compressing ration %')
parser.add_argument('--path', default='wts/unsupervised_ratio10', help='path to pretrained model')
parser.add_argument('--data', default='BSD68', help='which data to use')
parser.add_argument('--outpath', default=None, help='where to save predictions')
opts = parser.parse_args()

ratio = opts.ratio / 100.0
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

VLIST = 'data/%s.txt'%opts.data

PSZ = 33

# Setup Graphs
is_training = tf.placeholder_with_default(False, shape=[])
model = net.Net(is_training)

img = tf.placeholder(shape=[None,None], dtype=tf.float32)

imgs = img[tf.newaxis,:,:,tf.newaxis]
imgs = tf.space_to_depth(imgs, PSZ)
imsp = tf.shape(imgs)
imgs = tf.reshape(imgs, [-1,PSZ,PSZ,1])

mat = proc.load_mat(ratio)
signal, proxy = proc.compress(imgs, mat)

recon = model.generate(proxy)
recon = tf.reshape(recon, imsp)
recon = tf.depth_to_space(recon, PSZ)
recon = tf.squeeze(recon)

#########################################################################
# Start TF session (respecting OMP_NUM_THREADS)
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))
sess.run(tf.global_variables_initializer())

#########################################################################
# Load latest model
print("Restoring model from " + mfile )
ut.loadNet(mfile,model.weights,sess)
print("Done!")

mses, psnrs = [], []
for nm in open(VLIST, 'r'):
    nm = nm.strip()
    if 'foreman' in nm or 'Parrots' in nm:
        image = np.float32(imread(nm)[:,:,0]) / 65535.
        h, w = image.shape[:2]
    else:
        image = np.float32(imread(nm)) / 255.
        h, w = image.shape

    h_pad, w_pad = (PSZ-h%PSZ)%PSZ, (PSZ-w%PSZ)%PSZ
    image = np.pad(image, ((0,h_pad), (0,w_pad)), 'constant')

    pred = sess.run(recon, feed_dict={img: image})

    image = image[:h,:w]
    pred = pred[:h,:w]

    mse = np.mean((pred-image)**2)
    psnr = -10*np.log10(mse)
    print('%s: %.3f dB'%(os.path.basename(nm), psnr))

    if outpath is not None:
        out = np.clip(pred, 0., 1.)
        nm = '_'.join(nm.split(os.path.sep)[-2:])
        imsave('%s/%s'%(outpath,nm), np.uint8(out*255.))

    mses.append(mse)
    psnrs.append(psnr)

print('\n Mean PSNR: %.3f dB'%np.mean(psnrs))





