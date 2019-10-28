from imageio import imread, imsave
import numpy as np
import tensorflow as tf
import os
from glob import glob

SIZE = 128

def crop(labels):
    face = labels[1]
    height, width = face.shape[:2]
    ys, xs = np.nonzero(face>0.1)
    ly, hy = np.min(ys), np.max(ys)
    lx, hx = np.min(xs), np.max(xs)

    # Include a little bit more of neck
    hy = hy + 0.1*(hy-ly)
    hy = np.minimum(height, int(hy))

    h, w = hy-ly, hx-lx
    if h > w:
        pad = (h-w)//2
        if lx-pad < 0:
            lx = 0
            hx = np.minimum(width, lx+h)
        elif hx+pad > width:
            hx = width
            lx = np.maximum(0, hx-h)
        else:
            lx = lx-pad
            hx = lx+h

    elif h < w:
        pad = (w-h)//2
        if ly-pad < 0:
            ly = 0
            hy = np.minimum(height, ly+w)
        elif hy+pad > height:
            hy = height
            ly = np.maximum(0, hy-w)
        else:
            ly = ly-pad
            hy = ly+w
    
    return ly, hy, lx, hx

_ph = tf.placeholder(shape=[None,None,None], dtype=tf.float32)
_out = tf.image.resize_images(_ph, [SIZE,SIZE])
_sess = tf.Session()
def tf_resize(imgs, labels=False):
    resized = _sess.run(_out, feed_dict={_ph: imgs})
    return resized

nms = glob('data/Helen_segmentation/images/*.jpg')
nms = [os.path.basename(nm).replace('.jpg','') for nm in nms]
outdir = 'data/Helen_segmentation/cropped/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

for nm in nms:
    rgb = np.float32(imread('data/Helen_segmentation/images/%s.jpg'%nm)) / 255.0
    labels = []
    for i in range(11):
        label = imread('data/Helen_segmentation/labels/%s/%s_lbl%02d.png'%(nm, nm, i))
        label = np.float32(label) / 255.0
        labels.append(label)
    ly, hy, lx, hx = crop(labels)
    labels = np.stack(labels, axis=2)

    resized_rgb = tf_resize(rgb[ly:hy, lx:hx])

    resized_rgb = np.uint8(resized_rgb*255.0)
    imsave(outdir+'%s.png'%nm, resized_rgb)












