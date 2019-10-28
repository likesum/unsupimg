import re
import os
import time
import sys
import signal
from glob import glob
import numpy as np
import tensorflow as tf

import numpy as np
import os

# Logic for trapping ctrlc and setting stop
stop = False
_orig = None
def handler(a,b):
    global stop
    stop = True
    signal.signal(signal.SIGINT,_orig)
_orig = signal.signal(signal.SIGINT,handler)


# Log print
_log = None

def logopen(fn):
    global _log
    _log = open(fn,'a')

def mprint(s):
    sys.stdout.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    sys.stdout.flush()
    _log.write(time.strftime("%Y-%m-%d %H:%M:%S ") + s + "\n")
    _log.flush()

def vprint(it, nms, vals):
    s = '[%06d]' % it
    for i in range(len(nms)):
        s = s + ' ' + nms[i] + ' = %.3e'%vals[i]
    mprint(s)
    
# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest
class ckpter:
    def __init__(self,wcard):
        self.wcard = wcard
        self.load()
        
    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1)))
                 for l in lst]
            self.lst=sorted(lst,key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst=[]
            self.iter=0
            self.latest=None

    def clean(self,every=0,last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])


# Save/load networks
def saveNet(fn,dic,sess):
    wts = sess.run(dic)
    np.savez(fn,**wts)


def loadNet(fn,dic,sess):
    wts = np.load(fn)
    ops, fd = [], {}
    for k in dic.keys():
        ph = tf.placeholder(tf.float32)
        ops.append(dic[k].assign(ph).op)
        fd[ph] = wts[k]
    sess.run(ops, feed_dict=fd)


# Save/load Adam optimizer state
def saveAdam(fn,opt,vdict,sess):
    weights = {}
    beta1_power, beta2_power = opt._get_beta_accumulators()
    weights['b1p'] = beta1_power
    weights['b2p'] = beta2_power
    for nm in vdict.keys():
        v = vdict[nm]
        weights['m_%s' % nm] = opt.get_slot(v,'m')
        weights['v_%s' % nm] = opt.get_slot(v,'v')
    wts = sess.run(weights)
    np.savez(fn,**wts)


def loadAdam(fn,opt,vdict,sess):
    weights = np.load(fn)
    ph = tf.placeholder(tf.float32)
    beta1_power, beta2_power = opt._get_beta_accumulators()
    sess.run(beta1_power.assign(ph),
             feed_dict={ph: weights['b1p']})
    sess.run(beta2_power.assign(ph),
             feed_dict={ph: weights['b2p']})

    ops, fd = [], {}
    for nm in vdict.keys():
        v = vdict[nm]
        ph = tf.placeholder(tf.float32)
        ops.append(opt.get_slot(v,'m').assign(ph).op)
        fd[ph] = weights['m_%s' % nm]

        ph = tf.placeholder(tf.float32)
        ops.append(opt.get_slot(v,'v').assign(ph).op)
        fd[ph] = weights['v_%s' % nm]
    sess.run(ops, feed_dict=fd)
