#!/usr/bin/env python3

import os
import argparse

from utils import proc
import numpy as np
import tensorflow as tf

from utils import pix_stack as net
from utils.dataset import Dataset, tvSwap
import utils.utils as ut

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', default=10, type=int, help='compressing ratio %')
opts = parser.parse_args()

ratio = opts.ratio / 100.0
wts = 'wts/supervised_ratio%d'%(opts.ratio)

#########################################################################
TLIST='data/train.txt'
VLIST='data/dev.txt'

BSZ = 7
IMSZ = 330
CSZ = 33  # block size

LR = 1e-3

drops = {
    1: (11e4, 12e4, 13e4),
    4: (11e4, 12e4, 13e4),
    10: (16e4, 17e4, 18e4)
}

def get_lr(niter):
    drop = drops[opts.ratio]
    if niter < drop[0]:
        return LR
    elif niter >= drop[0] and niter < drop[1]:
        return LR / np.sqrt(10.)
    else:
        return LR / 10.0
    return LR

VALFREQ = 2e2
SAVEFREQ = 1e4
MAXITER = drops[opts.ratio][-1]

if not os.path.exists(wts):
    os.makedirs(wts)
#########################################################################

# Check for saved weights & optimizer states
msave = ut.ckpter(wts + '/iter_*.model.npz')
ssave = ut.ckpter(wts + '/iter_*.state.npz')
ut.logopen(wts+'/train.log')
niter = msave.iter

#########################################################################

# Setup Graphs
is_training = tf.placeholder_with_default(False, shape=[])
model = net.Net(is_training)

# Images loading setup
tset = Dataset(TLIST, BSZ, IMSZ, CSZ, niter, all_blocks=True)
vset = Dataset(VLIST, BSZ, IMSZ, CSZ, 0, isval=True)
batch, swpT, swpV = tvSwap(tset, vset)
lefts, rights, shifts, seeds = batch

# Get compressing matrix, fixed
mat = proc.load_mat(ratio)

# Generate compressed images
left_signal, left_proxy = proc.compress(proc.extract_blocks(lefts, CSZ), mat)
right_signal, right_proxy = proc.compress(proc.extract_blocks(rights, CSZ), mat)

# Reconstruct, same model for left and right images
left_recon = model.generate(left_proxy)
right_recon = model.generate(right_proxy)

left_recon = proc.group_blocks(left_recon, IMSZ, CSZ)
right_recon = proc.group_blocks(right_recon, IMSZ, CSZ)

# MSE
mse1 = tf.reduce_mean(tf.squared_difference(lefts, left_recon), axis=(1,2,3))
mse2 = tf.reduce_mean(tf.squared_difference(rights, right_recon), axis=(1,2,3))
mse = tf.concat([mse1, mse2], axis=0)
psnr = tf.reduce_mean(-10. * tf.log(mse) / tf.log(10.0))
mse = tf.reduce_mean(mse)

loss = mse
lvals, lnms = [mse, psnr], ['mse', 'psnr']
tnms = [l+'.t' for l in lnms]
vnms = [l+'.v' for l in lnms]

# Set up optimizer
lr = tf.placeholder(shape=[], dtype=tf.float32)
opt = tf.train.AdamOptimizer(lr)
tStep = opt.minimize(loss, var_list=list(model.weights.values()))
tStep = tf.group([tStep]+model.updateOps)

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
# Load saved weights if any
if niter > 0:
    mfn = wts+"/iter_%06d.model.npz" % niter
    sfn = wts+"/iter_%06d.state.npz" % niter

    ut.mprint("Restoring model from " + mfn )
    ut.loadNet(mfn,model.weights,sess)
    ut.mprint("Restoring state from " + sfn )
    ut.loadAdam(sfn,opt,model.weights,sess)
    ut.mprint("Done!")

#########################################################################
# Main Training loop

stop=False
ut.mprint("Starting from Iteration %d" % niter)
sess.run(tset.fetchOp,feed_dict=tset.fdict())

while niter < MAXITER and not ut.stop:

    ## Validate model every so often
    if niter % VALFREQ == 0:
        ut.mprint("Validating model")
        val_iter = vset.ndata // BSZ
        vloss, vset.niter = [], 0
        sess.run(vset.fetchOp,feed_dict=vset.fdict())
        for its in range(val_iter):
            sess.run(swpV)
            outs = sess.run(
                lvals+[vset.fetchOp],
                feed_dict={**vset.fdict(), is_training: False}
            )
            vloss.append(np.array(outs[:-1]))
        vloss = np.mean(np.stack(vloss, axis=0), axis=0)
        ut.vprint(niter, vnms, vloss.tolist())

    ## Run training step and print losses
    sess.run(swpT)
    if niter % 100 == 0:
        outs = sess.run(
            lvals+[tStep, tset.fetchOp],
            feed_dict={**tset.fdict(), lr: get_lr(niter), is_training: True}
        )
        ut.vprint(niter, tnms, outs[:-2])
        ut.vprint(niter, ['lr'], [get_lr(niter)])
    else:
        outs = sess.run(
            [loss, psnr, tStep, tset.fetchOp],
            feed_dict={**tset.fdict(), lr: get_lr(niter), is_training: True}
        )
        ut.vprint(niter, ['loss.t', 'psnr.t'], outs[:2])

    niter=niter+1
                    
    ## Save model weights if needed
    if SAVEFREQ > 0 and niter % SAVEFREQ == 0:
        mfn = wts+"/iter_%06d.model.npz" % niter
        sfn = wts+"/iter_%06d.state.npz" % niter

        ut.mprint("Saving model to " + mfn )
        ut.saveNet(mfn,model.weights,sess)
        ut.mprint("Saving state to " + sfn )
        ut.saveAdam(sfn,opt,model.weights,sess)
        ut.mprint("Done!")
        msave.clean(every=SAVEFREQ,last=1)
        ssave.clean(every=SAVEFREQ,last=1)


# Save last
if msave.iter < niter:
    mfn = wts+"/iter_%06d.model.npz" % niter
    sfn = wts+"/iter_%06d.state.npz" % niter

    ut.mprint("Saving model to " + mfn )
    ut.saveNet(mfn,model.weights,sess)
    ut.mprint("Saving state to " + sfn )
    ut.saveAdam(sfn,opt,model.weights,sess)
    ut.mprint("Done!")
    msave.clean(every=SAVEFREQ,last=1)
    ssave.clean(every=SAVEFREQ,last=1)
