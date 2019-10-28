import tensorflow as tf
from utils import proc


def swap(left_blurs, right_blurs, left_deblurs, right_deblurs, left_kernels, right_kernels, nstd=0):
    left_swap = proc.gen_blur(right_deblurs, left_kernels, nstd=nstd, seeds=None)
    right_swap = proc.gen_blur(left_deblurs, right_kernels, nstd=nstd, seeds=None)
    swap_loss = tf.reduce_mean(tf.abs(left_swap-left_blurs)) \
        + tf.reduce_mean(tf.abs(right_swap-right_blurs))
    return swap_loss


def rec(left_blurs, right_blurs, left_deblurs, right_deblurs, left_kernels, right_kernels, nstd=0):
    left_rec = proc.gen_blur(left_deblurs, left_kernels, nstd=nstd, seeds=None)
    right_rec = proc.gen_blur(right_deblurs, right_kernels, nstd=nstd, seeds=None)
    rec_loss = tf.reduce_mean(tf.abs(left_rec-left_blurs)) \
        + tf.reduce_mean(tf.abs(right_rec-right_blurs))
    return rec_loss


def get_loss(left_blurs, right_blurs, left_deblurs, right_deblurs, left_kernels, right_kernels, nstd=0):
    rec_loss = rec(left_blurs, right_blurs, left_deblurs, right_deblurs, left_kernels, right_kernels, nstd)
    swap_loss = swap(left_blurs, right_blurs, left_deblurs, right_deblurs, left_kernels, right_kernels, nstd)
    loss = rec_loss + swap_loss
    return loss, [rec_loss, swap_loss], ['rec_loss', 'swap_loss']



