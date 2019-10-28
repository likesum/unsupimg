import tensorflow as tf
from utils import proc

def swap(left_signal, right_signal, left_recon, right_recon, shifts, mat, csz=33):
    left_swap, right_swap = [], []
    left_recon, right_recon = tf.unstack(left_recon), tf.unstack(right_recon)
    for i in range(len(left_recon)):
        dy, dx = shifts[i,0], shifts[i,1]

        left = tf.concat([left_recon[i][:dy, dx:], right_recon[i][:-dy, :-dx]], axis=0)
        left = tf.concat([left_recon[i][:, :dx], left], axis=1)
        left_swap.append(left)

        right = tf.concat([left_recon[i][dy:, dx:], right_recon[i][-dy:, :-dx]], axis=0)
        right = tf.concat([right, right_recon[i][:, -dx:]], axis=1)
        right_swap.append(right)

    left_swap = proc.extract_blocks(tf.stack(left_swap), csz)
    right_swap = proc.extract_blocks(tf.stack(right_swap), csz)

    left_swap = proc.compress(left_swap, mat, proxy=False)
    right_swap = proc.compress(right_swap, mat, proxy=False)
    swap_loss = tf.reduce_mean((left_swap-left_signal)**2.) \
        + tf.reduce_mean((right_swap-right_signal)**2.)

    return swap_loss


def rec(left_signal, right_signal, left_recon, right_recon, mat, csz=33):
    left_recon = proc.extract_blocks(left_recon, csz)
    right_recon = proc.extract_blocks(right_recon, csz)
    left_recon = proc.compress(left_recon, mat, proxy=False)
    right_recon = proc.compress(right_recon, mat, proxy=False)
    rec_loss = tf.reduce_mean((left_recon-left_signal)**2.) \
        + tf.reduce_mean((right_recon-right_signal)**2.)

    return rec_loss


def get_loss(left_signal, right_signal, left_recon, right_recon, shifts, mat, csz=33):
    rec_loss = rec(left_signal, right_signal, left_recon, right_recon, mat, csz)
    swap_loss = swap(left_signal, right_signal, left_recon, right_recon, shifts, mat, csz)
    return rec_loss, swap_loss




