import numpy as np 
import tensorflow as tf

istest = False

# Remove BNs (use after replacing filters with popstats)    
def toTest():
    global istest
    istest = True

def batch_norm(net, name, inp, decay=0.99, epsilon=1e-8):

    if name+'_mu' not in net.weights:
        pop_mean = tf.Variable(tf.zeros([inp.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inp.get_shape()[-1]]), trainable=False)
        net.weights[name+'_mu'] = pop_mean
        net.weights[name+'_var'] = pop_var
    else:
        pop_mean = net.weights[name+'_mu']
        pop_var = net.weights[name+'_var']

    batch_mean, batch_var = tf.nn.moments(inp,[0,1,2])
    mean = tf.cond(net.is_training, lambda: batch_mean, lambda: pop_mean)
    var = tf.cond(net.is_training, lambda: batch_var, lambda: pop_var)
    out = tf.nn.batch_normalization(inp, mean, var, None, None, epsilon)

    # Add ops to update moving average
    meanOp = tf.assign(pop_mean,
                           pop_mean * decay + batch_mean * (1 - decay)).op
    varOp = tf.assign(pop_var,
                          pop_var * decay + batch_var * (1 - decay)).op
    net.updateOps += [meanOp, varOp]

    return out

def conv(net, name, inp, outch, ksz=4, stride=2, bn=1, relu=True, pad='SAME'):
    inch = inp.get_shape().as_list()[-1]
    ksz = [ksz,ksz,inch,outch]

    wnm = name + "_w"
    if wnm in net.weights.keys():
        w = net.weights[wnm]
    else:
        sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[2]))
        w = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
        net.weights[wnm] = w
        net.wd = net.wd + tf.nn.l2_loss(w)

    out = tf.nn.conv2d(inp,w,[1,stride,stride,1],pad)

    if bn==1 and not istest:
        out = batch_norm(net, name, out)

    bnm = name + "_b"
    if bnm in net.weights.keys():
        b = net.weights[bnm]
    else:
        b = tf.Variable(tf.constant(0,shape=[ksz[-1]],dtype=tf.float32))
        net.weights[bnm] = b
    out = out+b

    if relu:
        out = tf.nn.relu(out)

    return out


def dconv(net, name, inp, outch, outsz, ksz=4, stride=2, bn=1, relu=True, pad='SAME'):
    inch = inp.get_shape().as_list()[-1]
    bsz = inp.get_shape().as_list()[0]
    if bsz is None:
        bsz = tf.shape(inp)[0]
    outsp = tf.stack([bsz,outsz,outsz,outch])
    ksz = [ksz,ksz,outch,inch]

    wnm = name + "_w"
    if wnm in net.weights.keys():
        w = net.weights[wnm]
    else:
        sq = np.sqrt(3.0 / np.float32(ksz[0]*ksz[1]*ksz[3]))
        w = tf.Variable(tf.random_uniform(ksz,minval=-sq,maxval=sq,dtype=tf.float32))
        net.weights[wnm] = w
        net.wd = net.wd + tf.nn.l2_loss(w)

    out = tf.nn.conv2d_transpose(inp, w, output_shape=outsp, strides=[1,stride,stride,1], padding=pad)

    if bn==1 and not istest:
        out = batch_norm(net, name, out)
        
    bnm = name + "_b"
    if bnm in net.weights.keys():
        b = net.weights[bnm]
    else:
        b = tf.Variable(tf.constant(0,shape=[ksz[2]],dtype=tf.float32))
        net.weights[bnm] = b
    out = out+b

    if relu:
        out = tf.nn.relu(out)

    return out


class Net:
    def __init__(self, is_training):
        self.is_training = is_training
        self.weights = {}
        self.updateOps = []
        self.wd = 0.


    def generate(self, proxy, nch=32, prefix='', imsz=33, intermedia=False):
        if prefix != '':
            prefix = prefix+'_'

        out1 = self.unet(proxy, nch, prefix+'u1_', imsz)
        out = tf.concat([proxy, out1], axis=-1)
        res = self.unet(out, nch, prefix+'u2_', imsz)
        return out1 + res

    def unet(self, proxy, nch=32, prefix='', imsz=33):
        if prefix != '':
            prefix = prefix+'_'

        out = proxy

        # Assume image is (33 x 33 x dim)
        nfeat = nch
        e1 = conv(self, prefix+'econv_1', out, nfeat, ksz=2, stride=1, pad='VALID')
        e2 = conv(self, prefix+'econv_2', e1, 2*nfeat)
        e3 = conv(self, prefix+'econv_3', e2, 4*nfeat)
        e4 = conv(self, prefix+'econv_4', e3, 8*nfeat)
        e5 = conv(self, prefix+'econv_5', e4, 8*nfeat)
        e6 = conv(self, prefix+'econv_6', e5, 8*nfeat)

        d1 = dconv(self, prefix+'dconv_1', e6, 8*nfeat, 2)
        d1 = tf.concat([d1,e5], axis=-1)
        d2 = dconv(self, prefix+'dconv_2', d1, 8*nfeat, 4)
        d2 = tf.concat([d2,e4], axis=-1)
        d3 = dconv(self, prefix+'dconv_3', d2, 4*nfeat, 8)
        d3 = tf.concat([d3,e3], axis=-1)
        d4 = dconv(self, prefix+'dconv_4', d3, 2*nfeat, 16)
        d4 = tf.concat([d4,e2], axis=-1)
        d5 = dconv(self, prefix+'dconv_5', d4, nfeat, 32)
        d5 = tf.concat([d5,e1], axis=-1)
        out = dconv(self, prefix+'dconv_6', d5, nfeat, 33, ksz=2, stride=1, pad='VALID')

        out = conv(self, prefix+'end_1', out, nfeat, ksz=3, stride=1)
        out = conv(self, prefix+'end_2', out, 1, ksz=1, stride=1, bn=0, relu=False)
        return out



        




