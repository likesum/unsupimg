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
    # bsz = tf.shape(inp)[0]
    # outsp = [bsz,outsz,outsz,outch]
    # outsp = tf.concat([bsz, tf.stack([outsz,outsz,outch])], axis=0)
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

    def generate(self, img, nch=64):
        
        # Assume image is (128 x 128 x dim)
        nfeat = nch

        e1 = conv(self, 'econv_1', img, nfeat) # 64 x 64 x nfeat
        e2 = conv(self, 'econv_2', e1, 2*nfeat) # 32 x 32 x nfeat
        e3 = conv(self, 'econv_3', e2, 4*nfeat) # 16 x 16 x nfeat
        e4 = conv(self, 'econv_4', e3, 8*nfeat) # 8 x 8 x nfeat
        e5 = conv(self, 'econv_5', e4, 8*nfeat) # 4 x 4 x nfeat
        e6 = conv(self, 'econv_6', e5, 8*nfeat) # 2 x 2 x nfeat
        e7 = conv(self, 'econv_7', e6, 8*nfeat) # 1 x 1 x nfeat

        self.d1 = dconv(self, 'dconv_1', e7, 8*nfeat, 2)
        d1 = tf.concat([self.d1,e6], axis=-1)
        self.d2 = dconv(self, 'dconv_2', d1, 8*nfeat, 4)
        d2 = tf.concat([self.d2,e5], axis=-1)
        self.d3 = dconv(self, 'dconv_3', d2, 8*nfeat, 8)
        d3 = tf.concat([self.d3,e4], axis=-1)
        self.d4 = dconv(self, 'dconv_4', d3, 4*nfeat, 16)
        d4 = tf.concat([self.d4,e3], axis=-1)
        self.d5 = dconv(self, 'dconv_5', d4, 2*nfeat, 32)
        d5 = tf.concat([self.d5,e2], axis=-1)
        self.d6 = dconv(self, 'dconv_6', d5, nfeat, 64)
        d6 = tf.concat([self.d6,e1], axis=-1)

        out = dconv(self, 'dconv_7', d6, 3, 128, bn=False, relu=False)

        # Generate kernel
        k1 = dconv(self, 'kconv_1', e7, 8*nfeat, 2)
        k1 = tf.concat([k1,e6], axis=-1)
        k2 = dconv(self, 'kconv_2', k1, 8*nfeat, 4)
        k2 = tf.concat([k2,e5], axis=-1)
        k3 = dconv(self, 'kconv_3', k2, 8*nfeat, 8)
        k3 = tf.concat([k3,e4], axis=-1)
        k4 = dconv(self, 'kconv_4', k3, 4*nfeat, 16)
        k4 = tf.concat([k4,e3], axis=-1)
        k5 = dconv(self, 'kdconv_5', k4, 2*nfeat, 19, ksz=4, stride=1, pad='VALID')
        k6 = dconv(self, 'kdconv_6', k5, nfeat, 22, ksz=4, stride=1, pad='VALID')
        k7 = dconv(self, 'kdconv_7', k6, nfeat, 25, ksz=4, stride=1, pad='VALID')
        k8 = dconv(self, 'kdconv_8', k7, nfeat, 27, ksz=3, stride=1, pad='VALID')

        kout = conv(self, 'kconv_end', k8, 1, ksz=1, stride=1, bn=False, relu=False)
        kout = tf.reshape(kout, [-1, 27*27, 1])
        kout = tf.nn.softmax(kout, axis=1)
        kout = tf.reshape(kout, [-1,27,27,1])

        return out, kout



        




