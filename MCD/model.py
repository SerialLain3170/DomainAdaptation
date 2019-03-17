import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers

xp=cuda.cupy
cuda.get_device(0).use()

class CBR(Chain):
    def __init__(self, in_ch, out_ch, ksize):
        w = initializers.HeNormal()
        super(CBR, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, ksize,1,2, initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.relu(self.bn0(self.c0(x)))
        h = F.average_pooling_2d(h,3,2,1)

        return h

class Encoder(Chain):
    def __init__(self, in_ch, base=64):
        w = initializers.HeNormal()
        super(Encoder, self).__init__()
        with self.init_scope():
            self.cbr0 = CBR(in_ch, base, ksize=5)
            self.cbr1 = CBR(base, base, ksize=5)
            self.cbr2 = CBR(base, base*2, ksize=5)
            self.l0 = L.Linear(None, 3072, initialW=w)

    def __call__(self, x):
        h = self.cbr0(x)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.l0(h)

        return F.dropout(F.relu(h), ratio=0.5)

class Classification(Chain):
    def __init__(self):
        super(Classification, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l0 = L.Linear(3072, 2048, initialW=w)
            self.l2 = L.Linear(2048, 10, initialW=w)

            self.bn0 = L.BatchNormalization(2048)

    def __call__(self, x):
        h = F.leaky_relu(self.bn0(self.l0(x)))
        h = self.l2(h)

        return h