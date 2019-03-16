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
            self.c0 = L.Convolution2D(in_ch, out_ch, ksize,1,initialW=w)

    def __call__(self, x):
        h = F.relu(self.c0(x))
        h = F.average_pooling_2d(h,3,2,1)

        return h

class Encoder(Chain):
    def __init__(self, in_ch, base=8):
        w = initializers.HeNormal()
        super(Encoder, self).__init__()
        with self.init_scope():
            self.cbr0 = CBR(in_ch, base, ksize=5)
            self.cbr1 = CBR(base, base*2, ksize=5)
            self.cbr2 = CBR(base*2, base*15, ksize=4)
            self.l0 = L.Linear(None, 500, initialW=w)

    def __call__(self, x):
        h = self.cbr0(x)
        h = self.cbr1(h)
        h = F.dropout(self.cbr2(h), ratio=0.5)
        h = self.l0(h)

        return F.dropout(F.relu(h), ratio=0.5)

class Classification(Chain):
    def __init__(self):
        super(Classification, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l0 = L.Linear(500, 10, initialW=w)

    def __call__(self, x):
        h = self.l0(x)

        return h

class Discriminaor(Chain):
    def __init__(self):
        super(Discriminaor, self).__init__()
        w = initializers.HeNormal()
        with self.init_scope():
            self.l0 = L.Linear(500, 500, initialW=w)
            self.l1 = L.Linear(500, 500, initialW=w)
            self.l2 = L.Linear(500, 2, initialW=w)

    def __call__(self, x):
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = self.l2(h)

        return h