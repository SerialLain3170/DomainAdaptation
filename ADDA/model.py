import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers

xp=cuda.cupy
cuda.get_device(0).use()

class CBR(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.GlorotUniform()
        super(CBR, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1,initialW=w)
            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)

    def __call__(self, x):
        h = F.relu(self.bn0(self.c0(x)))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.average_pooling_2d(h,3,2,1)

        return h

class Encoder(Chain):
    def __init__(self, base=64):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.cbr0 = CBR(3, base)
            self.cbr1 = CBR(base, base*2)
            self.cbr2 = CBR(base*2, base*4)
            self.cbr3 = CBR(base*4, base*8)
            self.l0 = L.Linear(None, 1024)

    def __call__(self, x):
        h = self.cbr0(x)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = self.cbr3(h)
        h = self.l0(h)

        return h

class Classification(Chain):
    def __init__(self):
        super(Classification, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(1024, 10)

    def __call__(self, x):
        h = self.l0(x)

        return h