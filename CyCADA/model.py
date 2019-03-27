import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import cuda,Chain,initializers

xp = cuda.cupy
cuda.get_device(0).use()

class CBR(Chain):
    def __init__(self,in_ch,out_ch,up=False,down=False,predict=False,activation=F.relu):
        super(CBR,self).__init__()
        w=initializers.Normal(0.02)
        self.up=up
        self.down=down
        self.activation=activation
        self.predict=predict
        with self.init_scope():
            self.cpara=L.Convolution2D(in_ch,out_ch,3,1,1,initialW=w)
            self.cdown=L.Convolution2D(in_ch,out_ch,4,2,1,initialW=w)

            self.bn0=L.BatchNormalization(out_ch)

    def __call__(self,x):
        if x.shape[0] == 1:
            if self.up:
                h=F.unpooling_2d(x,2,2,0,cover_all=False)
                h=self.activation(self.in0(self.cpara(h)))

            elif self.down:
                h=self.activation(self.in0(self.cdown(x)))

            else:
                h=self.activation(self.in0(self.cpara(x)))

        else:
            if self.up:
                h=F.unpooling_2d(x,2,2,0,cover_all=False)
                h=self.activation(self.bn0(self.cpara(h)))

            elif self.down:
                h=self.activation(self.bn0(self.cdown(x)))

            else:
                h=self.activation(self.bn0(self.cpara(x)))

        return h

class ResBlock(Chain):
    def __init__(self,in_ch,out_ch):
        super(ResBlock,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(in_ch,out_ch)
            self.cbr1=CBR(out_ch,out_ch)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)

        return h+x

class Generator(Chain):
    def __init__(self,base=32):
        super(Generator,self).__init__()
        w=initializers.Normal(0.02)
        with self.init_scope():
            self.c0=L.Convolution2D(3,base,7,1,3,initialW=w)
            self.cbr0=CBR(base,base*2,down=True)
            self.cbr1=CBR(base*2,base*4,down=True)
            self.res0=ResBlock(base*4,base*4)
            self.res1=ResBlock(base*4,base*4)
            self.res2=ResBlock(base*4,base*4)
            self.res3=ResBlock(base*4,base*4)
            self.res4=ResBlock(base*4,base*4)
            self.res5=ResBlock(base*4,base*4)
            self.cbr2=CBR(base*4,base*2,up=True)
            self.cbr3=CBR(base*2,base,up=True)
            self.c1=L.Convolution2D(base,3,7,1,3,initialW=w)

            self.bn0=L.BatchNormalization(base)

    def __call__(self,x):
        if x.shape[0]==1:
            h=F.relu(self.in0(self.c0(x)))
        else:
            h=F.relu(self.bn0(self.c0(x)))
        h=self.cbr0(h)
        h=self.cbr1(h)
        h=self.res0(h)
        h=self.res1(h)
        h=self.res2(h)
        h=self.res3(h)
        h=self.res4(h)
        h=self.res5(h)
        h=self.cbr2(h)
        h=self.cbr3(h)
        h=self.c1(h)

        return F.tanh(h)

class Discriminator(Chain):
    def __init__(self,base=32):
        w = initializers.Normal(0.02)
        super(Discriminator,self).__init__()
        with self.init_scope():
            self.cbr0=CBR(3,base,down=True,activation=F.leaky_relu)
            self.cbr1=CBR(base,base*2,down=True,activation=F.leaky_relu)
            self.cbr2=CBR(base*2,base*4,down=True,activation=F.leaky_relu)
            self.cbr3=CBR(base*4,base*8,down=True,activation=F.leaky_relu)
            self.cout=L.Convolution2D(base*8,1,3,1,1,initialW=w)

    def __call__(self,x):
        h=self.cbr0(x)
        h=self.cbr1(h)
        h=self.cbr2(h)
        h=self.cbr3(h)
        h=self.cout(h)

        return h

class CBR_encoder(Chain):
    def __init__(self, in_ch, out_ch, ksize):
        w = initializers.HeNormal()
        super(CBR_encoder, self).__init__()
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
            self.cbr0 = CBR_encoder(in_ch, base, ksize=5)
            self.cbr1 = CBR_encoder(base, base, ksize=5)
            self.cbr2 = CBR_encoder(base, base*2, ksize=5)
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

class TaskNet(Chain):
    def __init__(self):
        super(TaskNet, self).__init__()
        with self.init_scope():
            self.encoder = Encoder(in_ch=3)
            self.classifier = Classification()

    def __call__(self, x):
        h = self.encoder(x)
        h = self.classifier(h)

        return h

class Discriminator_feature(Chain):
    def __init__(self, hidden=2048):
        super(Discriminator_feature, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, hidden)
            self.l1 = L.Linear(hidden, hidden)
            self.l2 = L.Linear(hidden, 1)

    def __call__(self, x):
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = self.l2(h)

        return h