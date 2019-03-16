import chainer
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers, datasets
from sklearn.manifold import TSNE
import numpy as np
import argparse
import cv2 as cv
from model import Encoder
import pylab

xp = cuda.cupy
cuda.get_device(0).use()

def prepare_dataset(index):
    img = cv.resize(test_mnist[index][0].transpose(1,2,0), (32, 32))
    img = img.transpose(2,0,1).astype(np.float32)

    return img

def tnse_visualize(model, model_type):
    feature_vector = None

    for b in range(0, Ntest, batchsize):
        test_batch = []
        for index in range(batchsize):
            img = prepare_dataset(b + index)
            test_batch.append(img)
        test_batch = chainer.as_variable(xp.array(test_batch).astype(xp.float32))

        with chainer.using_config('train', False):
            y = model(test_batch)
        y.unchain_backward()
        y = y.data.get()
        
        if feature_vector is None:
            feature_vector = y
        else:
            feature_vector = np.concatenate([feature_vector, y], axis=0)

    embed = tsne.fit_transform(feature_vector)
    pylab.plot(embed)
    pylab.axis('off')
    pylab.savefig('{}_embed.png'.format(model_type))

parser = argparse.ArgumentParser(description='Visualization')
parser.add_argument("--b", default=250, type=int, help="batch size")
args = parser.parse_args()

batchsize = args.b

source_encoder = Encoder(in_ch=3)
source_encoder.to_gpu()
serializers.load_npz('./encoder.model', source_encoder)

target_encoder = Encoder(in_ch=3)
target_encoder.to_gpu()
serializers.load_npz('./target_encoder.model', target_encoder)

tsne = TSNE(n_components=2)

_, test_mnist = datasets.get_mnist(ndim=3, rgb_format=True)
Ntest = 10000

print('starting TSNE...')
tnse_visualize(source_encoder, model_type='source_only')
tnse_visualize(target_encoder, model_type='adda')