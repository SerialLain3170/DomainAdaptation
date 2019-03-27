import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda

xp = cuda.cupy
cuda.get_device(0).use()

def rand_projection(embedding_dim, num_samples=128):
    projections = [w / np.sqrt((w**2)).sum() for w in np.random.normal(size=(num_samples, embedding_dim)).astype(np.float32)]
    projections = np.asarray(projections)

    return chainer.as_variable(cuda.to_gpu((projections)))
    

def sliced_wasserstein_distance(x, y, num_projections=128, p=2):
    embedding_dim = x.shape[1]
    projections = rand_projection(embedding_dim, num_projections)
    x_projection = F.matmul(x, projections.transpose(1,0))
    y_projection = F.matmul(y, projections.transpose(1,0))

    x_proj = chainer.as_variable(xp.sort(x_projection.data, axis=1))
    y_proj = chainer.as_variable(xp.sort(y_projection.data, axis=1))

    return F.mean_absolute_error(x_proj, y_proj)