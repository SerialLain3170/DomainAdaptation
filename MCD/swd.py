import numpy as np
import chainer
import chainer.functions as F


def rand_projection(embedding_dim, num_samples=50):
    projections = [w / np.sqrt((w**2)).sum() for w in np.random.normal(size=num_samples, embedding_dim)]
    projections = np.asarray(projections)

    return chainer.as_variable(cuda.to_gpu((projections)))
    

def sliced_wasserstein_distance(x, y, num_projections=50, p=2):
    embedding_dim = x.shape[1]
    projections = rand_projection(embedding_dim, num_projections)
    x_projection = F.matmul(x, projections)
    y_projection = F.matmul(y, projections)

    x_proj = chainer.as_variable(xp.sort(x_projection.data, axis=1))
    y_proj = chainer.as_variable(xp.sort(y_projection.data, axis=1))

    return F.mean_absolute_error(x_proj[0], y_proj[0])