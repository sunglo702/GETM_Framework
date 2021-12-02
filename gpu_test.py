import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn


# !nvidia-smi
print(mx.cpu(), mx.gpu(), mx.gpu(1))

x = nd.array([1, 2, 3])
print(x)

print(x.context)
x = nd.array([1, 2, 3], ctx=mx.gpu())