import d2lzh as d2l
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')

x = nd.arange(-8.0, 8.0, 1)
x.attach_grad()
with autograd.record():
    y = x.relu()
# xyplot(x, y, 'relu')

y.backward()
# xyplot(x, x.grad, 'grad of relu')

with autograd.record():
    y = x.sigmoid()
# xyplot(x, y, 'sigmoid')

y.backward()
# xyplot(x, x.grad, 'grad of sigmoid')

with autograd.record():
    y = x.tanh()
# xyplot(x, y, 'tanh')

y.backward()
# xyplot(x, x.grad, 'grad of tanh')

"""
    3.9.1 读取数据集
"""
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""
    3.9.2 定义模型参数
"""

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

"""
    3.9.3 定义激活函数
    使用maximum来实现RELU，不是直接调用MXNet的RELU函数
"""
def relu(X):
    return nd.maximum(X, 0)

"""
    3.9.4 定义模型
"""
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2

def net2(X1):
    X1 = X1.reshape((-1, num_hiddens1))
    H = relu(nd.dot(X1, W2) + b2)
    return nd.dot(H, W3) + b3
"""
    3.9.5 定义损失函数
"""

loss = gloss.SoftmaxCrossEntropyLoss()

"""
    3.9.6 训练模型
"""

num_epochs, lr = 10, 0.5
d2l.train_ch3(net2(net), train_iter, test_iter, loss, num_epochs, batch_size, params, lr)