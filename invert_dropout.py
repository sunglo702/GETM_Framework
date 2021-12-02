import d2lzh as d2l
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import loss as gloss, nn, data as gdata

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob

    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    return mask * X / keep_prob

X = nd.arange(16).reshape((2, 8))
print(dropout(X, 0))
X = nd.arange(16).reshape((2, 8))
print(dropout(X, 0.25))
X = nd.arange(16).reshape((2, 8))
print(dropout(X, 0.5))
X = nd.arange(16).reshape((2, 8))
print(dropout(X, 0.75))
X = nd.arange(16).reshape((2, 8))
print(dropout(X, 1))


"""
    3.13.1 定义模型参数
    这里两个隐藏层的多层感知器
"""
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

"""
    3.13.2 定义模型
    通过定义的模型将全连接层和激活函数RELU串起来
"""

drop_prob1, drop_prob2 = 0.2, 0.5
def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training(): # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1) # 在第一层全连接后添加丢弃层
    H2= (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():  # 只在训练模型时使用丢弃法
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层

    return nd.dot(H2, W3) + b3

num_epochs, lr, batch_size = 10, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

"""
    3.13.3 Gluon实现
"""
"""
drop_prob2 first drop_prob1 second 
epoch 1, loss 1.1129, train acc 0.568, test acc 0.772
epoch 2, loss 0.5824, train acc 0.783, test acc 0.817
epoch 3, loss 0.5064, train acc 0.814, test acc 0.846
epoch 4, loss 0.4568, train acc 0.832, test acc 0.855
epoch 5, loss 0.4360, train acc 0.839, test acc 0.854

drop_prob1 first drop_prob2 second 
epoch 1, loss 1.1504, train acc 0.551, test acc 0.767
epoch 2, loss 0.5842, train acc 0.781, test acc 0.838
epoch 3, loss 0.4938, train acc 0.817, test acc 0.838
epoch 4, loss 0.4506, train acc 0.834, test acc 0.857
epoch 5, loss 0.4179, train acc 0.849, test acc 0.864
"""

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dropout(drop_prob1), # 在第一层全连接后添加丢弃层
        nn.Dense(256, activation='relu'), nn.Dropout(drop_prob2), # 在第二层全连接后添加丢弃层
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)

