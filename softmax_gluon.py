import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

"""
    3.7.1 读取数据集
"""
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

"""
    3.7.2 定义和初始化模型
    softmax回归的输出层是一个全连接层，因此，添加一个输出个数为10的全连接层。使用均值为0，标准差为0.01的正态分布随机初始化模型的权重参数
"""

net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma = 0.01))

"""
    3.7.3 softmax和交叉熵损失函数
    分开定义softmax和交叉熵损失函数可能会造成数值不稳定，因此Gluon提供了一个包括softmax运算和交叉熵损失计算的函数
"""

loss = gloss.SoftmaxCrossEntropyLoss()

"""
    3.7.4 定义优化算法
    使用学习率为0.1的小批量随机梯度下降作为优化算法
"""
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.2})

"""
    3.7.5 训练模型
"""

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)

