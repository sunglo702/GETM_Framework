import os
import zipfile

from matplotlib import pyplot as plt
from mxnet import autograd, nd
import mxnet as mx
import random
import d2lzh as d2l
from mxnet.gluon import data as gdata
from IPython import display
import sys
import time
import math
import numpy as np

from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i+batch_size, num_examples)])
        yield features.take(j), labels.take(j) # take函数根据索引返回对应元素
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, lables):
    d2l.use_svg_display()
    # 这里的 _ 代表忽略（不使用）的变量
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12,12))
    for f, img, lbl in zip(figs, images, lables):
        f.imshow(img.reshape((28,28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
    '~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root) # 展开用户路径'~'
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]

    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0标识不用额外的进程来加快读取速度
    else:
        num_workers = 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                                  batch_size, shuffle=True,
                                  num_workers = num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                                 batch_size, shuffle=False,
                                 num_workers = num_workers)

    return train_iter, test_iter

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()

            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)

            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n+= y.size

        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition # 这里利用了广播机制

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)

    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def linreg(X, w, b):
    return nd.dot(X, w) + b

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx = None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i+1: i + num_steps + 1]
        yield X, Y

def data_iter_radom(corpus_indices, batch_size, num_steps, ctx = None):
    # 减 1 是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps 的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1 ) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)

def load_data_jay_lyrics():
    with zipfile.ZipFile('./data/data_jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')

    # print(corpus_chars[:40])

    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]

    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    print(vocab_size)

    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    sample = corpus_indices[:20]
    print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    print('indices:', sample)
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1, ), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()

    return ctx

def grad_clipping(params,  theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()

    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]

    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))

    return ''.join([idx_to_char[i] for i in output])


def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta, batch_size,
                          pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_radom
    else:
        data_iter_fn = d2l.data_iter_consecutive

    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter: # 如果使用相邻采样， 在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)

        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter: # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else: # 否则需要使用detach函数从计算图分离状态
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # output 有num_steps 个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)

                # 连结之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)

                # Y的形状是(batch_size, num_steps), 转置后在变成长度为batch * num_steps 的向量， 这样跟输出的行一一对应
                y = Y.T.reshape((-1, ))

                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx) # 裁剪梯度
            d2l.sgd(params, lr, 1) # 因为误差已经取过均值，梯度不用在做平均
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1 ) % pred_period == 0:
            print('epoch %d, preplexity %f, time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time()-start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))


def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps, lr, clipping_theta, batch_size,
                          pred_period, pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_radom
    else:
        data_iter_fn = d2l.data_iter_consecutive

    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter: # 如果使用相邻采样， 在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, ctx)

        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter: # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else: # 否则需要使用detach函数从计算图分离状态
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # output 有num_steps 个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)

                # 连结之后形状为(num_steps * batch_size, vocab_size)
                outputs = nd.concat(*outputs, dim=0)

                # Y的形状是(batch_size, num_steps), 转置后在变成长度为batch * num_steps 的向量， 这样跟输出的行一一对应
                y = Y.T.reshape((-1, ))

                # 使用交叉熵损失计算平均分类误差
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx) # 裁剪梯度
            d2l.sgd(params, lr, 1) # 因为误差已经取过均值，梯度不用在做平均
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1 ) % pred_period == 0:
            print('epoch %d, preplexity %f, time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time()-start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))



def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char, char_to_idx):
    # 使用model的成员函数来初始化隐藏状态
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]

    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1,1))
        (Y, state) = model(X, state) # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))

    return ''.join([idx_to_char[i] for i in output])


def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)

        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1, ))
                l = loss(output, y).mean()

            l.backward()

            # 梯度裁剪
            params = [p.data() for p in model.collect_params().values()]
            d2l.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1) # 因为已经误差取过均值，这里梯度还是不用在做平均

            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(prefix, pred_len, model, vocab_size, ctx, idx_to_char, char_to_idx))


class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)
        # 他的输出形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))

        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

def get_data_ch7():
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter = '\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, : -1]), nd.array(data[:1500, -1])


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
        print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))

    return results


def show_trace_2d(f, results):
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

def train_ch7(trainer_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    # initialize model
    net, loss = d2l.linreg, d2l.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean() # use average loss
            l.backward()
            trainer_fn([w, b], states, hyperparams) # iter model params
            if(batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss()) # record the moment train loss per 100 samples

        # print ouput and draw
        print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
        d2l.set_figsize()
        d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
        d2l.plt.xlabel('epoch')
        d2l.plt.ylabel('loss')

