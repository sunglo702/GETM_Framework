import d2lzh as d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

"""
    6.4.1 one_hot 向量
"""
print(nd.one_hot(nd.array([0, 2]), vocab_size))
def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
print(len(inputs), inputs[0].shape)

"""
    6.4.2 初始化模型参数
"""

num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)

    # 附上梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()

    return params

"""
    6.4.3 定义模型
"""
# 初始化隐藏状态，返回一个形状为（批量大小，隐藏单元个数）的值为0的NDArray组成的元组。
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )

def rnn(inputs, state, params):
    # inputs 和 outputs 都是 num_steps 个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        H = nd.relu(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)

    return outputs, (H,)

state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X.as_in_context(ctx), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)

"""
    6.4.4 定义预测函数
    基于前缀prefix来预测接下来的num_charts个字符
"""
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

print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))

"""
    6.4.5 梯度裁剪
    为了避免梯度爆炸，采用梯度裁剪，个人理解，等价于正则化
"""

def grad_clipping(params,  theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()

    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


## 这里有一个困惑度的概念来评价语言模型的好坏, perplexity

"""
    6.4.7 定义模型训练函数
"""

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


"""
    6.4.8 训练模型并创作歌词
"""

num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

# 以下是通过随机采样模型训练并创作歌词
# print(train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char,
#                       char_to_idx, True, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
#                       prefixes))

# 以下是通过相邻采样模型训练并创作歌词
print(train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len,
                      prefixes))

