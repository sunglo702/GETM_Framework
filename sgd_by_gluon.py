import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np
import time

def get_data_ch7():
    data = np.genfromtxt('data/airfoil_self_noise.dat', delimiter = '\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, : -1]), nd.array(data[:1500, -1])

features, labels = get_data_ch7()
print(features.shape)


def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels, batch_size=256, num_epochs=10):
    # initialize model
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()

    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)

    # create Trainer toiter model params
    trainer = gluon.Trainer(net.collect_params(), trainer_name, trainer_hyperparams)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size) # make average grad in Trainer instance
            if (batch_i + 1) * batch_size % 100 ==0:
                ls.append(eval_loss())

        # print ans and compose diagram
        print('loss : %f, %f sec per epoch' % (ls[-1], time.time() - start))
        d2l.set_figsize()
        d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
        d2l.plt.xlabel('epoch')
        d2l.plt.ylabel('loss')

train_gluon_ch7('sgd', {'learning_rate': 0.1}, features, labels, 10)