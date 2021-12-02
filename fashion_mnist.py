from IPython import display

import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time

"""
    3.5.1 获取数据集
"""
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

print(len(mnist_test), len(mnist_train))

feature, label = mnist_train[0]
print(feature.shape, feature.dtype)
print(label, type(label), label.dtype)

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
X, y = mnist_train[0:9]
show_fashion_mnist(X,get_fashion_mnist_labels(y))
display.set_matplotlib_formats('svg')

"""
    3.5.2 读取小批量
    mac 中可以通过使用多进程来加快数据的读取速度
"""

batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0 # 0标识不用额外的进程来加快读取速度
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers = num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers = num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))


