# 这个是用来研究怎么用CNN整进去的。倒也未必要组装上

import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import gzip
import json
import random
import numpy as np

class critic_CNN(paddle.nn.Layer):

    def __init__(self):
        super(critic_CNN, self).__init__()
        self.creat_CNN()

    def creat_CNN(self,env_parameters=0):

        in_channels = 1
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是1
        self.fc = Linear(in_features=980, out_features=1)       
    
    # 卷积层激活函数使用Relu，全连接层不使用激活函数
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)
        return x
    
    def train_critic_CNN(self,train_loader):
        self.train()
        # 使用SGD优化器，learning_rate设置为0.01
        opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=self.parameters())
        # 训练5轮
        EPOCH_NUM = 10
        # MNIST图像高和宽
        IMG_ROWS, IMG_COLS = 28, 28

        for epoch_id in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                #准备数据
                images, labels = data
                images = self.reshape_images(images)

                images = paddle.to_tensor(images)
                labels = paddle.to_tensor(labels)
                
                #前向计算的过程
                predicts = self(images)
                
                #计算损失，取一个批次样本损失的平均值
                loss = F.square_error_cost(predicts, labels)
                avg_loss = paddle.mean(loss)

                #每训练200批次的数据，打印下当前Loss的情况
                if batch_id % 200 == 0:
                    print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
                
                #后向传播，更新参数的过程
                avg_loss.backward()
                # 最小化loss,更新参数
                opt.step()
                # 清除梯度
                opt.clear_grad()

        #保存模型参数
        paddle.save(self.state_dict(), 'mnist.pdparams')
    

    def reshape_images(self,images,in_channels=1, IMG_ROWS=28, IMG_COLS=28):
        image_2D_list = [] 
        for image_1D in images:
            image_2D = image_1D.reshape(in_channels, IMG_ROWS, IMG_COLS)
            image_2D_list.append(image_2D)
        return image_2D_list


def load_data(mode='train'):
    # datafile = './work/mnist.json.gz'
    # print('loading mnist dataset from {} ......'.format(datafile))
    # # 加载json数据文件
    # data = json.load(gzip.open(datafile))
    # print('mnist dataset load done')
   
    # # 读取到的数据区分训练集，验证集，测试集
    # train_set, val_set, eval_set = data

    # if mode=='train':
    #     # 获得训练数据集
    #     imgs, labels = train_set[0], train_set[1]
    # elif mode=='valid':
    #     # 获得验证数据集
    #     imgs, labels = val_set[0], val_set[1]
    # elif mode=='eval':
    #     # 获得测试数据集
    #     imgs, labels = eval_set[0], eval_set[1]
    # else:
    #     raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    # 自己重新适配一下。
    
    if mode=='valid':
        mode = 'test'
    dataset = paddle.vision.datasets.MNIST(mode=mode)
    imgs, labels = dataset.images, dataset.labels
    print("训练数据集数量: ", len(imgs))
    
    # 校验数据
    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
    
    # 获得数据集长度
    imgs_length = len(imgs)
    
    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(imgs_length))
    # 读入数据时用到的批次大小
    BATCHSIZE = 100
    
    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            # 训练模式下打乱数据
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            # 将数据处理成希望的类型
            img = np.array(imgs[i]).astype('float32')
            label = np.array(labels[i]).astype('float32')
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                # 获得一个batchsize的数据，并返回
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据读取列表
                imgs_list = []
                labels_list = []
    
        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)
    return data_generator

if __name__ == "__main__":
    train_loader = load_data('train')
    # train_loader = load_data('valid')
    shishi = critic_CNN()
    shishi.train_critic_CNN(train_loader)


  