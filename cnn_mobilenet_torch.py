# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import mnist_me
import numpy as np
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

# train_data = torchvision.datasets.MNIST(
#     root='./mnist',
#     train=True,
#     transform=torchvision.transforms.ToTensor(),# 0-1
#     download=DOWNLOAD_MNIST
# )
train_data = mnist_me.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),# 0-1
    download=False
)
# 训练数据有6000张图片，28*28
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i'%train_data.train_labels[0])
# plt.show()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_data = mnist_me.MNIST(
    root='./mnist/',
    train=False
)

# 为了节约时间, 我们测试时只测试前2000个
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        def conv_bn(inp, oup, stride):# 标准卷积
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride): # mobile 卷积
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.conv1 = nn.Sequential(
            # nn.Conv2d( # 1*28*28
            #     in_channels=1, #高度。深度
            #     out_channels=16, #输出多少高度，也就是卷积核的数目
            #     kernel_size=5,
            #     stride=1,
            #     padding=2# if ==1, padding=(kenerl_size-1)/2=(5-1)/2
            # ),
            # nn.ReLU(),# 16, 28, 28
            conv_dw(1, 16, 1), # mobilenet的深度分离卷积
            nn.MaxPool2d(kernel_size=2)# 16, 14,14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(# 16 14 14
                in_channels=16,  # 高度。深度
                out_channels=32,  # 输出多少高度，也就是卷积核的数目
                kernel_size=5,
                stride=1,
                padding=2  # if ==1, padding=(kenerl_size-1)/2=(5-1)/2
            ),
            nn.ReLU(),# 32 14 14
            nn.MaxPool2d(kernel_size=2),# 32 7 7
        )
        self.out = nn.Linear(32*7*7, 10) # 全链接层 二维数据

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)# (batch, 32, 7, 7)
        # 展平数据 3维变2维
        x = x.view(x.size(0), -1) # 变成（batch，32*7*7）
        output = self.out(x)
        return output


cnn = CNN()
# print(cnn)
optimzer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimzer.zero_grad()
        loss.backward() # 计算梯度
        optimzer.step() # 应用梯度

        #在整个测试集（这里为2000））上计算准确度
        if step%50==0:
            test_out = cnn(test_x)
            pred_y = torch.max(test_out,1)[1].data.squeeze()#矩阵中去掉一层中括号，压榨
            acc = np.true_divide(sum(pred_y==test_y),test_y.size(0))
            print('epoch:', epoch,'| train loss:%.4f'%loss.data[0],
                  '| test acc:', acc
                  )


# 打印10个数据的预测结果
import time
s = time.time()
test_10out = cnn(test_x[:10])
print "predict time:", time.time() - s
pred_y = torch.max(test_10out,1)[1].data.numpy().squeeze()
print(pred_y, 'predict number')
print(test_y[:10].numpy(),'real number')

#训练中，标签并不是one-hot编码，就是普通的数字标签
