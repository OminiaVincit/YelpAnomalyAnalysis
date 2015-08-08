from chainer import Variable, FunctionSet
import chainer.functions as F

class NetModelMini(FunctionSet):

    def __init__(self):
        super(NetModelMini, self).__init__(
            conv1=F.Convolution2D(1, 32, 3, stride=1, pad=1),
            bn1=F.BatchNormalization(32),
            conv2=F.Convolution2D(32, 32, 3, stride=1, pad=1),
            fc3=F.Linear(32, 5)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)
        h = F.max_pooling_2d(F.relu(self.bn1(self.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2, stride=2)
        h = self.fc3(h)
    
        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h