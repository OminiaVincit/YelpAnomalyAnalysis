from chainer import Variable, FunctionSet
import chainer.functions as F

class NetModelFC(FunctionSet):

    def __init__(self):
        super(NetModelFC, self).__init__(
            conv1=F.Convolution2D(1, 32, 2, stride=1, pad=1),
            bn1=F.BatchNormalization(32),
            conv2=F.Convolution2D(32, 32, 2, stride=1, pad=1),
            bn2=F.BatchNormalization(32),
            conv3=F.Convolution2D(32, 64, 2, stride=1, pad=1),
            fc4 = F.Linear(256, 256),
            fc5 = F.Linear(256, 256),
            fc6 = F.Linear(256, 5)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)
        h = F.relu(self.bn1(self.conv1(x), test = not True))
        h = F.max_pooling_2d(h, 2, stride=2)
        
        h = F.relu(self.bn2(self.conv2(h), test = not True))
        h = F.max_pooling_2d(h, 2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(F.relu(self.fc4(h)), ratio=.5, train = train)
        h = F.dropout(F.relu(self.fc5(h)), ratio=.5, train = train)
        h = self.fc6(h)
    
        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h
