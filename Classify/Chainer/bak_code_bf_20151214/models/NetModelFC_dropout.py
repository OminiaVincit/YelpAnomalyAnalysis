from chainer import Variable, FunctionSet
import chainer.functions as F

class NetModelFC_dropout(FunctionSet):

    def __init__(self):
        super(NetModelFC_dropout, self).__init__(
            fc1 = F.Linear(64, 128),
            fc2 = F.Linear(128, 256),
            fc3 = F.Linear(256, 5)
        )

    def forward(self, x_data, y_data, train=True):
        x, t = Variable(x_data), Variable(y_data)
        h = F.dropout(F.relu(self.fc1(x)), train=train, ratio=0.25)
        h = F.dropout(F.relu(self.fc2(h)), train=train, ratio=0.25)
        h = self.fc3(h)
    
        if train:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t)
        else:
            return F.softmax_cross_entropy(h, t), F.accuracy(h, t), h
