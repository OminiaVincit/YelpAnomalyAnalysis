import cupy
import chainer.functions as F
from chainer import Variable
from  input_gradient_keeper import InputGradientKeeper
import numpy as np

def as_mat(x):
    return x.reshape(x.shape[0], x.size // x.shape[0])

def normalize_axis1(x):
    xp = cupy.get_array_module(*x)
    abs_x = abs(x)
    x = x / (1e-6 + abs_x.max(axis=1,keepdims=True))
    x_norm_2 = x**2
    return x / xp.sqrt(1e-6 + x_norm_2.sum(axis=1,keepdims=True))


def perturbation_with_L2_norm_constraint(x,norm):
    return norm * normalize_axis1(x)

def perturbation_with_max_norm_constraint(x,norm):
    xp = cupy.get_array_module(*x)
    return norm * xp.sign(x)

class AdversarialTrainer(object):

    def __init__(self, model, epsilon=1.0, norm_constraint_type='L2', lamb=1.0):
        self.model = model
        self.epsilon = epsilon
        self.norm_constraint_type = norm_constraint_type
        self.lamb = lamb

    def accuracy(self, x_data, t):
        return self.model.accuracy(x_data, t, train=False)

    # def accuracy_for_adversarial_examples(self,x,t):
    #     xadv,loss = self.get_adversarial_examples(x,t,test=True)
    #     return F.accuracy(self.nn.y_given_x(xadv,test=True),t)

    def cost_fitness(self, x, t, train=True):
        """
        :param x_data: input
        :param t: target
        :return: standard fitness loss ( cross entropy )
        """
        return self.model.loss(x.data, t, train)

    def cost_adversarial_training(self, x_data, t, train=True):
        xadv, ptb, cost_fitness = self.get_adversarial_examples(x_data, t, train=train)
        cost_fitness_adv = self.cost_fitness(xadv, t, train=train)
        return cost_fitness, self.lamb*cost_fitness_adv

    def get_adversarial_examples(self, x_data, t, train=True):
        x = Variable(x_data)
        input_gradient_keeper = InputGradientKeeper()
        x_ = input_gradient_keeper(x)
        cost_fitness = self.cost_fitness(x_, t, train=train)
        cost_fitness.backward()
        gx = input_gradient_keeper.gx
        if (self.norm_constraint_type == 'L2'):
            ptb = perturbation_with_L2_norm_constraint(gx, self.epsilon)
        else:
            ptb = perturbation_with_max_norm_constraint(gx, self.epsilon)
        xadv = x + ptb.reshape(x.data.shape)
        return xadv, ptb, cost_fitness