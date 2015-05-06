#!env python
# -*- coding:utf-8 -*-
u'''Optimization'''

import numpy as np

class RegularOptimizer():
    u'''Regular optimizer'''

    def __init__(self, data, target, alpha=0, max_iters = 10000):
        u'''Initialize'''
        X = data
        
        # Add colum one to data
        n_c = X.shape[1]
        n_r = X.shape[0]
        X_it = np.ones(shape=(n_r, n_c + 1))
        X_it[:, 1:(n_c+1)] = X

        self.data = X_it
        self.target = target
        features_size = data.shape[1]
        self.weights = np.zeros(shape=(features_size + 1, 1))
        self.base_lr = 1.0
        self.min_lr = 1e-8
        self.min_mag = 1e-10
        self.gamma = 0.8
        self.step = 1000
        self.iter = 0
        self.alpha = alpha
        self.max_iters = max_iters

        self.cost = None
        self.gradient = np.zeros(shape=(features_size + 1, 1))
        self.prev_gradient = np.zeros(shape=(features_size + 1, 1))

    def compute_cost(self):
        u'''Calculate cost function'''
        X = self.data
        Q = self.target
        m = X.shape[0]
        w = self.weights
        Diff = X.dot(w) - Q
        self.cost = (Diff.T.dot(Diff)) / float(2*m) + self.alpha * w.T.dot(w) / 2.0

    def compute_gradient(self):
        u'''Compute gradient of cost function'''
        X = self.data
        Q = self.target
        m = X.shape[0]
        w = self.weights
        Diff = X.dot(w) - Q
        self.gradient = (X.T.dot(Diff)) / float(m) + self.alpha * w

    def compute_gradient_cost(self):
        u'''Compute gradient and cost function'''
        X = self.data
        Q = self.target
        m = X.shape[0]
        w = self.weights
        Diff = X.dot(w) - Q
        self.gradient = X.T.dot(Diff) / float(m) + self.alpha * w
        self.cost = Diff.T.dot(Diff) / float(2*m) + self.alpha * w.T.dot(w) / 2.0

    def update(self):
        u'''Update paramters by gradient descent'''
        gradient = self.gradient
        magnitude = gradient.T.dot(gradient)
        magnitude = np.sqrt(magnitude)
        if magnitude < self.min_mag:
            return False
        direction = gradient.T.dot(self.prev_gradient)
        #if self.iter % self.step == 0:
        #    self.base_lr *= self.gamma
        if direction < 0:
            self.base_lr *= self.gamma

        self.weights -= self.gradient * self.base_lr
        self.iter += 1
        return True

    def forward(self):
        if self.iter >= self.max_iters or self.base_lr < self.min_lr:
            return False
        self.prev_gradient = self.gradient
        self.compute_gradient_cost()
        return self.update()
        
    def run(self):
        u'''Run optimizer'''
        self.compute_gradient_cost()
        while self.forward():
            print self.iter-1, self.cost[0][0], self.base_lr

    def solve(self):
        u'''Closed form solution'''
        X = self.data
        Q = self.target
        [m, f] = X.shape
        mtr1 = X.T.dot(X) + self.alpha * m * np.identity(f)
        try:
            inverse = np.linalg.inv(mtr1)
        except np.linalg.LinAlgError:
            # Not invertible. Skip this one.
            print 'Inverse error'
            pass
        else:
            # continue with what you were doing
            mtr2 = X.T.dot(Q)
            self.weights = inverse.dot(mtr2)
            self.compute_cost()


        



