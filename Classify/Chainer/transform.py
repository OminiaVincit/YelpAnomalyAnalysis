#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

class Transform(object):
    def __init__(self, **params):
        [setattr(self, key, value) for key, value in params.items()]

    def transform(self, img):
        self._img = img
        if hasattr(self, 'norm'):
            if self.norm:
                if not self._img.dtype == np.float32:
                    self._img = self._img.astype(np.float32)
                # global contrast normalization
                for ch in range(self._img.shape[2]):
                    im = self._img[:, :, ch]
                    im = (im - np.mean(im)) / \
                        (np.std(im) + np.finfo(np.float32).eps)
                    self._img[:, :, ch] = im

        if not self._img.dtype == np.float32:
            self._img = self._img.astype(np.float32)

        return self._img