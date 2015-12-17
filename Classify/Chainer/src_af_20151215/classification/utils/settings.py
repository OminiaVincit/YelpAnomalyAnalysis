#!env python
# -*- coding: utf-8 -*-
import os
import glob

class Settings:
    """
    For environment settings
    """

    def __init__(self):
        """
        Init 
        """
        pass

    # Sample control
    LEVELS = [-2, -1, 0, 1, 2, 3]
    TLABELS = [0, 1]

    SET_SAMPLES = dict()
    SET_SAMPLES[0] = [160, 160, 160, 160, 160, 160]
    SET_SAMPLES[1] = [0, 0, 80, 80, 80, 80]

    FEATURES_DIR = r'/home/zoro/work/Dataset/Features'