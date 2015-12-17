#!env python
# -*- coding: utf-8 -*-

import samples

class Task(object):
    def show(self):
        print '--------------------'
        print 'Task               : {}'.format(self.__class__.__name__)
        print '  Result  directory: {}'.format(self.result_dir)
        print '  Dataset directory: {}'.format(self.dataset_dir)
        print '  Default model    : {}'.format(self.default_model_desc)
        print '  Settings         : {}'.format(self.settings)
        print '--------------------'


class RVClassificationTopicsTaks(Task):
    def __init__(self):
        self.sample_creator_func = samples.RVTopicsSampleCreator
        self.dataset_dir         = r'/home/zoro/work/Dataset/Features'
        self.result_dir          = r'/home/zoro/work/classify_result'
        self.default_model_desc  = 'models.rv_classification_models@NetModel_BN'
        self.target_path_names   = ['RVClassification']
        self.is_regression       = False
        self.settings            = {
                'weight_decay': 0.001
                }

class RVCheckTopicsTaks(Task):
    def __init__(self):
        self.sample_creator_func = samples.RVForCheckSampleCreator
        self.dataset_dir         = r'/home/zoro/work/Dataset/Features_bak_20151214'
        self.result_dir          = r'/home/zoro/work/classify_result'
        self.default_model_desc  = 'models.rv_classification_models@NetModel_BN'
        self.target_path_names   = ['RVClassification']
        self.is_regression       = False
        self.settings            = {
                'weight_decay': 0.001
                }


# module variable. canonical name -> settings
def get_task_settings(task_name):
    task_setting_class = {
        'rv_topics': RVClassificationTopicsTaks,
        'rv_check': RVCheckTopicsTaks,
        }
    return task_setting_class[task_name]()

