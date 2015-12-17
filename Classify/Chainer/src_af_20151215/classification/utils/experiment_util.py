#!env python
import os
import shutil
import glob
import imp
import datetime

import nn_tools

_last_experiment_file = 'lastexp~'
def set_last_experiment(path):
    with open(_last_experiment_file, 'wb') as fo:
        fo.write(path)

def get_last_experiment():
    if os.path.isfile(_last_experiment_file):
        with open(_last_experiment_file, 'rb') as fo:
            path = fo.read()
    if os.path.isdir(path):
        return path
    return None

_experiment_now = datetime.datetime.now()
def now(): return _experiment_now

class ExperimentResult(object):
    u'''information associated to an experiment (misc config + network + parameters + training progress)'''
    def __init__(self, path):
        self.path = path
        self.model_backup_dir = os.path.join(self.path, 'backup_parameters')
        self.model_structure_path = os.path.join(self.path, 'model.py')
        self.model_parameters_path = os.path.join(self.path, 'model.parameters.pickle')
        self.preprocess_path = os.path.join(self.path, 'preprocess.pickle')
        self.training_log_path = os.path.join(self.path, 'training_log.db')
        self.split_dataset_path = os.path.join(self.path, 'dataset.pickle')
        self.dataset_config_path = os.path.join(self.path, 'dataset_config.pickle')
        self.model = None
        self.log = None
        if not os.path.isdir(self.path):
            os.makedirs(self.path)
        if not os.path.isdir(self.model_backup_dir):
            os.makedirs(self.model_backup_dir)

    @classmethod
    def load_dir(self, path):
        if not os.path.isdir(path):
            raise ValueError('no experiment found: %s' % path)
        result = ExperimentResult(path)
        return result

    def touch_epoch_file(self):
        u'''create or rename epoch file'''
        if self.model is not None:
            fn = os.path.join(self.path, 'epoch_%04d.deleteme' % self.model.total_epoch)
            candidates = list(glob.glob(os.path.join(self.path, 'epoch_*.deleteme')))
            if len(candidates) == 0:
                with open(fn, 'wb') as fo:
                    fo.write('DUMMY')
            else:
                # delete leaving one
                for f in candidates[:-1]:
                    os.unlink(f)
                shutil.move(candidates[-1], fn)

    def finalize(self):
        print 'finalizing ExperimentResult object: {}'.format(self.path)
        self.touch_epoch_file()
        # close log
        if self.log is not None:
            self.log.close()
            self.log = None
        # if no training is carried out, remove the meaningless result directory.
        if self.model is not None and self.model.total_epoch == 0:
            self.delete_directory()
        else:
            # otherwise, set path to '--continue'
            set_last_experiment(self.path)

    def delete_directory(self):
        u"""do not delete but rename to 'gomi'"""
        print 'Deleting result directory..'
        shutil.move(self.path, os.path.join(os.path.dirname(self.path), 'gomi'))

    def get_model_backup_path(self, model):
        return os.path.join(self.model_backup_dir, 'model.parameters.pickle.epoch_%04d' % model.total_epoch)

    def get_model_desc(self):
        with open(os.path.join(self.path, 'model_class.txt'), 'rb') as fi:
            model_module_name, model_class_name = fi.read().split('@')
            return ModelDescription(self.model_structure_path, model_module_name, model_class_name)

    def set_model_desc(self, model_desc):
        with open(os.path.join(self.path, 'model_class.txt'), 'wb') as fo:
            fo.write('%s@%s' % (model_desc.module_name, model_desc.class_name))

    def set_split_dataset(self, split_dataset):
        nn_tools.retrying_pickle_dump(split_dataset, self.split_dataset_path)

    def get_split_dataset(self):
        if not os.path.isfile(self.split_dataset_path):
            return None
        return nn_tools.retrying_pickle_load(self.split_dataset_path)

    def set_dataset_config(self, dataset_config):
        nn_tools.retrying_pickle_dump(dataset_config, self.dataset_config_path)

    def get_dataset_config(self):
        if not os.path.isfile(self.dataset_config_path):
            return None
        return nn_tools.retrying_pickle_load(self.dataset_config_path)

class ModelDescription(object):
    u'''identify a python module defining the model'''
    def __init__(self, file_path, module_name, class_name):
        self.file_path = file_path
        self.module_name = module_name
        self.class_name = class_name
        self.module = None

    @classmethod
    def from_desc_str(self, desc_str):
        '''package.module@class -> package/module.py class'''
        path, class_name = desc_str.split('@')
        parts = path.split('.')
        package = os.path.join(*parts[:-1])
        module =  parts[-1]
        return ModelDescription(os.path.join(package, module + '.py'), module, class_name)

    def _import_module(self):
        u'''import the module described by this object'''
        print 'Importing module from file:', self.file_path
        print 'Module name:', self.module_name
        with open(self.file_path, 'rb') as fi:
            module = imp.load_module(self.module_name, fi, self.file_path, ('.py', 'r', imp.PY_SOURCE))
        #module = imp.load_source(self.module_name, self.file_path)
        self.module = module
        return self.module_name, module

    def import_module_global(self):
        u'''import the module described by this object to global scope'''
        module_name, module = self._import_module()
        globals()[module_name] = module

    def get_model_class(self):
        if not self.module:
            self.import_module_global()
        return getattr(self.module, self.class_name)