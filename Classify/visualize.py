# -*- coding:utf-8 -*-

import pylab as Plot
import matplotlib.pyplot as plt
from pca import pca_2d, pca_3d, pca_2d_for_2_components, pca_3d_movie
import numpy as np
import os.path
import time

from utils import make_data, get_class_names_for_class_indices
from utils import get_all_class_names

BASE_DIR = r'../../Dataset/Features'
RESULT_DIR = r'../Results'

class Visualise:
    def __init__(self, site, filename, features, classes_to_visualise=None):
        """
        """
        self.path = RESULT_DIR
        self.filename = filename
        self.site = site
        self.features = features
        self.features_data = None
        self.class_indices = None
        self.classes_to_visualise = classes_to_visualise
        self.legend = []

    def __generate_input_data(self):
        """
        Generate the input data so that it can be visualized.
        """
        if not self.features_data == None:
            return

        if not os.path.isfile(self.filename):
            filename = os.path.join(BASE_DIR, 'yelp_all_features.txt')
        else:
            filename = self.filename
        print filename

        usecols = None
        if self.features == 'text_only':
            flist = range(15)
            flist.append(65)
            usecols = tuple(flist)
        elif self.features == 'topics_only':
            flist = range(15, 66)
            usecols = tuple(flist)

        ds = make_data(filename, usecols)
        self.features_data = ds.features
        self.class_indices = ds.target
        if not self.classes_to_visualise == None:
            self.__filter_input_data(self.classes_to_visualise)

        self.legend = get_class_names_for_class_indices(list(set(sorted(self.class_indices))))

    def __filter_input_data(self, classes_to_visualise):
        """
        Filter the input data corresponding to the classes to visualise.
        @param classes_to_visualise: A list containing names for the classes to visualise.
        """
        class_names = get_all_class_names()
        class_indices_to_visualise = []
        for i in range(len(class_names)):
            if class_names[i] in classes_to_visualise:
                class_indices_to_visualise.append(i)

        if not len(class_indices_to_visualise) == len(classes_to_visualise):
            print 'Not all class names were correct.'
            return

        tmp_output_data = []
        tmp_class_indices = []
        for i in range(len(self.features_data)):
            out = self.features_data[i]
            idx = self.class_indices[i]
            if idx in class_indices_to_visualise:
                tmp_output_data.append(out)
                tmp_class_indices.append(idx)

        self.input_data = np.array(tmp_output_data)
        self.class_indices = np.array(tmp_class_indices)

    def visualise_data_pca_2d(self, number_of_components=9):
        """
        Visualise the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param number_of_components: The number of principal components for the PCA plot.
        """

        if self.features_data == None:
            self.__generate_input_data()
        title = self.site + '_' + self.features
        title += '_PCA_2D_' + str(number_of_components) + '_components'
        title += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        title += str(time.time()).replace('.', '')

        pca_2d(self.features_data, self.class_indices, self.path, 
               title, number_of_components, self.legend)

    def visualise_data_pca_2d_two_components(self, component1, component2):
        """
        Visualise the input data on a 2D PCA plot. Specify two components, which will be
        the plotted.
        @param input_data: False if the output data of the DBN should be plottet. Otherwise True.
        @param component1: Principal component 1.
        @param componen2: Principal component 2.
        """
        if self.features_data == None:
            self.__generate_input_data()
        
        title = self.site + '_' + self.features
        title += '_PCA_2D_PC_' + str(component1) + '_PC_' + str(component2) 
        title += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        title += str(time.time()).replace('.', '')

        pca_2d_for_2_components(self.features_data, component1, component2, 
            self.class_indices, self.path, title, self.legend)

    def visualise_data_pca_3d(self, component1, component2, component3):
        """
        Visualise the input data on a 3D PCA plot for principal components 1 and 2.
        Parameters
        ----------
        """
        if self.features_data == None:
            self.__generate_input_data()
        
        title = self.site + '_' + self.features + '_PCA_3D_PC_'
        title += str(component1) + '_PC_' + str(component2)
        title += '_PC_' + str(component3) 
        title += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        title += str(time.time()).replace('.', '')

        pca_3d(self.features_data, component1, component2, component3, 
               self.class_indices, self.path, title, self.legend)

    def visualise_data_pca_3d_movie(self, component1, component2, component3):
        """
        Visualise the input data on a 3D PCA plot movie for principal components 1 and 2.
        Parameters
        ----------
        """
        if self.features_data == None:
            self.__generate_input_data()
        
        title = self.site + '_' + self.features + '_PCA_3D_Movie_PC_'
        title += str(component1) + '_PC_' + str(component2)
        title += '_PC_' + str(component3) 
        title += '_' + time.strftime('%Y-%m-%d_%H-%M-%S_')
        title += str(time.time()).replace('.', '')

        pca_3d_movie(self.features_data, component1, component2, component3, 
                     self.class_indices, self.path, title, self.legend)

if __name__ == '__main__':
    sitelist = ['yelp', 'tripadvisor', 'movies']
    features_list = ['text_only', 'topics_only', 'text_topics']

    for site in sitelist:
        for ft in features_list:
            filename = os.path.join(BASE_DIR, site + '_all_features.txt')
            vs = Visualise(site, filename, ft)
            vs.visualise_data_pca_2d(number_of_components=4)
            vs.visualise_data_pca_2d(number_of_components=9)
            vs.visualise_data_pca_3d(1,2,3)
            vs.visualise_data_pca_3d(1,3,2)
            vs.visualise_data_pca_3d(2,1,3)
            vs.visualise_data_pca_3d(2,3,1)
            vs.visualise_data_pca_3d(3,1,2)
            vs.visualise_data_pca_3d(3,2,1)


