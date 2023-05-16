# -*- coding: utf-8 -*-
"""
Compare strategies using random questions.
"""
import pickle
import os
import matplotlib.pyplot as plt 
import numpy as np
import tikzplotlib
import seaborn

conf_type = 'strong'
nb_questions = 15
nb_parameters = 4
path_data = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
path_results = 'results/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
    
if __name__ == '__main__':
    
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    
    try:
        with open(path_data + 'dataset.pk', 'rb') as f:
            dataset = pickle.load(f)
    except IOError:
        dataset = {}
    try:
        with open(path_data + 'possibilist.pk', 'rb') as f:
            data_possibilist = pickle.load(f)
    except IOError:
        data_possibilist = {}
    try:
        with open(path_data + 'l_out_of_n.pk', 'rb') as f:
            data_l_out_of_n = pickle.load(f)
    except IOError:
        data_l_out_of_n = {}
        
    rational_all = dataset['rational']
    nb_errors_real = nb_questions - np.count_nonzero(rational_all, axis = 1)
    nb_errors_detected = np.asarray(data_l_out_of_n['nb_errors_detected'])
    nb_errors_detected_group = np.zeros((np.max(nb_errors_real)+1, np.max(nb_errors_real)+1)).astype(int)
    for i in range(0, len(nb_errors_detected)):
        nb_errors_detected_group[nb_errors_detected[i], nb_errors_real[i]] += 1
    difference = nb_errors_real - nb_errors_detected
    difference_mean = np.mean(difference)
        
    fig, ax = plt.subplots()
    labels =  np.where(nb_errors_detected_group[:,:] > 0, nb_errors_detected_group[:,:] , '')
    labels = labels.astype(str)
    ax = seaborn.heatmap(nb_errors_detected_group[:,:], annot=labels, fmt = '',
                         cbar=False, cmap = "Blues")
    ax.invert_yaxis()
    ax.set_xlabel('Real number of errors', fontsize = 9)
    ax.set_ylabel('Number of errors detected', fontsize = 9)
    ax.set_xlim(0,np.max(nb_errors_real)+1)
    ax.set_ylim(0,np.max(nb_errors_detected)+1)
    fig.set_dpi(300.0)
    tikzplotlib.save(path_results + 'nb_wrong_answers_detected.tex')
    plt.savefig(path_results + 'nb_wrong_answers_detected.png', dpi=300)
