# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:43:55 2023

@author: adamloic
"""

import numpy as np
from scipy.stats import wilcoxon

conf_type = 'strong'
nb_questions = 15
nb_parameters = 4
path_data = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
path_results = 'results/comparison_uncertainty_strategy/'

if __name__ == '__main__':

    data_minimax_regret = np.genfromtxt(path_data + 'dataset_minimax_regret.csv', delimiter=",").T
    data_maximax = np.genfromtxt(path_data + 'dataset_maximax.csv', delimiter=",").T
    data_maximin = np.genfromtxt(path_data + 'dataset_maximin.csv', delimiter=",").T
    
    for i in range(0, data_minimax_regret.shape[1]):
        
        mean_diff = np.mean(data_minimax_regret[:,i]) - np.mean(data_maximax[:,i])
        if mean_diff == 0:
            print('Nope égal')
            continue
        direction = 'greater' if mean_diff > 0 else 'less'
        _, p = wilcoxon(data_minimax_regret[:,i], data_maximax[:,i], alternative = direction)
        print(mean_diff, p, p < 0.05)
        
    print('')
    
    for i in range(0, data_minimax_regret.shape[1]):
        
        mean_diff = np.mean(data_minimax_regret[:,i]) - np.mean(data_maximin[:,i])
        direction = 'greater' if mean_diff > 0 else 'less'
        _, p = wilcoxon(data_minimax_regret[:,i], data_maximin[:,i], alternative = direction)
        print(mean_diff, p, p < 0.05)
    
    print('')
    print("---")
    print('')
    
    data = [data_minimax_regret, data_maximax, data_maximin]
    for d in data:
        for i in range(1, d.shape[1]):
            
            mean_diff = np.mean(d[:,0]) - np.mean(d[:,i])
            if mean_diff == 0:
                print('Nope égal')
                continue
            direction = 'greater' if mean_diff > 0 else 'less'
            _, p = wilcoxon(d[:,0], d[:,i], alternative = direction)
            print(mean_diff, p, p < 0.05)

        print("")
        