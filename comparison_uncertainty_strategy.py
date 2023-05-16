# -*- coding: utf-8 -*-
"""
Statistical tests.
"""

from scipy import stats

import pickle
import os
import matplotlib.pyplot as plt 
import numpy as np
import tikzplotlib
import matplotlib.lines as mlines

conf_type = 'strong'
nb_questions = 15
nb_parameters = 4
path_data = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
path_results = 'results/comparison_uncertainty_strategy/'

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
    try:
        with open(path_data + 'mcs.pk', 'rb') as f:
            data_mcs = pickle.load(f)
    except IOError:
        data_mcs = {}
    try:
        with open(path_data + 'epsilon.pk', 'rb') as f:
            data_epsilon = pickle.load(f)
    except IOError:
        data_epsilon = {}

    nb_repetitions = dataset['alternatives'].shape[0]
    mcs_confidence_real_regret = np.zeros((nb_repetitions,3))
    mcs_size_real_regret = np.zeros((nb_repetitions,3))
    mcs_size_confidence_real_regret = np.zeros((nb_repetitions,3))
    mcs_best_real_regret = np.zeros((nb_repetitions,3))

    for i in range(0,300):
        
        mcs_size = data_mcs['size'][i]
        mcs_confidence_mean = data_mcs['confidence_mean'][i]
        mcs_res_minimax_regret = data_mcs['real_regret_minimax_regret_mcs'][i]
        mcs_res_maximax = data_mcs['real_regret_maximax_mcs'][i]
        mcs_res_maximin = data_mcs['real_regret_maximin_mcs'][i]

        max_confidence_mcs_idx = np.random.choice(np.where(mcs_confidence_mean == np.max(mcs_confidence_mean))[0])
        mcs_size_real_regret[i,:] = [mcs_res_minimax_regret[max_confidence_mcs_idx],
                                     mcs_res_maximax[max_confidence_mcs_idx],
                                     mcs_res_maximin[max_confidence_mcs_idx]]

        max_size_mcs_idx_list = np.where(mcs_size == np.max(mcs_size))[0]
        max_size_mcs_idx = np.random.choice(max_size_mcs_idx_list)
        mcs_confidence_real_regret[i,:] = [mcs_res_minimax_regret[max_size_mcs_idx],
                                           mcs_res_maximax[max_size_mcs_idx],
                                           mcs_res_maximin[max_size_mcs_idx]]

        max_size_confidence_mean = mcs_confidence_mean[max_size_mcs_idx_list]
        max_size_max_confidence_mcs_idx = np.random.choice(np.where(max_size_confidence_mean == np.max(max_size_confidence_mean))[0])
        mcs_size_confidence_real_regret[i,:] = [mcs_res_minimax_regret[max_size_max_confidence_mcs_idx],
                                                mcs_res_maximax[max_size_max_confidence_mcs_idx],
                                                mcs_res_maximin[max_size_max_confidence_mcs_idx]]

        mcs_best_real_regret[i,:] = [np.min(mcs_res_minimax_regret),
                                     np.min(mcs_res_maximax),
                                     np.min(mcs_res_maximin)]
        
    data_minimax_regret_non_zero = [data_possibilist['real_regret_minimax_regret_zero'][data_possibilist['inconsistency'] != 0],
                                    data_possibilist['real_regret_minimax_regret_ignorance'][data_possibilist['inconsistency'] != 0],
                                    data_l_out_of_n['real_regret_minimax_regret_l_out_of_n'][data_possibilist['inconsistency'] != 0],
                                    mcs_best_real_regret[:,0][data_possibilist['inconsistency'] != 0],
                                    mcs_confidence_real_regret[:,0][data_possibilist['inconsistency'] != 0],
                                    mcs_size_confidence_real_regret[:,0][data_possibilist['inconsistency'] != 0],
                                    mcs_size_real_regret[:,0][data_possibilist['inconsistency'] != 0],
                                    data_epsilon['real_regret_minimax_regret_epsilon'][data_possibilist['inconsistency'] != 0]]
    
    data_maximax_non_zero = [data_possibilist['real_regret_maximax_zero'][data_possibilist['inconsistency'] != 0],
                             data_possibilist['real_regret_maximax_ignorance'][data_possibilist['inconsistency'] != 0],
                             data_l_out_of_n['real_regret_maximax_l_out_of_n'][data_possibilist['inconsistency'] != 0],
                             mcs_best_real_regret[:,1][data_possibilist['inconsistency'] != 0],
                             mcs_confidence_real_regret[:,1][data_possibilist['inconsistency'] != 0],
                             mcs_size_confidence_real_regret[:,1][data_possibilist['inconsistency'] != 0],
                             mcs_size_real_regret[:,1][data_possibilist['inconsistency'] != 0],
                             data_epsilon['real_regret_maximax_epsilon'][data_possibilist['inconsistency'] != 0]]
    
    data_maximin_non_zero = [data_possibilist['real_regret_maximin_zero'][data_possibilist['inconsistency'] != 0],
                             data_possibilist['real_regret_maximin_ignorance'][data_possibilist['inconsistency'] != 0],
                             data_l_out_of_n['real_regret_maximin_l_out_of_n'][data_possibilist['inconsistency'] != 0],
                             mcs_best_real_regret[:,2][data_possibilist['inconsistency'] != 0],
                             mcs_confidence_real_regret[:,2][data_possibilist['inconsistency'] != 0],
                             mcs_size_confidence_real_regret[:,2][data_possibilist['inconsistency'] != 0],
                             mcs_size_real_regret[:,2][data_possibilist['inconsistency'] != 0],
                             data_epsilon['real_regret_maximin_epsilon'][data_possibilist['inconsistency'] != 0]]
    
    data_all = data_minimax_regret_non_zero + data_maximax_non_zero + data_maximin_non_zero
    
    for d in data_all:
        print(stats.shapiro(d)[1] < 0.05)
    
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    j = 0
    for data_criterion in [data_minimax_regret_non_zero, data_maximax_non_zero, data_maximin_non_zero]:
        y = np.asarray([0,5,10,15,20,25,30,35]) + np.ones(8) * j
        for i in range(0, len(data_criterion)):
            
            data = data_criterion[i]
            data_mean = np.mean(data)
            data_IC = stats.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data),
                              scale=stats.sem(data))
            ax.plot((data_IC[0],data_IC[1]),(y[i],y[i]), '-', color = cmap((2-j)/2))
            ax.plot((data_mean,data_mean),(y[i],y[i]), '|', color = cmap((2-j)/2))
        j+=1
    ax.set_xlim(-0.005,0.3)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1:7] = ['zero', 'ignorence', 'l out of n', 'MCS best', 'MCS conf',
                   'MCS size + conf', 'MCS size', 'epsilon']
    ax.set_yticklabels(labels)
    line_mmr = mlines.Line2D([], [], color=cmap(2/2), marker='|', linestyle='-',
                             label='regret')
    line_max = mlines.Line2D([], [], color=cmap(1/2), marker='|', linestyle='-',
                             label='max')
    line_min = mlines.Line2D([], [], color=cmap(0/2), marker='|', linestyle='-',
                             label='min')

    plt.legend(loc="upper left", handles=[line_mmr, line_max, line_min])
    ax.set_xlabel('Real regret', fontsize = 9)
    fig.set_dpi(300.0)
    tikzplotlib.save(path_results + conf_type + '_' + str(nb_parameters) + '_' + str(nb_questions) + '.tex')
    plt.savefig(path_results + conf_type + '_' + str(nb_parameters) + '_' + str(nb_questions) + '.png', dpi=300, bbox_inches = 'tight')
    
    np.savetxt(path_data + 'dataset_minimax_regret.csv', data_minimax_regret_non_zero, delimiter=",")
    np.savetxt(path_data + 'dataset_maximax.csv', data_maximax_non_zero, delimiter=",")
    np.savetxt(path_data + 'dataset_maximin.csv', data_maximin_non_zero, delimiter=",")