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
criterion = 'minimax_regret'
nb_questions = 15
path_data_4 = 'data/criteria_' + str(4) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
path_data_5 = 'data/criteria_' + str(5) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'
path_results = 'results/comparison_nb_criteria/'

if __name__ == '__main__':

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    try:
        with open(path_data_4 + 'dataset.pk', 'rb') as f:
            dataset_4 = pickle.load(f)
    except IOError:
        dataset_4 = {}
    try:
        with open(path_data_5 + 'dataset.pk', 'rb') as f:
            dataset_5 = pickle.load(f)
    except IOError:
        dataset_5 = {}

    try:
        with open(path_data_4 + 'possibilist.pk', 'rb') as f:
            data_possibilist_4 = pickle.load(f)
    except IOError:
        data_possibilist_4 = {}
    try:
        with open(path_data_5 + 'possibilist.pk', 'rb') as f:
            data_possibilist_5 = pickle.load(f)
    except IOError:
        data_possibilist_5 = {}

    try:
        with open(path_data_4 + 'l_out_of_n.pk', 'rb') as f:
            data_l_out_of_n_4 = pickle.load(f)
    except IOError:
        data_l_out_of_n_4 = {}
    try:
        with open(path_data_5 + 'l_out_of_n.pk', 'rb') as f:
            data_l_out_of_n_5 = pickle.load(f)
    except IOError:
        data_l_out_of_n_5 = {}
                
    try:
        with open(path_data_4 + 'mcs.pk', 'rb') as f:
            data_mcs_4 = pickle.load(f)
    except IOError:
        data_mcs_4 = {}
    try:
        with open(path_data_5 + 'mcs.pk', 'rb') as f:
            data_mcs_5 = pickle.load(f)
    except IOError:
        data_mcs_5 = {}

    try:
        with open(path_data_4 + 'epsilon.pk', 'rb') as f:
            data_epsilon_4 = pickle.load(f)
    except IOError:
        data_epsilon_4 = {}
    try:
        with open(path_data_5 + 'epsilon.pk', 'rb') as f:
            data_epsilon_5 = pickle.load(f)
    except IOError:
        data_epsilon_5 = {}
        
    nb_repetitions = dataset_5['alternatives'].shape[0]
    mcs_confidence_real_regret = np.zeros((nb_repetitions,2))
    mcs_size_real_regret = np.zeros((nb_repetitions,2))
    mcs_size_confidence_real_regret = np.zeros((nb_repetitions,2))
    mcs_best_real_regret = np.zeros((nb_repetitions,2))
    
    data_mcs = [data_mcs_4, data_mcs_5]
    for i in range(0,300):
        for k in range(0, 2):
            
            mcs_size = data_mcs[k]['size'][i]
            mcs_confidence_mean = data_mcs[k]['confidence_mean'][i]
            mcs_res_criterion = data_mcs[k]['real_regret_' + criterion + '_mcs'][i]
    
            max_confidence_mcs_idx = np.random.choice(np.where(mcs_confidence_mean == np.max(mcs_confidence_mean))[0])
            mcs_size_real_regret[i,k] = mcs_res_criterion[max_confidence_mcs_idx]
    
            max_size_mcs_idx_list = np.where(mcs_size == np.max(mcs_size))[0]
            max_size_mcs_idx = np.random.choice(max_size_mcs_idx_list)
            mcs_confidence_real_regret[i,k] = mcs_res_criterion[max_size_mcs_idx]
    
            max_size_confidence_mean = mcs_confidence_mean[max_size_mcs_idx_list]
            max_size_max_confidence_mcs_idx = np.random.choice(np.where(max_size_confidence_mean == np.max(max_size_confidence_mean))[0])
            mcs_size_confidence_real_regret[i,k] = mcs_res_criterion[max_size_max_confidence_mcs_idx]
            
            mcs_best_real_regret[i,k] = np.min(mcs_res_criterion)
            
    data_parameters_4 = [data_possibilist_4['real_regret_' + criterion + '_zero'][data_possibilist_4['inconsistency'] != 0],
                         data_possibilist_4['real_regret_' + criterion + '_ignorance'][data_possibilist_4['inconsistency'] != 0],
                         data_l_out_of_n_4['real_regret_' + criterion + '_l_out_of_n'][data_possibilist_4['inconsistency'] != 0],
                         mcs_best_real_regret[:,1][data_possibilist_4['inconsistency'] != 0],
                         mcs_confidence_real_regret[:,1][data_possibilist_4['inconsistency'] != 0],
                         mcs_size_confidence_real_regret[:,1][data_possibilist_4['inconsistency'] != 0],
                         mcs_size_real_regret[:,1][data_possibilist_4['inconsistency'] != 0],
                         data_epsilon_4['real_regret_' + criterion + '_epsilon'][data_possibilist_4['inconsistency'] != 0]]    
    
    data_parameters_5 = [data_possibilist_5['real_regret_' + criterion + '_zero'][data_possibilist_5['inconsistency'] != 0],
                         data_possibilist_5['real_regret_' + criterion + '_ignorance'][data_possibilist_5['inconsistency'] != 0],
                         data_l_out_of_n_5['real_regret_' + criterion + '_l_out_of_n'][data_possibilist_5['inconsistency'] != 0],
                         mcs_best_real_regret[:,0][data_possibilist_5['inconsistency'] != 0],
                         mcs_confidence_real_regret[:,0][data_possibilist_5['inconsistency'] != 0],
                         mcs_size_confidence_real_regret[:,0][data_possibilist_5['inconsistency'] != 0],
                         mcs_size_real_regret[:,0][data_possibilist_5['inconsistency'] != 0],
                         data_epsilon_5['real_regret_' + criterion + '_epsilon'][data_possibilist_5['inconsistency'] != 0]]
    
    data_all = data_parameters_4 + data_parameters_5
    
    for d in data_all:
        print(stats.shapiro(d)[1] < 0.05)
    
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    j = 0
    for data_questions in [data_parameters_4, data_parameters_5]:
        y = np.asarray([0,5,10,15,20,25,30,35]) + np.ones(8) * j
        for i in range(0, len(data_questions)):
            
            data = data_questions[i]
            data_mean = np.mean(data)
            data_IC = stats.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data),
                              scale=stats.sem(data))
            ax.plot((data_IC[0],data_IC[1]),(y[i],y[i]), '-', color = cmap(j/1))
            ax.plot((data_mean,data_mean),(y[i],y[i]), '|', color = cmap(j/1))
        j+=1
    ax.set_xlim(-0.005,0.3)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[1:7] = ['zero', 'ignorence', 'l out of n', 'MCS best', 'MCS conf',
                   'MCS size + conf', 'MCS size', 'epsilon']
    ax.set_yticklabels(labels)
    line_4 = mlines.Line2D([], [], color=cmap(0/1), marker='|', linestyle='-',
                           label='4')
    line_5 = mlines.Line2D([], [], color=cmap(1/1), marker='|', linestyle='-',
                           label='5')

    plt.legend(loc="upper left", handles=[line_4, line_5])
    ax.set_xlabel('Real regret', fontsize = 9)
    fig.set_dpi(300.0)
    tikzplotlib.save(path_results + conf_type + '_' + criterion + '.tex')
    plt.savefig(path_results + conf_type + '_' + criterion + '.png', dpi=300, bbox_inches = 'tight')