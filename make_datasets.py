# -*- coding: utf-8 -*-
"""Create datasets with questions and answers"""

import os
import sys
import time
import pickle
import multiprocessing
from multiprocessing import Value
import numpy as np
from alternatives.data_preparation import generate_alternatives_score
from elicitation.models import ModelWeightedSum
from elicitation.elicitation import make_questions_random

nb_parameters = 4
nb_questions = 15
nb_repetitions = 300
nb_alternatives = 50
conf_type = 'uniform'
path = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'

def init_globals(counter):
    global cnt
    cnt = counter

def conf_set():

    if conf_type == "strong":
        confidence_values = np.round(np.random.beta(7, 2, size = ((nb_repetitions, nb_questions))),
                                     decimals = 2)
    elif conf_type == "weak":
        confidence_values = np.round(np.random.beta(2, 7, size = ((nb_repetitions, nb_questions))),
                                     decimals = 2)
    elif conf_type == "intermediate":
        confidence_values = np.round(np.random.beta(5, 5, size = ((nb_repetitions, nb_questions))),
                                     decimals = 2)
    elif conf_type == "uniform":
        confidence_values = np.round(np.random.uniform(0.01, 0.99, size = ((nb_repetitions, nb_questions))),
                                     decimals = 2)
    else:
        raise NotImplementedError("I did not code that.")

    random_mask = np.random.uniform(size = (nb_repetitions, nb_questions))
    rational = np.where(random_mask <= confidence_values + (1-confidence_values)/2, 1, 0)
    non_zeros_lines = np.count_nonzero(rational, axis = 1) #We add an error if none.
    for j in range(0, nb_repetitions):
        if non_zeros_lines[j] == nb_questions:
            rational[j, np.random.randint(0, nb_questions)] = 0
    return confidence_values, rational

def make_dataset(alternatives, model_values, rational):
    model = ModelWeightedSum(model_values)
    res = make_questions_random(alternatives, model, nb_questions, rational)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

if __name__ == '__main__':

    alternatives_all = np.zeros((nb_repetitions, nb_alternatives, nb_parameters))
    for i in range(0, nb_repetitions):
        alternatives_all[i,:,:] = generate_alternatives_score(nb_alternatives,
                                                              nb_parameters = nb_parameters,
                                                              value = nb_parameters/2)
    model_values_all = np.random.dirichlet(np.ones(nb_parameters), size = nb_repetitions)
    confidence_values_all, rational_all = conf_set()

    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        dataset = pool.starmap(make_dataset, zip(alternatives_all, model_values_all,
                                                 rational_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time dataset : ", time.time() - start_time)

    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/dataset.pk','wb') as f:
        d = {}
        d['alternatives'] = alternatives_all
        d['model'] = model_values_all
        d['confidence'] = confidence_values_all
        d['rational'] = rational_all
        d['A'] = np.asarray([d['A'] for d in dataset])
        d['b'] = np.asarray([d['b'] for d in dataset])
        pickle.dump(d,f)
