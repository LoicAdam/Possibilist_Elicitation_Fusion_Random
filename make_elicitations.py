# -*- coding: utf-8 -*-
"""Other MCSs"""

import sys
import time
import pickle
import multiprocessing
from multiprocessing import Value
import numpy as np
from scipy.optimize import linprog
from elicitation.elicitation import get_polytopes, get_recommendation
from elicitation.models import ModelWeightedSum
from elicitation.polytope import Polytope
from fusion.l_out_n import find_incorrect_answers, k_among_n_fusion
from fusion.mcs import get_answers, find_all_maximum_coherent_subsets, update_possibility_list

conf_type = 'uniform'
nb_questions = 15
nb_parameters = 4
path = 'data/criteria_' + str(nb_parameters) + '/' + str(conf_type) + '/questions_' + str(nb_questions) + '/'

def init_globals(counter):
    global cnt
    cnt = counter

def polytopes(model_values, confidence, A, b):
    list_polytopes = get_polytopes(ModelWeightedSum(model_values), confidence, A, b)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return list_polytopes

def recommendation_possibilist(polytope_list, possibility_list, alternatives,
                               model_values, criterion):
    res_zero = get_recommendation(polytope_list, possibility_list, alternatives,
                                  ModelWeightedSum(model_values), criterion, "zero")
    res_ignorence = get_recommendation(res_zero['value_list'], possibility_list, alternatives,
                                       ModelWeightedSum(model_values), criterion,
                                       "ignorance", polytopes = False)
    d = {}
    d['value_list'] = res_zero['value_list']
    d['best_alternative_zero'] = res_zero["best_alternative"]
    d['real_regret_zero'] = res_zero["real_regret"]
    d['best_alternative_ignorance'] = res_ignorence["best_alternative"]
    d['real_regret_ignorance'] = res_ignorence["real_regret"]
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return d

def get_number_errors(polytope_list):
    nb_detected_incorrect_answers = find_incorrect_answers(polytope_list)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return np.min(nb_detected_incorrect_answers)

def l_out_of_n(polytope_list, nb_detected_incorrect_answers):
    possibility_list = k_among_n_fusion(polytope_list, nb_questions - nb_detected_incorrect_answers,
                                        nb_questions)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return possibility_list

def recommendation_l_out_of_n(value_list, possibility_list, alternatives, model_values,
                              criterion):
    res = get_recommendation(value_list, possibility_list, alternatives,
                             ModelWeightedSum(model_values), criterion,
                             polytopes = False)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

def list_all_mcs(polytope_list, confidence):
    answers = get_answers(polytope_list, nb_questions)
    mcs_list = find_all_maximum_coherent_subsets(answers, nb_questions)
    d = {}
    d['mcs'] = mcs_list
    d['confidence'] = [confidence[mcs] for mcs in mcs_list]
    d['confidence_mean'] = np.asarray([np.mean(confidence[mcs]) for mcs in mcs_list])
    d['size'] = np.asarray([len(mcs) for mcs in mcs_list])
    d['answers'] = answers
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return d

def recommendation_all_mcs(mcs_list, polytope_list, value_list, answers,
                           alternatives, model_values, criterion):
    real_regret_list = []
    for i in range(0, len(mcs_list)):
        mcs = mcs_list[i]
        updated_possibility_list = update_possibility_list(answers, mcs, "product")
        real_regret_list.append(get_recommendation(value_list, updated_possibility_list,
                                                   alternatives, ModelWeightedSum(model_values),
                                                   criterion, polytopes = False))
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return real_regret_list

def epsilon_consistency(A_ub, b_ub, alternatives, model_values, criterion):
    model = ModelWeightedSum(model_values)
    constraints = model.get_model_constrainsts()
    A_eq = constraints['A_eq']
    b_eq = constraints['b_eq']
    bounds = constraints['bounds']
    n, p = A_ub.shape
    c = np.ones((p+n,1))
    c[0:p] = 0
    A_ub_new = np.hstack((A_ub, -np.identity(n)))
    A_eq_new = np.hstack((A_eq, np.ones((1,n))))
    bounds_new = bounds
    bounds_new = bounds_new + tuple((0, None) for _ in range(n))
    linprog_res = linprog(c, A_ub_new, b_ub, A_eq_new, b_eq, bounds_new,
                          method = 'highs')
    b_ub_new = b_ub + linprog_res.x[p:]
    new_polytope = Polytope(A_ub,b_ub_new,A_eq,b_eq, bounds)
    res = get_recommendation([new_polytope], [1], alternatives,
                             model, criterion)
    with cnt.get_lock():
        cnt.value += 1
        print(cnt.value)
    sys.stdout.flush()
    return res

if __name__ == '__main__':
    
    try:
        with open(path + 'dataset.pk','rb') as f:
            d = pickle.load(f)
    except IOError:  #file doesn't exist, no high-scores registered.
        d = {}

    alternatives_all = d['alternatives']
    model_values_all = d['model']
    confidence_values_all = d['confidence']
    rational_all = d['rational']
    A_all = d['A']
    b_all = d['b']
    nb_repetitions = alternatives_all.shape[0]

    number_of_workers = np.minimum(np.maximum(multiprocessing.cpu_count() - 2,1), 30)
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        polytopes = pool.starmap(polytopes, zip(model_values_all, confidence_values_all,
                                                A_all, b_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time polytopes : ", time.time() - start_time)
    
    polytope_all = [d['polytope_list'] for d in polytopes if d is not None]
    possibility_all = [d['possibility_list'] for d in polytopes if d is not None]
    inconsistency_all = [d['inconsistency'][-1] for d in polytopes if d is not None]
    
    nones = [i for i, x in enumerate(polytopes) if x is None]
    if len(nones) != 0:
        alternatives_all = np.delete(alternatives_all, nones, 0)
        model_values_all = np.delete(model_values_all, nones, 0)
        confidence_values_all = np.delete(confidence_values_all, nones, 0)
        rational_all = np.delete(rational_all, nones, 0)
        A_all = np.delete(A_all, nones, 0)
        b_all = np.delete(b_all, nones, 0)
        nb_repetitions = nb_repetitions - len(nones)
        with open(path + '/dataset.pk','wb') as f:
            d = {}
            d['alternatives'] = alternatives_all
            d['model'] = model_values_all
            d['confidence'] = confidence_values_all
            d['rational'] = rational_all
            d['A'] = A_all
            d['b'] = b_all
            pickle.dump(d,f)

    ### General possibilist elicitation ###

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        minimax_regret = pool.starmap(recommendation_possibilist,
                                      zip(polytope_all, possibility_all,
                                          alternatives_all, model_values_all,
                                          np.repeat("minimax regret", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations minimax regret: ", time.time() - start_time)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        maximax = pool.starmap(recommendation_possibilist,
                               zip(polytope_all, possibility_all,
                                   alternatives_all, model_values_all,
                                   np.repeat("maximax", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations maximax: ", time.time() - start_time)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        maximin = pool.starmap(recommendation_possibilist,
                               zip(polytope_all, possibility_all,
                                   alternatives_all, model_values_all,
                                   np.repeat("maximin", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations maximin: ", time.time() - start_time)

    with open(path + 'possibilist.pk','wb') as f:
        d = {}
        d['real_regret_minimax_regret_zero'] = np.asarray([d['real_regret_zero'] for d in minimax_regret])
        d['real_regret_minimax_regret_ignorance'] = np.asarray([d['real_regret_ignorance'] for d in minimax_regret])
        d['real_regret_maximax_zero'] = np.asarray([d['real_regret_zero'] for d in maximax])
        d['real_regret_maximax_ignorance'] = np.asarray([d['real_regret_ignorance'] for d in maximax])
        d['real_regret_maximin_zero'] = np.asarray([d['real_regret_zero'] for d in maximin])
        d['real_regret_maximin_ignorance'] = np.asarray([d['real_regret_ignorance'] for d in maximin])
        d['inconsistency'] = np.asarray(inconsistency_all)
        pickle.dump(d,f)

    ### K out of n ###

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        detected_errors = pool.starmap(get_number_errors, zip(polytope_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time number errors: ", time.time() - start_time)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        l_out_of_n_fusion = pool.starmap(l_out_of_n,
                                         zip(polytope_all, detected_errors))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time l_out_of_n: ", time.time() - start_time)

    minimax_regret_values = [d['value_list'] for d in minimax_regret]
    maximax_values = [d['value_list'] for d in maximax]
    maximin_values = [d['value_list'] for d in maximin]

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        minimax_regret_l_out_of_n = pool.starmap(recommendation_l_out_of_n,
                                                 zip(minimax_regret_values, l_out_of_n_fusion,
                                                     alternatives_all, model_values_all,
                                                     np.repeat("minimax regret", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations minimax regret l-out-of-n: ", time.time() - start_time)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        maximax_l_out_of_n = pool.starmap(recommendation_l_out_of_n,
                                          zip(maximax_values, l_out_of_n_fusion,
                                              alternatives_all, model_values_all,
                                              np.repeat("maximax", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations maximax l-out-of-n: ", time.time() - start_time)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        maximin_l_out_of_n = pool.starmap(recommendation_l_out_of_n,
                                          zip(maximin_values, l_out_of_n_fusion,
                                              alternatives_all, model_values_all,
                                              np.repeat("maximin", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations maximin l-out-of-n: ", time.time() - start_time)

    with open(path + 'l_out_of_n.pk','wb') as f:
        d = {}
        d['real_regret_minimax_regret_l_out_of_n'] = np.asarray([d['real_regret'] for d in minimax_regret_l_out_of_n])
        d['real_regret_maximax_l_out_of_n'] = np.asarray([d['real_regret'] for d in maximax_l_out_of_n])
        d['real_regret_maximin_l_out_of_n'] = np.asarray([d['real_regret'] for d in maximin_l_out_of_n])
        d['nb_errors_detected'] = detected_errors
        pickle.dump(d,f)
        
    ### MCS ###
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        mcs_all_res = pool.starmap(list_all_mcs, zip(polytope_all, confidence_values_all))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time list all MCS: ", time.time() - start_time)

    mcs_all = [d['mcs'] for d in mcs_all_res]
    mcs_all_confidence = [d['confidence'] for d in mcs_all_res]
    mcs_all_confidence_mean = [d['confidence_mean'] for d in mcs_all_res]
    mcs_all_size = [d['size'] for d in mcs_all_res]
    mcs_answers = [d['answers'] for d in mcs_all_res]
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        minimax_regret_mcs = pool.starmap(recommendation_all_mcs,
                                          zip(mcs_all, polytope_all, minimax_regret_values,
                                              mcs_answers, alternatives_all, model_values_all, 
                                              np.repeat("minimax regret", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendation minimax regret mcs: ", time.time() - start_time)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        maximax_mcs = pool.starmap(recommendation_all_mcs,
                                   zip(mcs_all, polytope_all, maximax_values,
                                       mcs_answers, alternatives_all, model_values_all,
                                       np.repeat("maximax", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendation maximax mcs: ", time.time() - start_time)
    
    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        maximin_mcs = pool.starmap(recommendation_all_mcs,
                                   zip(mcs_all, polytope_all, maximin_values,
                                       mcs_answers, alternatives_all, model_values_all,
                                       np.repeat("maximin", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendation maximin mcs: ", time.time() - start_time)
    
    with open(path + 'mcs.pk','wb') as f:
        d = {}
        d['mcs'] = mcs_all
        d['confidence'] = mcs_all_confidence
        d['confidence_mean'] = mcs_all_confidence_mean
        d['size'] = mcs_all_size
        d['real_regret_minimax_regret_mcs'] = [np.asarray([mcs['real_regret'] for mcs in mcs_list]) for mcs_list in minimax_regret_mcs]
        d['real_regret_maximax_mcs'] = [np.asarray([mcs['real_regret'] for mcs in mcs_list]) for mcs_list in maximax_mcs]
        d['real_regret_maximin_mcs'] = [np.asarray([mcs['real_regret'] for mcs in mcs_list]) for mcs_list in maximin_mcs]
        pickle.dump(d,f)
    
    ### Epsilon ###

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        minimax_regret_epsilon = pool.starmap(epsilon_consistency,
                                              zip(A_all, b_all, alternatives_all,
                                                  model_values_all, np.repeat("minimax regret", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations minimax regret epsilon: ", time.time() - start_time)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        maximax_epsilon = pool.starmap(epsilon_consistency,
                                       zip(A_all, b_all, alternatives_all,
                                           model_values_all, np.repeat("maximax", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations maximax epsilon: ", time.time() - start_time)

    start_time = time.time()
    cnt = Value('i', 0)
    with multiprocessing.Pool(initializer=init_globals, initargs=(cnt,), processes=number_of_workers) as pool:
        maximin_epsilon = pool.starmap(epsilon_consistency,
                                       zip(A_all, b_all, alternatives_all,
                                           model_values_all, np.repeat("maximin", nb_repetitions)))
    sys.stdout.flush()
    pool.close()
    pool.join()
    print("Time recommendations maximin epsilon: ", time.time() - start_time)

    with open(path + 'epsilon.pk','wb') as f:
        d = {}
        d['real_regret_minimax_regret_epsilon'] = np.asarray([d['real_regret'] for d in minimax_regret_epsilon])
        d['real_regret_maximax_epsilon'] = np.asarray([d['real_regret'] for d in maximax_epsilon])
        d['real_regret_maximin_epsilon'] = np.asarray([d['real_regret'] for d in maximin_epsilon])
        pickle.dump(d,f)
