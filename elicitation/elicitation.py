# -*- coding: utf-8 -*-
"""Elicitation"""

import time
import numpy as np
from alternatives.data_preparation import get_pareto_efficient_alternatives
from elicitation.question_strategies import RandomQuestionStrategy
from elicitation.dm import get_choice_fixed
from elicitation.choice_calculation import pmr_polytope, min_polytope, max_polytope
from elicitation.focal_set import compute_epmr_emr, compute_emax_emin
from elicitation.choice_strategies import minimax_regret_choice, maximax_choice, maximin_choice
from elicitation.polytope import Polytope, construct_constrainst, cut_polytope, intersection_checker

def make_questions_random(alternatives, model, nb_questions, rational):
    """
    Possibilist elicitation with CSS.

    Parameters
    ----------
    alternatives : array_like
        Alternatives.
    model : Model
        Model.
    nb_questions : integer
        Number of questions.
    rational : list
        To know if some answers should be rational or not.
    Returns
    -------
    dict
        Questions, answers and some info.

    """
    alternatives = get_pareto_efficient_alternatives(alternatives) #Get rid of non optimal solutions.
    question_strategy = RandomQuestionStrategy(alternatives)
    A_list = np.zeros((nb_questions, alternatives.shape[1]))
    b_list = np.zeros((nb_questions))
    for ite in range(0, nb_questions):
        candidate_alt, candidate_alt_id = question_strategy.give_candidate()
        worst_alt, _ = question_strategy.give_oponent(candidate_alt_id)
        choice = get_choice_fixed(candidate_alt, worst_alt, rational[ite], model)
        best_prefered = choice['accepted']
        new_constraint_a, new_constraint_b = construct_constrainst(candidate_alt, worst_alt, best_prefered, model)
        A_list[ite] = new_constraint_a
        b_list[ite] = new_constraint_b
    d = {}
    d['A'] = A_list
    d['b'] = b_list
    return d

def get_polytopes(model, confidence, A_ub, b_ub, t_norm = 'product',
                  min_possibility = 0):
    '''
    Get all the poytopes used in an elicitation

    Parameters
    ----------
    model : Model
        The model.
    confidence : list
        Confidence degrees.
        constraints_A_ub : array_like
            2-D array of values representing A for the constrainst Ax <= b.
        constraints_b_ub : array_like
            1-D array of values representing b for the constrainst Ax <= b.
    t_norm : string, optional
        Which T-norm to use. The default is 'product'.
    min_possibility : float, optional
        Min possibility to consider a polytope. The default is 0.

    Returns
    -------
    dict
        The polytopes, possibility for each polytope and some info.

    '''
    nb_questions = A_ub.shape[0]

    constraints = model.get_model_constrainsts()
    constraints_a = constraints['A_eq']
    constraints_b = constraints['b_eq']
    bounds = constraints['bounds']
    first_polytope = Polytope(None,None,constraints_a, constraints_b, bounds)

    polytope_list = []
    polytope_list.append(first_polytope)
    inconsistency_list = np.zeros(nb_questions)

    start_time = time.time()

    for ite in range(0, nb_questions):
        A = A_ub[ite,:]
        b = b_ub[ite]
        possibility_list = []
        new_polytope_list = []

        for polytope in polytope_list:
            side = intersection_checker(polytope, A, b)
            if side is None:
                return None
            '''If the new constrainst intersects with the current polytope:
            - Create two new ones,
            - Keep those with a suffissant possibility,
            - Delete the original polytope'''
            if side == 0:
                polytope_1, polytope_2 = cut_polytope(polytope, A, b, confidence[ite], t_norm)
                del polytope
                if polytope_1.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope_1)
                    possibility_list.append(polytope_1.get_possibility())
                else:
                    del polytope_1
                if polytope_2.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope_2)
                    possibility_list.append(polytope_2.get_possibility())
                else:
                    del polytope_2
            #Else, just update the possibility.
            else:
                if side == 1:
                    polytope.add_answer(A, b, 1, t_norm)
                else:
                    polytope.add_answer(-A, -b, 1-confidence[ite], t_norm)
                if polytope.get_possibility() > min_possibility:
                    new_polytope_list.append(polytope)
                    possibility_list.append(polytope.get_possibility())
                else:
                    del polytope

        inconsistency_list[ite] = 1-np.max(possibility_list)
        polytope_list = new_polytope_list

    d = {}
    d['time'] = time.time() - start_time
    d['inconsistency'] = inconsistency_list
    d['possibility_list'] = possibility_list
    d['polytope_list'] = polytope_list
    return d

def get_recommendation(things_list, possibility_list, alternatives, model,
                       criterion = "minimax regret", inconsistency_type = 'zero',
                       polytopes = True):
    """
    Determine the optimal recommendation according to some criterion from polytopes or values.

    Parameters
    ----------
    things_list : list
        List of polytopes or values.
    possibility_list : list
        List of possibility for each polytope.
    alternatives : array_like
        Alternatives.
    model : Model
        The model.
    criterion : string, optional
        Which criterion to use. The default is 'minimax regret'.
    inconsistency_type : string, optional
        Inconsistency in the EPMR/Emax. The default is 'zero'.
    polytopes : bool, optional
        Do we use polytopes in things_list. The default is True.
        
    Returns
    -------
    dict
        Information about the recommended alternative.

    """
    scores = model.get_model_score(alternatives)
    if criterion == "minimax regret":
        f_value = pmr_polytope
        f_ecompute = compute_epmr_emr
        f_choice = minimax_regret_choice
    elif criterion == 'maximax':
        f_value = max_polytope
        f_ecompute = compute_emax_emin
        f_choice = maximax_choice
    elif criterion == "maximin":
        f_value = min_polytope
        f_ecompute = compute_emax_emin
        f_choice = maximin_choice
    else:
        raise NotImplementedError("I didn't do that.")

    if polytopes is True:
        value_list = []
        for polytope in things_list:
            value_list.append(f_value(alternatives, polytope, model))
    else:
        value_list = things_list

    if criterion in ('maximax', 'maximin'):
        ecriterion = f_ecompute(value_list, possibility_list, criterion, inconsistency_type)
    elif criterion == "minimax regret":
        _, ecriterion = f_ecompute(value_list, possibility_list, inconsistency_type)
    else:
        raise NotImplementedError("I didn't do that.")

    _, best_alt_id, _ = f_choice(alternatives, ecriterion)
    regret = np.max(scores) - scores[best_alt_id]

    result = {}
    result['best_alternative'] = best_alt_id
    result['real_regret'] = regret
    if polytopes is True:
        result['value_list'] = value_list
    return result
