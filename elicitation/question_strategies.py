# -*- coding: utf-8 -*-
"""Strategies to determine the questions."""

import numpy as np

class RandomQuestionStrategy():
    """Random questions"""

    def __init__(self, alternatives):
        """
        Parameters
        ----------
        alternatives : array_like
            Alternatives.

        Returns
        -------
        None.

        """
        self._alternatives = alternatives
        self._nb_alternatives = len(alternatives)
        self._visited_pairs = np.zeros((self._nb_alternatives,self._nb_alternatives))
        np.fill_diagonal(self._visited_pairs, 1)

    def set_pair_visited(self, alt_idx_1,alt_idx_2):
        """
        Indicate that a pair of alternatives was already compared

        Parameters
        ----------
        alt_idx_1 : integrer
            Alternative 1.
        alt_idx_2 : integrer
            Alternative 2.

        Returns
        -------
        None.

        """
        self._visited_pairs[alt_idx_1,alt_idx_2] = 1
        self._visited_pairs[alt_idx_2,alt_idx_1] = 1

    def give_candidate(self):
        """
        Get a candidate (random).

        Returns
        -------
        candidate_alt : array_like
            The candidate alternative.
        candidate_alt_id : integrer
            Its indice.

        """
        candidate_alt_id = -1
        alternatives_random = np.arange(0, self._nb_alternatives)
        np.random.shuffle(alternatives_random)
        for i in alternatives_random:
            if not np.all(self._visited_pairs[i,:] == 1):
                candidate_alt_id = i
                break
        candidate_alt = self._alternatives[candidate_alt_id]
        return candidate_alt, candidate_alt_id

    def give_oponent(self, candidate_alt_id):
        """
        Get a random opponent

        Parameters
        ----------
        candidate_alt_id : integer
            The indice of the candidate alternative.

        Returns
        -------
        worst_alt : array_like
            The oponent alternative.
        worst_alt_id : integrer
            Its indice.

        """
        worst_alt_id = -1
        alternatives_random = np.arange(0, self._nb_alternatives)
        np.random.shuffle(alternatives_random)
        for j in alternatives_random:
            if self._visited_pairs[candidate_alt_id,j] != 1:
                worst_alt_id = j
                break
        self._visited_pairs[candidate_alt_id, worst_alt_id] = 1
        self._visited_pairs[worst_alt_id, candidate_alt_id] = 1
        worst_alt = self._alternatives[worst_alt_id]
        return worst_alt, worst_alt_id
    