#!/usr/bin/env python
# -*- coding: utf-8 -*-

# todo guardar la mejor posición
# todo hacer que stop_condition dependa de parámetros que se le pasan a la propia función
from copy import deepcopy
from random import random


def simulated_annealing(init, init_error, error, neighbour, acceptance, stop_condition,
                        init_temp, temperature_update=lambda x: x * 0.95):
    """
    Global minimum search using simulated annealing

    :param init: Initial parameters
    :type init: whatever
    :param init_error: Initial error
    :type init_error: float
    :param error: Error function
    :type error: function output_error = error(params)
    :param neighbour: Give neighbours of received param
    :type neighbour: function iterable = neighbour(params)
    :param acceptance: Probability of change when error is worse
    :type acceptance: function probability = acceptance(error_diff, temperature)
    :param stop_condition: Condition to stop at
    :type stop_condition: function boolean = stop_condition(temperature)
    :param init_temp: Temperature at which start
    :type init_temp: float
    :param temperature_update: Way of update temperature
    :type temperature_update: function new_temp = temperature_update(temp)
    :return: Optimum parameters
    """
    if stop_condition(init_temp):
        return init
    else:
        new, new_error = update_neighbour(init, init_error, neighbour, error, acceptance, init_temp)
        new_temp = temperature_update(init_temp)
        return simulated_annealing(new, new_error, error, neighbour, acceptance,
                                   stop_condition, new_temp, temperature_update)


def update_neighbour(init, init_error, neighbour, error, acceptance, temp):
    final_params = deepcopy(init)
    final_error = init_error
    for new_params in neighbour(final_params):
        new_error = error(new_params)
        if acceptance(final_error - new_error, temp) > random():
            final_params, final_error = new_params, new_error
    return final_params, final_error
