#!/usr/bin/env python
# -*- coding: utf-8 -*-

# todo guardar la mejor posición
# todo hacer que stop_condition dependa de parámetros que se le pasan a la propia función
from random import random


def simulated_annealing(init, error, neighbour, acceptance, stop_condition,
                        init_temp, temperature_update=lambda x: x * 0.95):
    """Búsqueda del mínimo global usando el algoritmo simulated annealing"""
    if stop_condition(init_temp):
        return init
    else:
        new = update_neighbour(init, neighbour, error, acceptance, init_temp)
        new_temp = temperature_update(init_temp)
        return simulated_annealing(new, error, neighbour, acceptance, stop_condition, new_temp, temperature_update)


def update_neighbour(init, neighbour, error, acceptance, temp):
    neighbour_test = neighbour(init)
    if acceptance(error(neighbour_test), temp) > random():
        return neighbour_test
    else:
        return init

