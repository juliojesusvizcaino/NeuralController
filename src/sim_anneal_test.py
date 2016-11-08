#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from math import sin, exp
from random import random, uniform

from simulatedAnnealing import simulated_annealing


def error(x):
    return 0.3*sin(x['x']) + sin(1.3*x['x']) + 0.9*sin(4.2*x['x']) + x['y']**2 + 0.002*random()


def neighbour(x):
    # y = map(lambda x: x + uniform(-1, 1), x)
    # y = map(lambda x: 10 if x > 10 else x, y)
    # y = map(lambda x: 0 if x < 0 else x, y)
    y = {key: value + uniform(-1, 1) for key, value in x.items()}
    y = {key: 10 if value > 10 else value for key, value in y.items()}
    y = {key: 0 if value < 0 else value for key, value in y.items()}
    return y


def acceptance(error, t):
    if error > 0:
        return 1.0
    else:
        return exp(error/t)


def stop_condition(t):
    return True if t < 0.001 else False


def main():
    init_pos = {'x': -10, 'y': 5}
    init_error = error(init_pos)
    best = simulated_annealing(init_pos, init_error, error, neighbour, acceptance, stop_condition, 100.0, lambda x: 0.98*x)
    print best

if __name__ == "__main__":
    main()