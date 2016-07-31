#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from math import sin, exp
from random import random, uniform

from simulatedAnnealing import simulated_annealing


def error(x):
    return 0.3*sin(x) + sin(1.3*x) + 0.9*sin(4.2*x) + 0.002*random()


def neighbour(x):
    y = x + uniform(-1, 1)
    if y > 10:
        y = 10
    if y < 0:
        y = 0
    return y


def acceptance(error, t):
    if error > 0:
        return 1.0
    else:
        return exp(error/t)


def stop_condition(t):
    return True if t < 0.001 else False


def main():
    init_pos = -10
    init_error = error(init_pos)
    best = simulated_annealing(init_pos, init_error, error, neighbour, acceptance, stop_condition, 100.0)
    print best

if __name__ == "__main__":
    main()