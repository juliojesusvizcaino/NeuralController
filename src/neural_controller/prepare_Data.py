#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

inputs = [[1, 2], [2, 3], [3, 5]]
outputs = [[1, 2, 3, 4], [1, 2], [2, 3, 4, 5]]

def make_example(inputs, outputs):
    ex = tf.train.SequenceExample()
    ex_inputs = ex.feature_lists.feature_list["inputs"]
    ex_outputs = ex.feature_lists.feature_list["outputs"]

    for inp, outp in zip(inputs, outputs):
        ex_inputs.feature.add().float_list.value.append(inp)
        ex_outputs.feature.add().float_list.value.append(outp)

    return ex

data = make_example(inputs, outputs)

print(data)