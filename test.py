# -*- coding: utf-8 -*-
__author__ = 'Min'

"""
Test code
"""

from word2vec import Word2Vec
corpus = ["You are really great", "You are also very awesome"]

pl_dimension = 4
hl_dimension = 4
window_size = 2
learning_rate = 0.1


model = Word2Vec(corpus, pl_dimension, hl_dimension, window_size, learning_rate)
model.train()