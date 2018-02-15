# -*- coding: utf-8 -*-
__author__ = 'Min'

import numpy as np
import re
import random
from collections import Counter
import math


class Vocab:
    def __init__(self, spelling, id):
        self.spelling = spelling
        self.id = id


class Word:
    def __init__(self, vocab, vector):
        self.vocab = vocab
        self.vector = vector


def softmax(x):
    """
    Calculate softmax based probability for given input vector
    :param x: numpy array/list
    :return: softmax of input array
    """
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def corpus2window(splitted_corpus, window_size):
    """
    Convert a word to one-hot encoded vector
    return center word and context word vectors

    :param corpus: corpus to change
    :return: {'center_word':center word obj, 'context_words': context words objects list}
    """
    corpus = []
    for _corpus in splitted_corpus:
        corpus += _corpus
    splitted_corpus = corpus
    return_array = []
    vocab_array = [Vocab(spelling=splitted_corpus[0], id=1)]  # First vocab
    for i in range(len(splitted_corpus)):
        center_word_vec = np.zeros(len(splitted_corpus))
        center_word_vec[i] = 1.0
        center_word_vocab = None

        for v in vocab_array:
            if v.spelling == splitted_corpus[i]:
                # not a new word
                center_word_vocab = v
                break

        if center_word_vocab is None:
            # New word
            center_word_vocab = Vocab(spelling=splitted_corpus[i], id=len(vocab_array)+1)
            vocab_array.append(center_word_vocab)

        context_words = []
        for window in range(i-window_size, i+window_size+1):
            if 0 <= window < len(splitted_corpus) and window != i:
                context_word_vec = np.zeros(len(splitted_corpus))
                context_word_vec[window] = 1.0

                window_word_vocab = None

                for v in vocab_array:
                    if v.spelling == splitted_corpus[window]:
                        # not a new word
                        window_word_vocab = v
                        break
                if window_word_vocab is None:
                    window_word_vocab = Vocab(spelling=splitted_corpus[window], id=len(vocab_array) + 1)
                    vocab_array.append(window_word_vocab)
                context_words.append(Word(vocab=window_word_vocab, vector=context_word_vec))
        return_array.append(
            {
                'center_word': Word(vocab=center_word_vocab, vector=center_word_vec),
                'context_words': context_words
            }
        )
    return return_array


def decision(probability):
    return random.random() < probability


def subsample(splitted_corpus):
    """
    Subsample words in corpus.
    threshold: 1e-5
    :param splitted_corpus: splitted_corpus to subsample
    :return: subsampled corpus
    """
    threshold = 1e-5
    whole_corpus_words = []
    for words in splitted_corpus:
        whole_corpus_words += words

    word_counts = Counter(whole_corpus_words)
    total_count = len(whole_corpus_words)

    survive_prob = {
        word: 1 - np.sqrt(threshold / count / total_count)
        for word, count in word_counts.items()
    }

    return_corpus = []
    for words in splitted_corpus:
        return_corpus.append([word for word in words if decision(survive_prob[word])])
    return return_corpus


def split(corpus):
    """
    split the corpus on special characters and space
    :param corpus: corpus to split
    :return: splitted corpus
    """
    return [re.findall(r'[^a-zA-Z0-9\s]|\w+', text) for text in corpus]


def cosine_similarity(v1,v2):
    """
    get cosine similarity.
    (v1 dot v2)/{||v1||*||v2||)
    :param v1:
    :param v2:
    :return:
    """
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)
