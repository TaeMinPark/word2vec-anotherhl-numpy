# -*- coding: utf-8 -*-
__author__ = 'Min'

import numpy as np
import utils


class Word2Vec:
    def __init__(self, corpus, pl_dimension, hl_dimension, window_size, learning_rate, training_method='cbow'):
        """
        Initializing Word2vec class.
        :param corpus: corpus to train
        :param pl_dimension: projection layer dimension
        :param hl_dimension: hidden layer dimension
        :param window_size: window size
        :param learning_rate: learning
        :param training_method: training method (skip-gram or cbow) default: cbow
        """
        self.corpus = corpus
        self.pl_dimension = pl_dimension
        self.hl_dimension = hl_dimension
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.loss = 0.0
        splitted_corpus = utils.split(corpus)
        subsampled_splitted_corpus = utils.subsample(splitted_corpus)
        self.corpus_window_array = utils.corpus2window(subsampled_splitted_corpus, 2)
        np.random.seed(100)
        self.weight1 = np.random.rand(len(self.corpus_window_array), pl_dimension)
        self.weight2 = np.random.rand(pl_dimension, hl_dimension)
        self.weight3 = np.random.rand(hl_dimension, len(splitted_corpus))
        self.training_method = training_method

    def train(self):
        """
        train method.
        train word2vec and print out progress
        :return: this function does not return anything
        """
        for i, current_window in enumerate(self.corpus_window_array):
            if self.training_method == 'cbow':
                self.weight1, self.weight2, self.weight3, self.loss = self.cbow(current_window['context_words'],
                                                      current_window['center_word'],
                                                      self.weight1, self.weight2, self.weight3,
                                                      self.loss, learning_rate, self.corpus_window_array)
            elif self.training_method == 'skip-gram':
                print('LMAO. Does not support skip-gram yet.')
            else:
                print('invalid training method')

            print(
                "Training word #{} \n-------------------- \n\n \t spelling = {},"
                    .format(i, current_window['center_word'].vector))
            print("\t weight1 = {}\n\t weight2 = {}\n\t weight3 = {}\n loss = {}"
                  .format(self.weight1, self.weight2, self.weight3, self.loss))

    def cbow(self, contexts, label, weight1, weight2, weight3, loss, learning_rate, corpus_window_array):
        """
        Implementation of Continuous Bag-of-Words Word2Vec model

        :param contexts: all the context words (thee represent the inputs)
        :param label: the center word (this represents the label) (word type)
        :param weight1: weights from the input to the projection layer
        :param weight2: weithts from the projection layer to hidden layer
        :param weight3: Weights from the hidden layer to output layer
        :param loss: float that represents the current value of the loss function
        :param learning_rate: learning rate
        :param corpus_window_array: array of corpus_window
        :return: updated weights and loss
        """
        p = np.mean([np.dot(weight1.T, x.vector) for x in contexts], axis=0)
        s = np.dot(weight2.T, p)
        h = np.tanh(s)
        u = np.dot(weight3.T, h)
        y = utils.softmax(u)

        e = y - label.vector

        dW3 = np.zeros(weight3.shape)
        dW2 = np.zeros(weight2.shape)
        dW1 = np.zeros(weight1.shape)

        # update weight3
        num_w3col, num_w3row = weight3.shape
        num_w2col, num_w2row = weight2.shape
        num_w1col, num_w1row = weight1.shape

        # backpropagation
        for a in range(num_w3col):
            for b in range(num_w3row):
                b_ = []
                for j in range(num_w2col):
                    p = []
                    for k in range(num_w1col):
                        p.append(np.dot(weight1[k][j], corpus_window_array[k]['center_word'].vector))
                    b_.append(np.dot(weight2[j][a], np.sum(p)))
                dW3[a][b] = np.dot(e[b], (np.tanh(np.sum(b_))))

        for a in range(num_w2col):
            for b in range(num_w2row):
                U = []
                for c in range(num_w1col):
                    p = []
                    for k in range(num_w1col):
                        p.append(np.dot(weight1[k][a], corpus_window_array[k]['center_word'].vector))
                    p = np.sum(p)

                    U.append(np.dot(e[c], np.dot(np.dot(weight3[b][c], np.tanh(np.dot(weight2, p))), p)))
                dW2[a][b] = np.sum(U)

        for a in range(num_w1col):
            for b in range(num_w1row):
                U = []
                for c in range(num_w1col):
                    L = []
                    for l in range(num_w3col):
                        L.append(np.dot(np.dot(np.dot(weight3[l][c], np.tanh(
                            np.dot(np.dot(weight2[b][l], weight1[a][b]),
                                   corpus_window_array[a]['center_word'].vector))),
                                               weight2[b][l]), corpus_window_array[a]['center_word'].vector))
                    U.append(np.dot(e[c], np.sum(L)))
                dW1[a][b] = np.sum(U)

        new_weight1 = weight1 - learning_rate * dW1
        new_weight2 = weight2 - learning_rate * dW2
        new_weight3 = weight3 - learning_rate * dW3

        loss += - float(u[label.vector == 1]) + np.log(np.sum(np.exp(u)))

        return new_weight1, new_weight2, new_weight3, loss

    def most_similar(self, word_to_compare, topN = 10):
        """
        return topN most similar words compared to word_to_compare
        use cosine similarity

        :param word_to_compare: word to compare
        :param topN: topN results
        :return: topN most similar words compared to word_to_compare. tuple: (similarity, Word obj)
        """

        word_to_compare_representation = np.dot(word_to_compare.vector, self.weight1)
        distances = []
        for center_word, contexts in self.corpus_window_array:
            distances.append(
                (utils.cosine_similarity(word_to_compare_representation, np.dot(center_word.vector, self.weight1)),
                 center_word)
            )

        distances.sort()
        for i in range(len(distances)):
            if distances[i][1].vocab == word_to_compare.vocab:
                del distances[i]
        return distances[-topN:]