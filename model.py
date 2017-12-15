#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:10:56 2017

@author: diana
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from MLP import MLP
import itertools

def disableTrainingForEmbeddings(model, *embeddingLayers):
    for e in embeddingLayers:
        e.weight.requires_grad = False

class DependencyParseModel(nn.Module):
    def __init__(self, word_embeddings_dim, tag_embeddings_dim, vocabulary_size, tag_uniqueCount, label_uniqueCount, pretrainedWordEmbeddings=None, pretrainedTagEmbeddings=None):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocabulary_size, word_embeddings_dim)
        # if pretrainedWordEmbeddings.any():
        #     assert pretrainedWordEmbeddings.shape == (vocabulary_size, word_embeddings_dim)
        #     self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrainedWordEmbeddings))
        #
        self.tag_embeddings = nn.Embedding(tag_uniqueCount, tag_embeddings_dim)

        # if pretrainedTagEmbeddings.any():
        #     assert pretrainedTagEmbeddings.shape == (tag_uniqueCount, tag_embeddings_dim)
        #     self.tag_embeddings.weight.data.copy_(torch.from_numpy(pretrainedTagEmbeddings))
        #
        # # Save computation time by not training already trained word vectors
        # disableTrainingForEmbeddings(self, self.word_embeddings, self.tag_embeddings)
        
        self.inputSize = word_embeddings_dim + tag_embeddings_dim # The number of expected features in the input x
        self.hiddenSize = self.inputSize  #* 2 # 512? is this the same as outputSize?
        self.nLayers = 2
        
        self.biLstm = nn.LSTM(self.inputSize, self.hiddenSize, self.nLayers, bidirectional=True)
        
        self.nDirections = 2
        self.batch = 1
        self.hidden, self.cell = self.initHiddenCellState()
        
    def initHiddenCellState(self):
        hiddenState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize))
        cellState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize))
        
        return hiddenState, cellState
    
    def forward(self, words_tensor, tags_tensor):
        wordEmbeds = self.word_embeddings(words_tensor)
        tagEmbeds = self.tag_embeddings(tags_tensor)
        inputTensor = torch.cat((wordEmbeds, tagEmbeds), 1)
        hVector, (self.hidden, self.cell) = self.biLstm(inputTensor.view(len(words_tensor), 1, -1), (self.hidden, self.cell))


        return hVector
