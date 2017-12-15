# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from conlluFilesOperations import ConlluFileReader
from dataProcessor import DataProcessor
from wordEmbeddingsReader import GloVeFileReader
from gensim.models import Word2Vec
import numpy as np
from model import DependencyParseModel
import torch.nn as nn
from torch.autograd import Variable
from random import shuffle
import time
import matplotlib.pyplot as plt
from MLP import MLP
import itertools


unknownMarker = '<unk>'

sentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
sentencesDependencies = sentencesReader.readSentencesDependencies(unknownMarker)

dataProcessor = DataProcessor(sentencesDependencies)
w2i, t2i, l2i, i2w, i2t, i2l = dataProcessor.buildDictionaries()
sentencesInWords, sentencesInTags = dataProcessor.getTrainingSetsForWord2Vec()

word_embeddings_dim = 50
posTags_embeddings_dim = 50
minCountWord2Vec_words = 5
minCountWord2Vec_tags = 0

# Train the POS tags
POSTagsModel = Word2Vec(sentencesInTags, size=posTags_embeddings_dim, window=5, min_count=minCountWord2Vec_tags, workers=4)

# Read the word embeddings
wordEmbeddingsReader = GloVeFileReader(r"GloVe/glove.6B.50d.txt")
wordsEmbeddings = wordEmbeddingsReader.readWordEmbeddings()

# Or train the embeddings too
wordsModel = Word2Vec(sentencesInWords, size=word_embeddings_dim, window=5, min_count=minCountWord2Vec_words, workers=4)

# LSTM training prep
vocabularySize = len(w2i)
tagsUniqueCount = len(t2i)
labelsUniqueCount = len(l2i)

pretrainedWordEmbeddings = np.empty((vocabularySize, word_embeddings_dim))
for k,v in i2w.items():
    if v in wordsEmbeddings:
        pretrainedWordEmbeddings[k] = wordsEmbeddings[v]
    elif v in wordsModel.wv.vocab:
        pretrainedWordEmbeddings[k] = wordsModel[v]
    else:
        pretrainedWordEmbeddings[k] = wordsModel[unknownMarker]

pretrainedTagEmbeddings = np.empty((tagsUniqueCount, posTags_embeddings_dim))
for k,v in i2t.items():
    assert v in POSTagsModel.wv.vocab
    pretrainedTagEmbeddings[k] = POSTagsModel[v]

mlpForScoresInputSize = (word_embeddings_dim + posTags_embeddings_dim) * 2

model = DependencyParseModel(word_embeddings_dim, posTags_embeddings_dim, vocabularySize, tagsUniqueCount, labelsUniqueCount, pretrainedWordEmbeddings, pretrainedTagEmbeddings)
mlpArcsScores = MLP(mlpForScoresInputSize * 2, mlpForScoresInputSize, 1)
mlpLabels = MLP(mlpForScoresInputSize * 2, mlpForScoresInputSize * 2, labelsUniqueCount)

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = nn.ParameterList(list(parameters))
optimizer = torch.optim.Adam(parameters, lr=0.1, weight_decay=1e-5)


epochs = 10
lossgraph = []
counter = 0
outputarray = []
outputarrayarcs = []
outputarraylabels = []

dummyoutput = Variable(torch.randn((18, 1, 200)))

# Calculate loss
loss = nn.CrossEntropyLoss()

print('start training..')


# shuffle(sentencesDependencies)
# model.hiddenState, model.cellState = model.initHiddenCellState()
for epoch in range(epochs):
    starttime = time.time()
    counter = 0
    total_output = 0
    for s in sentencesDependencies:
        # Clear hidden and cell previous state
        model.hiddenState, model.cellState = model.initHiddenCellState()

        # clear gradients
        optimizer.zero_grad()

        sentenceInWords, sentenceInTags = s.getSentenceInWordsAndInTags()

        wordsToIndices = [w2i[w] for w in sentenceInWords]
        words_tensor = Variable(torch.LongTensor(wordsToIndices))

        tagsToIndices = [t2i[t] for t in sentenceInTags]
        tags_tensor = Variable(torch.LongTensor(tagsToIndices))

        arcs_refdata = s.getHeadsForWords()
        arcs_target = Variable(torch.from_numpy(arcs_refdata).long(), requires_grad=False)
        labels_refdata = s.getLabelsForWords(l2i)
        labels_target = Variable(torch.from_numpy(labels_refdata).long(), requires_grad=False)
        # Forward pass
        hVector = model(words_tensor, tags_tensor)

        perms = list(itertools.product([x for x in range(len(sentenceInWords))], repeat=2))
        arcsTensor = Variable(torch.FloatTensor(len(sentenceInWords) + 1, len(sentenceInWords) + 1).zero_())
        for perm in perms:
            arcsTensor[perm[0] + 1, perm[1] + 1] = mlpArcsScores(hVector[perm[0], :, :], hVector[perm[1], :, :])

        # initiliaze dummy targets
        dummyoutputarcs = Variable(torch.LongTensor(len(sentenceInWords) + 1).zero_(), requires_grad=False)
        dummyoutputlabels = Variable(torch.LongTensor(len(sentenceInWords)).zero_(), requires_grad=False)

        arcs_loss = loss(arcsTensor, arcs_target)

        labelTensor = Variable(torch.FloatTensor(len(sentenceInWords), labelsUniqueCount).zero_())
        for i, head in enumerate(labels_refdata):
            if head == 0:
                continue
            labelTensor[i, :] = mlpLabels(hVector[i - 1, :, :], hVector[head - 1, :, :])


        label_loss = loss(labelTensor, labels_target)
        total_loss = arcs_loss + label_loss
        total_loss.backward(retain_graph=True)
        # print(list(model.parameters())[0].grad)

        optimizer.step()

        counter += 1

        if counter == 3:
            break
    print(epoch)


# print(outputarray)
# print(lossgraph)
#
# date = str(time.strftime("%d_%m_%H_%M"))
# savename = "DependencyParserModel_" + date + ".pkl"
# imagename = "DependencyParserModel_" + date + ".jpg"
#
# torch.save(model.state_dict(), savename)
#
# fig, axes = plt.subplots(2,2)
# axes[0, 0].plot(lossgraph)
# axes[0, 1].plot(outputarray)
# axes[1, 0].plot(outputarrayarcs)
# axes[1, 1].plot(outputarraylabels)
# axes[0, 0].set_title('Loss per epoch')
# axes[0, 1].set_title('Loss per sentence')
# axes[1, 0].set_title('Loss arcs MLP')
# axes[1, 1].set_title('Loss label MLP')
# fig.subplots_adjust(hspace=0.5)
# fig.subplots_adjust(wspace=0.5)
# plt.savefig(imagename)
