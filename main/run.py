# coding: utf-8

from keras.utils.np_utils import to_categorical
import numpy as np
import os
import argparse
from data_utils import *
from model import create_model

parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true')
args = parser.parse_args()
train = args.train

train_stories, test_stories, vocab_size, story_maxlen, question_maxlen, answer_maxlen, word_idx, voc = extract_data()

inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, question_maxlen,
                                                               answer_maxlen)
answers_train = np.array([to_categorical(i, vocab_size) for i in answers_train])
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, question_maxlen,
                                                            answer_maxlen)
answers_test = np.array([to_categorical(i, vocab_size) for i in answers_test])

GLOVE_DIR = 'data/glove/glove.6B/'
EMBEDDING_DIM = 100

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word_idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = create_model(vocab_size, story_maxlen, question_maxlen, answer_maxlen)

if train:
    model.fit([inputs_train, queries_train], answers_train, batch_size=2000, epochs=500,
              validation_data=([inputs_test, queries_test], answers_test))
    model.save_weights("weights.h5")
else:
    model.load_weights('weights.h5')
    print(model.evaluate([inputs_test, queries_test], answers_test, batch_size=2000))
