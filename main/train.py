# coding: utf-8

from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Dropout, Masking
from keras.layers import concatenate
from keras.layers import GRU, BatchNormalization, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from functools import reduce
import numpy as np
import re
import os


def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)', sent) if x.strip()]


def parse_stories(lines, n=True):
    data = []
    story = []
    for line in lines:
        line = line.strip()
        if n:
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
        else:
            pass
        if '?' in line:
            if n:
                q, a, supporting = line.split('\t')
                a = tokenize(a.replace(',', ' '))
            else:
                q = line
            q = tokenize(q)
            substory = [x for x in story if x]
            if n:
                data.append((substory, q, a))
            else:
                data.append((substory, q))
            story.append('')
            if not n:
                story = []
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, max_length=None):
    with open(f) as file:
        data = parse_stories(file.readlines())
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if
            not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen, answer_maxlen, n=False):
    X = []
    Xq = []
    Y = []
    if n:
        for story, query in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in query]
            X.append(x)
            Xq.append(xq)
        return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen)
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        xa = [word_idx[w] for w in answer]
        X.append(x)
        Xq.append(xq)
        Y.append(xa)
    return (pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen),
            pad_sequences(Y, maxlen=answer_maxlen))


challenges = {"1"        : "qa1_single-supporting-fact",
              "2": "qa2_two-supporting-facts",
              "3"        : "qa3_three-supporting-facts",
              "4": "qa4_two-arg-relations",
              "5": "qa5_three-arg-relations",
              "6"        : "qa6_yes-no-questions",
              "7": "qa7_counting",
              "8": "qa8_lists-sets",
              "9"        : "qa9_simple-negation",
              "10": "qa10_indefinite-knowledge",
              "11": "qa11_basic-coreference",
              "12"       : "qa12_conjunction",
              "13": "qa13_compound-coreference",
              "14": "qa14_time-reasoning",
              "15"       : "qa15_basic-deduction",
              "16": "qa16_basic-induction",
              "17": "qa17_positional-reasoning",
              "18"       : "qa18_size-reasoning",
              "19": "qa19_path-finding",
              "20": "qa20_agents-motivations",
              "all": "all"}

challenge_type = 'all'
challenge = challenges[challenge_type] + '_{}.txt'

DATA_DIR = 'data/babi/en-10k'

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(os.path.join(DATA_DIR, challenge.format('train')), max_length=200)
test_stories = get_stories(os.path.join(DATA_DIR, challenge.format('test')), max_length=200)

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + answer)
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
answer_maxlen = max(map(len, (x for _, _, x in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen,
                                                               answer_maxlen)
answers_train = np.array([to_categorical(i, vocab_size) for i in answers_train])
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen,
                                                            answer_maxlen)
answers_test = np.array([to_categorical(i, vocab_size) for i in answers_test])
print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')

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

input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

hid_dim = 50

context_rec = Masking()(input_sequence)
context_rec = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix])(context_rec)
context_rec = BatchNormalization()(context_rec)
context_rec = Bidirectional(GRU(hid_dim, return_sequences=True))(context_rec)
context_rec = BatchNormalization()(context_rec)
context_rec = Dropout(0.15)(context_rec)

question_rec = Masking()(question)
question_rec = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix])(question_rec)
question_rec = BatchNormalization()(question_rec)
question_rec = Bidirectional(GRU(hid_dim))(question_rec)
question_rec = BatchNormalization()(question_rec)
question_rec = RepeatVector(inputs_train.shape[1])(question_rec)
question_rec = Dropout(0.15)(question_rec)

answer = concatenate([context_rec, question_rec])

answer = GRU(hid_dim)(answer)
answer = BatchNormalization()(answer)
answer = RepeatVector(answer_maxlen)(answer)
answer = Dropout(0.2)(answer)
answer = GRU(hid_dim, return_sequences=True)(answer)
answer = BatchNormalization()(answer)
answer = Dropout(0.2)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

# build the final model
model = Model([input_sequence, question], answer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train
model.fit([inputs_train, queries_train], answers_train, batch_size=2000, epochs=500,
          validation_data=([inputs_test, queries_test], answers_test))

model.save_weights("weights.h5")

print("Training finished")
