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

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen,
                                                               answer_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen,
                                                            answer_maxlen)
answers_test = np.array([to_categorical(i, vocab_size) for i in answers_test])

input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

hid_dim = 50

context_rec = Masking()(input_sequence)
context_rec = Embedding(input_dim=vocab_size, output_dim=100)(context_rec)
context_rec = BatchNormalization()(context_rec)
context_rec = Bidirectional(GRU(hid_dim, return_sequences=True))(context_rec)
context_rec = BatchNormalization()(context_rec)
context_rec = Dropout(0.15)(context_rec)

question_rec = Masking()(question)
question_rec = Embedding(input_dim=vocab_size, output_dim=100)(question_rec)
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

model = Model([input_sequence, question], answer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights("weights.h5")

print(model.evaluate([inputs_test, queries_test], answers_test, batch_size=1500))
