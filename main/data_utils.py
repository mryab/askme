# coding=utf-8

import re
import os
from functools import reduce
from keras.preprocessing.sequence import pad_sequences


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


def vectorize_stories(data, word_idx, story_maxlen, question_maxlen, answer_maxlen, n=False):
    X = []
    Xq = []
    Y = []
    if n:
        for story, question in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in question]
            X.append(x)
            Xq.append(xq)
        return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen)
    for story, question, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in question]
        xa = [word_idx[w] for w in answer]
        X.append(x)
        Xq.append(xq)
        Y.append(xa)
    return (pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=question_maxlen),
            pad_sequences(Y, maxlen=answer_maxlen))


def vectorize_story(story, word_idx, story_maxlen):
    xq = [[word_idx[w] for sent in story.lower().split('\n') for w in tokenize(sent)]]
    return pad_sequences(xq, maxlen=story_maxlen)


def vectorize_question(question, word_idx, question_maxlen):
    xq = [[word_idx[w] for w in tokenize(question.lower())]]
    return pad_sequences(xq, maxlen=question_maxlen)


def extract_data(challenge_type='all', test_data=True, lower=False):
    challenge = challenge_type + '_{}.txt'
    data_dir = 'data/babi/en-10k'

    train_stories = get_stories(os.path.join(data_dir, challenge.format('train')), max_length=100)
    if test_data:
        test_stories = get_stories(os.path.join(data_dir, challenge.format('test')), max_length=100)
    else:
        test_stories = [[[], [], []]]
    vocab = set()
    for story, q, answer in train_stories + test_stories:
        vocab |= set(story + q + answer)
    vocab = sorted(vocab)

    vocab_size = len(vocab) + 1
    story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
    question_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
    answer_maxlen = max(map(len, (x for _, _, x in train_stories + test_stories)))

    if lower:
        word_idx = dict((c.lower(), i + 1) for i, c in enumerate(vocab))
    else:
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    voc = {v: k for k, v in word_idx.items()}
    voc[0] = ""

    return train_stories, test_stories, vocab_size, story_maxlen, question_maxlen, answer_maxlen, word_idx, voc
