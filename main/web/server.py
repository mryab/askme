import glob
import flask
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
            x = [word_idx[w.lower()] for w in story]
            xq = [word_idx[w.lower()] for w in query]
            X.append(x)
            Xq.append(xq)
        return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen)
    for story, query, answer in data:
        x = [word_idx[w.lower()] for w in story]
        xq = [word_idx[w.lower()] for w in query]
        xa = [word_idx[w.lower()] for w in answer]
        X.append(x)
        Xq.append(xq)
        Y.append(xa)
    return (pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen),
            pad_sequences(Y, maxlen=answer_maxlen))


def vectorize_story(story):
    xq = [[word_idx[w] for sent in story.lower().split('\n') for w in tokenize(sent)]]
    return pad_sequences(xq, maxlen=story_maxlen)


def vectorize_question(question):
    xq = [[word_idx[w] for w in tokenize(question.lower())]]
    return pad_sequences(xq, maxlen=query_maxlen)


app = flask.Flask(__name__)
model = None
train_stories = None
inputs_train, queries_train, answers_train = None, None, None
story_maxlen, query_maxlen = None, None
word_idx = None
voc = None


def init():
    """ Initialize web app """
    global model, inputs_train, queries_train, answers_train, voc, train_stories, story_maxlen, query_maxlen, word_idx

    challenge_type = 'all'
    challenge = challenge_type + '_{}.txt'

    DATA_DIR = '../data/babi/en-10k'

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

    word_idx = dict((c.lower(), i + 1) for i, c in enumerate(vocab))
    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen,
                                                                   answer_maxlen)
    answers_train = np.array([to_categorical(i, vocab_size) for i in answers_train])

    voc = {v: k for k, v in word_idx.items()}
    voc[0] = ""

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

    model.load_weights("../weights.h5")


def run():
    app.run(host='0.0.0.0', port=6006)


@app.route('/')
def index():
    return flask.render_template("index.html")


@app.route('/get/story', methods=['GET'])
def get_story():
    question_idx = np.random.randint(inputs_train.shape[0])
    story_txt, question_txt, correct_answer = train_stories[question_idx]
    # Format text
    story_txt = ' '.join(story_txt).replace(' . ', '.\n').replace(' .', '.')
    question_txt = ' '.join(question_txt).replace(' ?', '?')
    correct_answer = ' '.join(correct_answer)

    return flask.jsonify({"question_idx"  : question_idx, "story": story_txt, "question": question_txt,
                          "correct_answer": correct_answer})


@app.route('/get/answer', methods=['GET'])
def get_answer():
    question_idx = int(flask.request.args.get('question_idx'))
    user_question = flask.request.args.get('user_question', '')
    user_story = flask.request.args.get('user_story', '')
    if user_story == '':
        inp = inputs_train[question_idx].reshape(1, story_maxlen)
    else:
        for w in tokenize(user_story):
            if w.lower() not in word_idx:
                return flask.jsonify({"pred_answer": "Word " + w + " is not in vocabulary!"})
        inp = vectorize_story(user_story).reshape(1, story_maxlen)
    if user_question == '':
        quest = queries_train[question_idx].reshape(1, query_maxlen)
    else:
        for w in tokenize(user_question):
            if w.lower() not in word_idx:
                return flask.jsonify({"pred_answer": "Word " + w + " is not in vocabulary!"})
        quest = vectorize_question(user_question).reshape(1, query_maxlen)

    pred_answer = " ".join(list(map(lambda x: voc[x], np.argmax(model.predict([inp, quest]), axis=2)[0]))).strip(" ")

    return flask.jsonify({"pred_answer": pred_answer})


if __name__ == "__main__":
    init()
    run()
