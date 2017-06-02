# coding: utf-8

import flask
import numpy as np
from data_utils import *
from model import create_model

app = flask.Flask(__name__)
model = None
train_stories = None
story_maxlen, question_maxlen = None, None
word_idx = None
voc = None


def run():
    global model, voc, train_stories, story_maxlen, question_maxlen, word_idx

    train_stories, test_stories, vocab_size, story_maxlen, question_maxlen, answer_maxlen, word_idx, voc = extract_data(
        test_data=False, lower=True)

    model = create_model(vocab_size, story_maxlen, question_maxlen, answer_maxlen)

    model.load_weights("weights.h5")
    app.run(host='0.0.0.0', port=6006)


@app.route('/')
def index():
    return flask.render_template("index.html")


@app.route('/get/story', methods=['GET'])
def get_story():
    question_idx = np.random.randint(len(train_stories))
    story_txt, question_txt, correct_answer = train_stories[question_idx]
    story_txt = ' '.join(story_txt).replace(' . ', '.\n').replace(' .', '.')
    question_txt = ' '.join(question_txt).replace(' ?', '?')
    correct_answer = ' '.join(correct_answer).lower()

    return flask.jsonify({"story": story_txt, "question": question_txt, "correct_answer": correct_answer})


@app.route('/get/answer', methods=['GET'])
def get_answer():
    user_question = flask.request.args.get('question')
    user_story = flask.request.args.get('story')

    for w in tokenize(user_story):
        if w.lower() not in word_idx:
            return flask.jsonify({"pred_answer": "Word " + w + " is not in vocabulary!"})

    inp = vectorize_story(user_story, word_idx, story_maxlen).reshape(1, story_maxlen)

    for w in tokenize(user_question):
        if w.lower() not in word_idx:
            return flask.jsonify({"pred_answer": "Word " + w + " is not in vocabulary!"})

    quest = vectorize_question(user_question, word_idx, question_maxlen).reshape(1, question_maxlen)
    pred_answer = " ".join(list(map(lambda x: voc[x], np.argmax(model.predict([inp, quest]), axis=2)[0]))).strip(" ")

    return flask.jsonify({"pred_answer": pred_answer})


if __name__ == "__main__":
    run()
