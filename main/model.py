# coding: utf-8

from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Dropout, Masking
from keras.layers import concatenate
from keras.layers import GRU, BatchNormalization, RepeatVector
from keras.layers.wrappers import Bidirectional
from keras import backend as K

def my_metric(true, pred):
    return K.mean(K.equal(K.mean(K.equal(K.argmax(true, -1), K.argmax(pred, -1)), -1), 1))

def create_model(vocab_size, story_maxlen, query_maxlen, answer_maxlen, embs=None):
    input_sequence = Input((story_maxlen,))
    question = Input((query_maxlen,))

    hid_dim = 50

    context_rec = Masking()(input_sequence)
    if embs is not None:
        context_rec = Embedding(input_dim=vocab_size, output_dim=100, weights=[embs])(context_rec)
    else:
        context_rec = Embedding(input_dim=vocab_size, output_dim=100, weights=[embs])(context_rec)
    context_rec = BatchNormalization()(context_rec)
    context_rec = Bidirectional(GRU(hid_dim, return_sequences=True))(context_rec)
    context_rec = BatchNormalization()(context_rec)
    context_rec = Dropout(0.15)(context_rec)

    question_rec = Masking()(question)
    if embs is not None:
        question_rec = Embedding(input_dim=vocab_size, output_dim=100, weights=[embs])(question_rec)
    else:
        question_rec = Embedding(input_dim=vocab_size, output_dim=100)(question_rec)
    question_rec = BatchNormalization()(question_rec)
    question_rec = Bidirectional(GRU(hid_dim))(question_rec)
    question_rec = BatchNormalization()(question_rec)
    question_rec = RepeatVector(story_maxlen)(question_rec)
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', my_metric])
    return model
