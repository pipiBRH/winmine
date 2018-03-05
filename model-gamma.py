from __future__ import absolute_import
from __future__ import division

import argparse
import os.path
import sys
import time

import tensorflow as tf
import numpy as np
import minesweeper
import dataset


MODEL_NAME = 'model-gamma'
LOG_DIR = os.path.abspath('./model/{}/{}/'.format(
    MODEL_NAME,
    time.strftime('run-%Y%m%d-%H%M%S')))


NODE_S_STATUS = 24 * 12
HIDE_LAYER_WIDTH = 28
LABELS_A = 2

TRAIN_STEP = 5000
CHECKPOINT_STEP = 100
REPORT_STEP = 5


def encodeArea(area):
    code = [
        '1', '2', '3', '4', '5', '6', '7', '8',
        minesweeper.__TOKEN_EMPTY__,
        minesweeper.__TOKEN_MARK__,
        minesweeper.__TOKEN_MASK__,
        minesweeper.__TOKEN_OUTOFBORD__
    ]
    __ = len(code)
    code = np.array(code, dtype=str)

    item = list()

    for y in range(5):
        for x in range(5):
            if x == y == 2:
                continue

            item.append((code == area[y, x]).astype(float))

    item = np.array(item).reshape(24 * __)
    item = item.tolist()

    return item


def encodeData(dataset):

    dataset.case = list()
    dataset.labels = list()

    for (ans, mask, area) in zip(dataset.ans, dataset.mask, dataset.area):
        dataset.labels.append([
            float(ans == minesweeper.__TOKEN_MINE__),
            float(ans != minesweeper.__TOKEN_MINE__)])

        area[mask == minesweeper.__TOKEN_MASK__] = minesweeper.__TOKEN_MASK__
        area[mask == minesweeper.__TOKEN_MARK__] = minesweeper.__TOKEN_MARK__

        dataset.case.append(encodeArea(area))

    return dataset


def mkModel(X, CONSTANT_BIAS=None):

    if CONSTANT_BIAS is None:
        CONSTANT_BIAS = float(-0.5)

    def createFC(shape, XN, finalName=None, scopeName=None):

        if scopeName is None:
            scopeName = 'FC'

        with tf.name_scope(scopeName):
            WN = tf.Variable(tf.truncated_normal(shape, stddev=0.35), dtype=tf.float32, name="W")
            CN = tf.Variable(tf.constant(value=CONSTANT_BIAS, dtype=tf.float32, name='C'))
            YN = tf.matmul(XN, WN) + CN
            SN = tf.sigmoid(YN, name=finalName)

            tf.summary.histogram('W', WN)
            tf.summary.histogram('Y', YN)
            tf.summary.histogram('C', CN)
            tf.summary.histogram('SIGMOID', SN)

            return SN

    HIDE_LAYER_1 = createFC((NODE_S_STATUS, HIDE_LAYER_WIDTH), X)
    HIDE_LAYER_2 = createFC((HIDE_LAYER_WIDTH, HIDE_LAYER_WIDTH), HIDE_LAYER_1)
    HIDE_LAYER_3 = createFC((HIDE_LAYER_WIDTH, HIDE_LAYER_WIDTH), HIDE_LAYER_2)
    # HIDE_LAYER_4 = createFC((HIDE_LAYER_WIDTH, HIDE_LAYER_WIDTH), HIDE_LAYER_3)
    # HIDE_LAYER_5 = createFC((HIDE_LAYER_WIDTH, HIDE_LAYER_WIDTH), HIDE_LAYER_4)
    # HIDE_LAYER_6 = createFC((HIDE_LAYER_WIDTH, HIDE_LAYER_WIDTH), HIDE_LAYER_5)
    MODEL = createFC((HIDE_LAYER_WIDTH, LABELS_A), HIDE_LAYER_3, 'HYPE', 'LAST')

    return MODEL


def main():

    X0 = tf.placeholder(tf.float32, shape=(None, NODE_S_STATUS), name='INPUT')
    HYPE = mkModel(X0)

    YANS = tf.placeholder(dtype=tf.float32, name='LABELS')
    QValue = tf.placeholder(dtype=tf.float32, name='QVALUE')

    with tf.name_scope('LOSS'):
        inter = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=YANS,
            logits=HYPE)

        LOSS = tf.reduce_mean(inter * QValue, name='LOSS_FUNC')

        tf.summary.scalar('LOSS', LOSS)

    with tf.name_scope('train'):
        # train = tf.train.GradientDescentOptimizer(0.05).minimize(LOSS)
        train = tf.train.AdamOptimizer(0.0003, name='TRAINER_FUNC').minimize(LOSS)
        tf.add_to_collection('TRAINER_GROUP', train)

    summ = tf.summary.merge_all()

    with tf.name_scope('validation'):
        correct_prediction = tf.equal(tf.argmax(HYPE, 1), tf.argmax(YANS, 1))
        accuracy_func = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        accuracyIN = tf.summary.scalar('train_dataset', accuracy_func)
        accuracyOUT = tf.summary.scalar('test_dataset', accuracy_func)

    print('loading dataset')

    validationSet = encodeData(dataset.SerialDataGen().load('./data/set-01.json'))

    trainSet = dataset.SerialDataGen()

    for file in range(2, 10 + 1):
        tmp = dataset.SerialDataGen().load('./data/set-{:02d}.json'.format(file))
        trainSet.merge(tmp)

    trainSet = encodeData(trainSet.rot())

    modelSaver = tf.train.Saver(max_to_keep=int(TRAIN_STEP / CHECKPOINT_STEP))
    writer = tf.summary.FileWriter(LOG_DIR)

    print('training')
    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        writer.add_graph(sess.graph)

        for i in range(TRAIN_STEP + 1):
            sess.run(train, {X0: trainSet.case, YANS: trainSet.labels, QValue: trainSet.Q})

            if i % REPORT_STEP == 0:
                other = sess.run(summ, {X0: trainSet.case, YANS: trainSet.labels, QValue: trainSet.Q})
                trainAccuracy = sess.run(accuracyIN, {X0: trainSet.case, YANS: trainSet.labels})
                testAccuracy = sess.run(accuracyOUT, {X0: validationSet.case, YANS: validationSet.labels})

                writer.add_summary(other, i)
                writer.add_summary(trainAccuracy, i)
                writer.add_summary(testAccuracy, i)

            if i % CHECKPOINT_STEP == 0:
                modelSaver.save(sess, os.path.join(LOG_DIR, MODEL_NAME), global_step=i)


if __name__ == '__main__':
    main()
