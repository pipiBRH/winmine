

import pygame
import sys
import minesweeper
import dataset
import os.path
import tensorflow as tf
import numpy as np
import argparse
import math
from enum import Enum


pygame.init()


WAIT_BETWEEN_CYCLE = 200
WAIT_N_CYCLE = 5

BOARD_WIDTH = 24
BOARD_HEIGHT = 24
BOARD_MINES = 99


class SolvingResult(Enum):
    CORRECT = 1
    INCORRECT = 2


class LoadResource(object):
    def __init__(self):
        super().__init__()

        SOURCE_IMAGE_WIDTH = 128
        self.width = int(SOURCE_IMAGE_WIDTH / 4)
        __ = self.width

        # load & rescale texture
        self.img = pygame.transform.scale(
            pygame.image.load('./texture/texture.jpg'),
            (4 * __, 3 * __))

        # define texture mapping in PyGame format
        self.localmap = {
            #     (     x,      y,  w,  w)
            '?': (0 * __, 0 * __, __, __),
            'F': (1 * __, 0 * __, __, __),
            'x': (2 * __, 0 * __, __, __),

            '-': (3 * __, 0 * __, __, __),

            '1': (0 * __, 1 * __, __, __),
            '2': (1 * __, 1 * __, __, __),
            '3': (2 * __, 1 * __, __, __),
            '4': (3 * __, 1 * __, __, __),

            '5': (0 * __, 2 * __, __, __),
            '6': (1 * __, 2 * __, __, __),
            '7': (2 * __, 2 * __, __, __),
            '8': (3 * __, 2 * __, __, __)
        }

        self.font = pygame.font.Font(None, 30)


RSC = LoadResource()
screen = pygame.display.set_mode((BOARD_WIDTH * RSC.width, BOARD_HEIGHT * RSC.width))


def markingArea(center, width, color):

    (x, y) = center

    x *= RSC.width
    y *= RSC.width
    __ = int(width / 2)

    x -= __ * RSC.width
    y -= __ * RSC.width

    w = width * RSC.width

    pygame.draw.rect(screen, color, (x, y, w, w), 3)


def crossArea(center, width, color, style=1):

    (x, y) = center
    x *= RSC.width
    y *= RSC.width
    __ = int(width / 2)

    x -= __ * RSC.width
    y -= __ * RSC.width

    w = width * RSC.width

    pygame.draw.line(screen, color, (x, y), (x + w, y + w), 5)
    pygame.draw.line(screen, color, (x + w, y), (x, y + w), 5)


def drawBoard(area):
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            screen.blit(
                RSC.img,
                dest=(x * RSC.width, y * RSC.width),
                area=RSC.localmap[area[y][x]])


def drawView(local, Q, color):
    markingArea(local, 5, color)
    crossArea(local, 1, color)
    text = RSC.font.render('{:.2f}'.format(Q), 2, (0, 0, 0))

    (x, y) = local
    y = (y - 1) * RSC.width
    x = x * RSC.width

    screen.blit(text, (x, y))


def main(TARGET_MODEL, MODEL_DEFINE):

    TestingModel = __import__(MODEL_DEFINE[:-3])
    TARGET_MODEL = os.path.abspath(TARGET_MODEL)
    TARGET_DIR = os.path.dirname(TARGET_MODEL) + '/'
    TARGET_CHECKPOINT_NAME = TARGET_MODEL[:-5]

    print('TARGET MODEL           : {}'.format(TARGET_MODEL))
    print('TARGET_DIR             : {}'.format(TARGET_DIR))
    print('TARGET_CHECKPOINT_NAME : {}'.format(TARGET_CHECKPOINT_NAME))

    targetModel = tf.train.import_meta_graph(TARGET_MODEL)
    board = minesweeper.MineSweeperBord(BOARD_WIDTH, BOARD_HEIGHT, BOARD_MINES)
    FLAG_RESTART = WAIT_N_CYCLE

    with tf.Session() as sess:

        targetModel.restore(sess, TARGET_CHECKPOINT_NAME)
        graph = tf.get_default_graph()

        HYPE = graph.get_tensor_by_name('LAST/HYPE:0')
        X0 = graph.get_tensor_by_name('INPUT:0')
        predictor = tf.argmax(HYPE, 1)

        counter = list()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            # restart board & counter
            if FLAG_RESTART >= WAIT_N_CYCLE:
                FLAG_RESTART = 0

                board.restart()
                counter = list()

                while True:
                    if board.touchReview((int(BOARD_WIDTH / 2), int(BOARD_HEIGHT / 2))):
                        break

                    board.restart()

            drawBoard(board.game(True))

            # collecting candiate
            candiate = list()
            for local in board.getDarkLocal():

                item = list()
                item.append(local)

                (area, mask) = board.getArea(local)
                item.append(dataset.visibility((area, mask)))
                item.append(area)
                item.append(mask)

                candiate.append(item)

            # if having any candite
            if len(candiate) > 0:

                # select most likely solvable candiate (sorting by Q(visibility))
                candiate = sorted(candiate, key=lambda x: x[1], reverse=True)[:1]

                (local, Q, area, mask) = candiate[0]

                # encode area for model to solving
                T = np.copy(area)
                T[mask == minesweeper.__TOKEN_MASK__] = minesweeper.__TOKEN_MASK__
                T[mask == minesweeper.__TOKEN_MARK__] = minesweeper.__TOKEN_MARK__
                candiate[0].append(TestingModel.encodeArea(T))

                # solving candiate
                prediction = sess.run(predictor, {X0: list(map(lambda x: x[4], candiate))})
                rawValue = sess.run(HYPE,      {X0: list(map(lambda x: x[4], candiate))})

                prediction = prediction[0]
                (local, Q) = candiate[0][:2]
                (Y, N) = rawValue[0]

                print('solving #{:4d} ,Y = {:.3f}, N = {:.3f}'.format(len(counter) + 1, Y, N))
                if prediction == 0:
                    drawView(local, Q, (255, 0, 0))
                elif prediction == 1:
                    drawView(local, Q, (0, 255, 0))

                # check prediction is correct
                if prediction == 0 and board.peeking(local) != minesweeper.__TOKEN_MINE__:
                    print('failse positive. Q = {:.2f}'.format(Q))
                    counter.append(SolvingResult.INCORRECT)
                elif prediction == 1 and board.peeking(local) == minesweeper.__TOKEN_MINE__:
                    print('failse negative. Q = {:.2f}'.format(Q))
                    counter.append(SolvingResult.INCORRECT)
                else:
                    counter.append(SolvingResult.CORRECT)

                # progress in game
                if board.peeking(local) == minesweeper.__TOKEN_MINE__:
                    board.markMine(local)
                else:
                    board.touchReview(local)

            else:
                if FLAG_RESTART == 0:
                    AC_RATE = float(counter.count(SolvingResult.CORRECT)) / len(counter)
                    print('     AC RATE = {:.2f}%'.format(100.0 * AC_RATE))
                    # print('SOLVING RATE = {:.8f}%'.format(100.0 * math.pow(0.5, counter.count(SolvingResult.INCORRECT))))

                FLAG_RESTART += 1

            # process all drawing & wait
            print()
            pygame.display.flip()
            pygame.time.wait(WAIT_BETWEEN_CYCLE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', required=True, help='model save')
    parser.add_argument('--model', '-m', required=True, help='model define file (.py)')
    args = parser.parse_args()

    main(args.target, args.model)
