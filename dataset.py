import minesweeper
import json
import numpy as np


class DataContainer(object):
    def __init__(self):
        super().__init__()

        self.Q = list()
        self.ans = list()
        self.mask = list()
        self.area = list()

    def save(self, file: str):
        with open(file, 'wt') as f:
            item = dict()
            item['Q'] = list()
            item['ans'] = list()
            item['area'] = list()
            item['mask'] = list()

            for (Q, ans, area, mask) in zip(self.Q, self.ans, self.area, self.mask):
                duplcate = False
                # check data is unique
                if Q in item['Q'] and ans in item['ans']:
                    for (uArea, uMask) in zip(item['area'], item['mask']):
                        if (area == uArea).all() and (mask == uMask).all():
                            duplcate = True
                            break

                if not duplcate:
                    item['Q'].append(Q)
                    item['ans'].append(ans)
                    item['area'].append(area)
                    item['mask'].append(mask)

            del item['Q']
            del item['ans']

            item['area'] = list(map(lambda x: x.tolist(), item['area']))
            item['mask'] = list(map(lambda x: x.tolist(), item['mask']))

            f.write(json.dumps(item))

    def load(self, file: str):
        with open(file, 'rt') as f:
            item = json.loads(f.read())

            self.Q = list()
            self.ans = list()
            self.mask = list()
            self.area = list()

            for (area, mask) in zip(item['area'], item['mask']):
                self.mask.append(np.array(mask, dtype=object))
                self.area.append(np.array(area, dtype=object))

            for (area, mask) in zip(self.area, self.mask):
                self.Q.append(visibility((area, mask)))
                self.ans.append(area[2, 2])

        return self

    # merge 2 dataset
    def merge(self, x):

        for (Q, ans, area, mask) in zip(x.Q, x.ans, x.area, x.mask):
            self.Q.append(Q)
            self.ans.append(ans)
            self.area.append(area)
            self.mask.append(mask)

        return self

    # rotate dataset
    def rot(self):
        for i in range(len(self.Q)):
            Q = self.Q[i]
            ans = self.ans[i]
            refArea = self.area[i]
            refMask = self.mask[i]

            for rotate in range(3):
                self.Q.append(Q)
                self.ans.append(ans)

                refArea = np.rot90(refArea)
                refMask = np.rot90(refMask)

                self.area.append(refArea)
                self.mask.append(refMask)

        return self


class DataGenerator(object):
    def __init__(self, Q=0.5, amount=10, pattern=None):

        def rotateData(mask, area):
            tmask, tarea = np.copy(mask), np.copy(area)

            for T in range(4):
                yield tmask, tarea

                tmask = np.rot90(tmask)
                tarea = np.rot90(tarea)

        self.pattern = pattern

        center = (2, 2)

        self.Q = list()
        self.area = list()
        self.mask = list()
        self.ans = list()

        while amount > len(self.Q):
            for (q, area, mask) in generator(Q, pattern):
                if len(self.Q) >= amount:
                    break

                for (tmask, tarea) in rotateData(mask, area):
                    self.Q.append(q)
                    self.area.append(tarea)
                    self.mask.append(tmask)
                    self.ans.append(area[center])


class SerialDataGen(DataContainer):
    def __init__(self):
        super().__init__()

    def gen(self, pattern=None, boardCount=1, catchINRound=3):

        if pattern is None:
            pattern = (24, 24, 99)

        self.pattern = pattern
        width, height, __ = self.pattern
        board = minesweeper.MineSweeperBord(*self.pattern)

        self.Q = list()
        self.ans = list()
        self.mask = list()
        self.area = list()

        for __ in range(boardCount):
            while True:
                board.restart()
                if board.touchReview((int(width / 2), int(height / 2))):
                    break

            while True:
                candiate = list()
                for local in board.getDarkLocal():

                    (area, mask) = board.getArea(local)

                    candiate.append([local,
                                     visibility((area, mask)),
                                     area,
                                     mask])

                candiate = sorted(candiate, key=lambda x: x[1], reverse=True)[:catchINRound]

                if len(candiate) > 0:
                    for (__, Q, area, mask) in candiate:
                        self.Q.append(Q)
                        self.ans.append(area[(2, 2)])
                        self.area.append(area)
                        self.mask.append(mask)

                    (local, __, __, __) = candiate[0]
                    if board.peeking(local) == minesweeper.__TOKEN_MINE__:
                        board.markMine(local)
                    else:
                        board.touchReview(local)

                else:
                    break

        return self

# return P|(0, 1) solveability of dataset


def visibility(dataset):
    V = [[2,  4,  6,  4, 2],
         [4,  9, 12,  9, 4],
         [6, 12,  0, 12, 6],
         [4,  9, 12,  9, 4],
         [2,  4,  6,  4, 2]]

    V = np.array(V, dtype=np.float)
    S = V.sum()

    (area, mask) = dataset

    V[mask == minesweeper.__TOKEN_MASK__] = 0
    V[area == minesweeper.__TOKEN_OUTOFBORD__] *= 0.4

    return V.sum() / S


def generator(Q=0.5, pattern=None):

    if pattern is None:
        pattern = (11, 11, 9)

    width, height, mines = pattern

    center = int(width / 2)
    center = (center, center)

    width, height, mines = pattern

    board = minesweeper.MineSweeperBord(width, height, mines)

    while True:
        if board.touchReview(center):
            break

        board.restart()

    for local in board.getDarkLocal():
        dataset = board.getArea(local)
        __ = visibility(dataset)

        if __ >= Q:
            (area, mask) = dataset
            yield (__, area, mask)


if __name__ == '__main__':
    a = SerialDataGen().gen()
    b = SerialDataGen().gen()

    a.merge(b)
    a.save('/tmp/dump')

    c = SerialDataGen().load('/tmp/dump')

    # c.rot()
    print(len(c.Q))
