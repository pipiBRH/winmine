import random
import json
from enum import Enum
import numpy as np

__TOKEN_MINE__ = 'x'
__TOKEN_EMPTY__ = '-'
__TOKEN_MASK__ = '?'
__TOKEN_OUTOFBORD__ = '!'
__TOKEN_MARK__ = 'F'


class Dummy(object):
    def __init__(self):
        super().__init__()


class MineSweeperBord(object):
    def __init__(self, width, height, mines):
        super().__init__()
        self.width = width
        self.height = height
        self.mines = mines

        self.tostr = np.vectorize(str, otypes=[str])

        self.restart()

    def restart(self):
        def Layer():
            return np.ndarray(shape=(self.width, self.height),
                              dtype=object)

        self.layers = Dummy()
        self.layers.mines = Layer()
        self.layers.hints = Layer()
        self.layers.masks = Layer()
        self.layers.flags = Layer()

        self.layers.hints.fill(0)
        self.layers.masks.fill(__TOKEN_MASK__)

        def __(x):
            return random.randint(0, x - 1)

        for i in range(self.mines):
            while True:
                local = __(self.width), __(self.height)

                if self.layers.mines[local] is None:
                    self.layers.mines[local] = __TOKEN_MINE__

                    for p in self.area(3,
                                       local,
                                       ignoreCenter=True,
                                       inBoard=True):

                        self.layers.hints[p] += 1

                    break

        self.layers.hints[self.layers.mines is not None] = None
        self.layers.hints[self.layers.hints == 0] = None

    def area(self, size, local, ignoreCenter=False, inBoard=False):
        (cx, cy) = local

        __ = int(size / 2)

        for y in range(cy - __, cy + __ + 1):
            for x in range(cx - __, cx + __ + 1):
                p = (x, y)

                if p == local and ignoreCenter:
                    continue

                if inBoard and (not self.inBoard(p)):
                    continue
                yield p

    def inBoard(self, local):
        (x, y) = local

        if (x < 0) or (x >= self.width):
            return False

        if (y < 0) or (y >= self.height):
            return False

        return True

    def addLayers(self, layers):
        result = np.ndarray(
            shape=layers[0].shape,
            dtype=object)

        for ly in layers:
            mask = (~(ly is None)) & (result is None)
            result[mask] = ly[mask]
        print(result)
        return result

    def game(self, mask=False):
        q = list()

        if mask is True:
            q.append(self.layers.flags)
            q.append(self.layers.masks)

        q.append(self.layers.mines)
        q.append(self.layers.hints)

        __ = self.addLayers(q)
        __[__ is None] = __TOKEN_EMPTY__

        f = np.vectorize(str)

        return f(__).T.tolist()

    def touchReview(self, local):
        if not self.inBoard(local):
            return True

        if self.layers.masks[local] is None:
            return True
        else:
            self.layers.masks[local] = None

            if self.layers.mines[local] == __TOKEN_MINE__:
                return False

            if self.layers.hints[local] is None:

                for p in self.area(
                        3,
                        local,
                        ignoreCenter=True,
                        inBoard=True):

                    self.touchReview(p)

                return True

    def markMine(self, local):
        if self.inBoard(local):
            if not self.layers.masks[local] is None:
                self.layers.flags[local] = __TOKEN_MARK__
                self.layers.masks[local] = None

    def getDarkLocal(self):
        for y in range(self.height):
            for x in range(self.width):
                local = (x, y)
                if not self.layers.masks[local] is None:
                    yield local

    def getArea(self, local):
        (x, y) = local
        size = 5
        __ = int(size / 2)

        left, right = x - __, x + __ + 1
        top, bottom = y - __, y + __ + 1

        while left < 0:
            left += 1

        while right > self.width:
            right -= 1

        while top < 0:
            top += 1

        while bottom > self.height:
            bottom -= 1

        stack = list()
        stack.append(self.layers.mines[left: right, top: bottom])
        stack.append(self.layers.hints[left: right, top: bottom])
        area = self.addLayers(stack)

        area[area is None] = __TOKEN_EMPTY__
        area = self.tostr(area)

        stack = list()
        stack.append(self.layers.masks[left: right, top: bottom])
        stack.append(self.layers.flags[left: right, top: bottom])
        mask = self.addLayers(stack)

        # mask = (mask != None)

        if ((right - left) < size) or ((bottom - top) < size):

            xoffset = __ - x
            yoffset = __ - y
            left += xoffset
            right += xoffset
            top += yoffset
            bottom += yoffset

            pmask = np.ndarray(shape=(size, size), dtype=mask.dtype)
            pmask.fill(None)

            parea = np.ndarray(shape=(size, size), dtype=area.dtype)
            parea.fill(__TOKEN_OUTOFBORD__)

            pmask[left: right, top: bottom] = mask[:, :]
            parea[left: right, top: bottom] = area[:, :]

            return parea.T, pmask.T

        else:
            return area.T, mask.T

    def peeking(self, local):
        for l in [self.layers.mines, self.layers.hints]:
            if not l[local] is None:
                return str(l[local])

        return __TOKEN_EMPTY__


if __name__ == '__main__':
    board = MineSweeperBord(5, 5, 3)
    print(board.game())

    # for p in board.getDarkLocal():
    #     area, mask = board.getArea(p)
    #     print(p)
    #     print(area)
    #     print()
    #     print(mask)
    #     print()
