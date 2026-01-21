from collections import namedtuple
import numpy as np
import random


Point = namedtuple("Point", "x, y")

BLOCK_SIZE = 50


class Tile:
    def __init__(self, pos: Point, parent: Point):
        self.parent = parent
        self.pos = pos

    def __repr__(self):
        return f"{self.pos} {self.parent}"


def _true_check(a, b, get_value=False):
    for i in a:
        if i.pos == b.pos:
            if get_value:
                return i
            return True
    return False


def list_check(a):
    b = []
    for i in a:
        if not _true_check(b, i):
            b.append(i)

    return b


def _get_parent(lista, target):
    for i in lista:
        if i.pos == target.parent:
            return i


class PathFinder:
    def __init__(self, map: np.array, start: Point, end: Point, w=1900, h=900):
        self.h = h
        self.w = w
        self.map = map
        self.start = start
        self.end = end
        self.route = None
        self.available = []
        self.to_check = [Tile(self.start, self.start)]
        self.dir_route = []

        self.loop()

    def loop(self):
        while type(self.route) != list:
            self._path_finding()
        self._convert_route()

    def _convert_route(self):
        for i in self.route[::-1]:
            if i.parent.x < i.pos.x:
                self.dir_route.append(1)
            if i.parent.x > i.pos.x:
                self.dir_route.append(2)
            if i.parent.y < i.pos.y:
                self.dir_route.append(4)
            if i.parent.y > i.pos.y:
                self.dir_route.append(3)

    def _path_finding(self):
        available = []
        for i in self.to_check:
            neighbours = []
            for j in range(-1, 2):
                if j == 0:
                    for k in range(1, -2, -1):
                        if 0 <= i.pos.x - k < self.map.shape[0] and 0 <= i.pos.y < self.map.shape[1]:
                            neighbours.append(Tile(Point(i.pos.x - k, i.pos.y), i.pos))
                elif 0 <= i.pos.x < self.map.shape[0] and 0 <= i.pos.y - j < self.map.shape[1]:
                    neighbours.append(Tile(Point(i.pos.x, i.pos.y - j), i.pos))

            for i in neighbours:
                if not self.map[i.pos.x, i.pos.y] and not _true_check(self.available, i):
                    available.append(i)

        self.to_check = list_check(available)
        if _true_check(self.to_check, Tile(self.end, self.end)):
            end_tile = _true_check(self.to_check, Tile(self.end, self.end), get_value=True)
            route = [end_tile]
            while not _true_check(route, Tile(self.start, self.start)):
                the_list = self.to_check
                the_list.extend(self.available)
                route.append(_get_parent(the_list, route[-1]))
                self.route = route
            self.available = self.to_check
            self.to_check = []

        for i in available:
            if not _true_check(self.available, i):
                self.available.append(i)


class Bot:
    def __init__(self, map: np.array, randomness=0.4):
        self.randomness = randomness
        self.map = map
        self.target = None
        self.route = []
        self.step = 0
        self.random_move = 0

    def forward(self, pos: Point, target: Point):
        self.random_move -= 1

        if target == pos:
            action = random.randint(1, 4)
            return action
        else:
            if self.target != target or len(self.route) <= self.step or self.random_move == 0:
                self.target = target
                path_finder = PathFinder(self.map,
                                         Point(int(pos.x / BLOCK_SIZE), int(pos.y / BLOCK_SIZE)),
                                         Point(int(target.x / BLOCK_SIZE), int(target.y / BLOCK_SIZE)))
                self.route = path_finder.dir_route
                self.step = 0

            if np.random.choice([True, False], p=(self.randomness, 1 - self.randomness)):
                self.random_move = 1
                action = random.randint(1, 4)
            else:
                action = self.route[self.step]
                self.step += 1
            return action
