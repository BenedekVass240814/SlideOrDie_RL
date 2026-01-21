import numpy as np
from collections import namedtuple
import random

BLOCK_SIZE = 50

Point = namedtuple("Point", "x, y")


class MapGenerator:
    def __init__(self, w=1600, h=900):
        self.h = h
        self.w = w
        p = 0.30
        self.map = np.random.choice(a=[True, False],
                                    size=(int(self.w / BLOCK_SIZE), int(self.h / BLOCK_SIZE)),
                                    p=(p, 1 - p))
        self.map[13:20, 0] = True
        self.spawn_point = Point(int(self.map.shape[0] / 2),
                                 int(self.map.shape[1] / 2))
        self.available = set()
        self.to_check = [self.spawn_point]
        self.gen()
        self.map[self.spawn_point.x, self.spawn_point.y] = False
        self.map[self.enemy_spawn.x, self.enemy_spawn.y] = False

    def gen(self):
        for i in range(100):
            self._path_finding()
        self._fill_map()
        while True:
            try:
                self.enemy_spawn = random.choice(list(self.available))
                return 0
            except:
                pass

    def _path_finding(self):
        available = []
        for i in self.to_check:
            neighbours = set()
            for j in range(-1, 2):
                if j == 0:
                    for k in range(1, -2, -1):
                        if 0 <= i.x - k < self.map.shape[0] and 0 <= i.y < self.map.shape[1]:
                            neighbours.add(Point(i.x - k, i.y))
                elif 0 <= i.x < self.map.shape[0] and 0 <= i.y - j < self.map.shape[1]:
                    neighbours.add(Point(i.x, i.y - j))

            for i in neighbours:
                if not self.map[i.x, i.y] and i not in self.available:
                    available.append(Point(i.x, i.y))

        self.to_check = set(available)

        for i in available:
            self.available.add(i)

    def _fill_map(self):
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if Point(i, j) not in self.available:
                    self.map[i, j] = True
