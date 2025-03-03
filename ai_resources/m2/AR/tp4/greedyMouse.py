# coding:utf-8

import random
from tqdm import tqdm
import setup
from qlearn import QLearn
from expectedSarsa import ExpectedSarsa
from sarsa import Sarsa
import config as cfg
from queue import Queue
import matplotlib.pyplot as plt


def pick_random_location():
    while 1:
        x = random.randrange(world.width)
        y = random.randrange(world.height)
        cell = world.get_cell(x, y)
        if not (cell.wall or len(cell.agents) > 0):
            return cell


class Cat(setup.Agent):
    def __init__(self, filename):
        self.cell = None
        self.catWin = 0
        self.color = cfg.cat_color
        with open(filename) as f:
            lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        self.fh = len(lines)
        self.fw = max([len(x) for x in lines])
        self.grid_list = [[1 for x in range(self.fw)] for y in range(self.fh)]
        self.move = [(0, -1), (1, -1), (
                1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

        for y in range(self.fh):
            line = lines[y]
            for x in range(min(self.fw, len(line))):
                t = 1 if (line[x] == 'X') else 0
                self.grid_list[y][x] = t


    # using BFS algorithm to move quickly to target.
    def bfs_move(self, target):
        if self.cell == target:
            return

        for n in self.cell.neighbors:
            if n == target:
                self.cell = target  # if next move can go towards target
                return

        best_move = None
        q = Queue()
        start = (self.cell.y, self.cell.x)
        end = (target.y, target.x)
        q.put(start)
        step = 1
        V = {}
        preV = {}
        V[(start[0], start[1])] = 1

        while not q.empty():
            grid = q.get()

            for i in range(8):
                ny, nx = grid[0] + self.move[i][0], grid[1] + self.move[i][1]
                if nx < 0 or ny < 0 or nx > (self.fw-1) or ny > (self.fh-1):
                    continue
                if self.get_value(V, (ny, nx)) or self.grid_list[ny][nx] == 1:  # has visit or is wall.
                    continue

                preV[(ny, nx)] = self.get_value(V, (grid[0], grid[1]))
                if ny == end[0] and nx == end[1]:
                    V[(ny, nx)] = step + 1
                    seq = []
                    last = V[(ny, nx)]
                    while last > 1:
                        k = [key for key in V if V[key] == last]
                        seq.append(k[0])
                        assert len(k) == 1
                        last = preV[(k[0][0], k[0][1])]
                    seq.reverse()

                    best_move = world.grid[seq[0][0]][seq[0][1]]

                q.put((ny, nx))
                step += 1
                V[(ny, nx)] = step

        if best_move is not None:
            self.cell = best_move

        else:
            dir = random.randrange(cfg.directions)
            self.go_direction(dir)

    def get_value(self, mdict, key):
        try:
            return mdict[key]
        except KeyError:
            return 0

    def update(self):
        if self.cell != mouse.cell:
            self.bfs_move(mouse.cell)


class Cheese(setup.Agent):
    def __init__(self):
        self.color = cfg.cheese_color

    def update(self):
        pass


class Mouse(setup.Agent):
    def __init__(self, ai):
        self.ai = None
        self.ai = ai
        self.catWin = 0
        self.mouseWin = 0
        self.lastState = None
        self.lastAction = None
        self.color = cfg.mouse_color


    def update(self):
        state = self.calculate_state()
        reward = cfg.MOVE_REWARD

        if self.cell == cat.cell:
            self.catWin += 1
            reward = cfg.EATEN_BY_CAT

            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, state, reward)



            self.lastState = None
            self.cell = pick_random_location()
            return

        if self.cell == cheese.cell:
            self.mouseWin += 1
            reward = cfg.EAT_CHEESE
            cheese.cell = pick_random_location()

        action = self.ai.choose_action(state)

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, state, reward)


        self.lastState = state
        self.lastAction = action
        self.go_direction(action)

    def calculate_state(self):
        def cell_value(cell):
            if cat.cell is not None and (cell.x == cat.cell.x and cell.y == cat.cell.y):
                return 3
            elif cheese.cell is not None and (cell.x == cheese.cell.x and cell.y == cheese.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return tuple([cell_value(world.get_relative_cell(self.cell.x + dir[0], self.cell.y + dir[1])) for dir in dirs])


class MouseSarsa(setup.Agent):
    def __init__(self, ai):
        self.ai = None
        self.ai = ai
        self.catWin = 0
        self.mouseWin = 0
        self.lastState = None
        self.lastAction = None
        self.color = cfg.mouse_color


    def update(self):
        state = self.calculate_state()
        reward = cfg.MOVE_REWARD

        if self.cell == cat.cell:
            self.catWin += 1
            reward = cfg.EATEN_BY_CAT

            if self.lastState is not None and self.lastAction is not None:
                self.ai.learn(self.lastState, self.lastAction, state, None, reward)

            self.lastState = None
            self.lastAction = None
            self.cell = pick_random_location()
            return

        if self.cell == cheese.cell:
            self.mouseWin += 1
            reward = cfg.EAT_CHEESE
            cheese.cell = pick_random_location()

        action = self.ai.choose_action(state)

        if self.lastState is not None and self.lastAction is not None:
            self.ai.learn(self.lastState, self.lastAction, state, action, reward)

        self.lastState = state
        self.lastAction = action

        self.go_direction(action)


    def calculate_state(self):
        def cell_value(cell):
            if cat.cell is not None and (cell.x == cat.cell.x and cell.y == cat.cell.y):
                return 3
            elif cheese.cell is not None and (cell.x == cheese.cell.x and cell.y == cheese.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return tuple([cell_value(world.get_relative_cell(self.cell.x + dir[0], self.cell.y + dir[1])) for dir in dirs])


class MouseExpectedSarsa(setup.Agent):
    def __init__(self, ai):
        self.ai = None
        self.ai = ai
        self.catWin = 0
        self.mouseWin = 0
        self.lastState = None
        self.lastAction = None
        self.color = cfg.mouse_color


    def update(self):
        state = self.calculate_state()
        reward = cfg.MOVE_REWARD

        if self.cell == cat.cell:
            self.catWin += 1
            reward = cfg.EATEN_BY_CAT

            if self.lastState is not None and self.lastAction is not None:
                self.ai.learn(self.lastState, self.lastAction, state, reward)

            self.lastState = None
            self.lastAction = None
            self.cell = pick_random_location()
            return

        if self.cell == cheese.cell:
            self.mouseWin += 1
            reward = cfg.EAT_CHEESE
            cheese.cell = pick_random_location()

        action = self.ai.choose_action(state)

        if self.lastState is not None and self.lastAction is not None:
            self.ai.learn(self.lastState, self.lastAction, state, reward)

        self.lastState = state
        self.lastAction = action

        self.go_direction(action)


    def calculate_state(self):
        def cell_value(cell):
            if cat.cell is not None and (cell.x == cat.cell.x and cell.y == cat.cell.y):
                return 3
            elif cheese.cell is not None and (cell.x == cheese.cell.x and cell.y == cheese.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        return tuple([cell_value(world.get_relative_cell(self.cell.x + dir[0], self.cell.y + dir[1])) for dir in dirs])


if __name__ == '__main__':
    qlearn = QLearn(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)
    sarsa = Sarsa(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)
    expected_sarsa = ExpectedSarsa(actions=range(cfg.directions), alpha=0.1, gamma=0.9, epsilon=0.1)

    mouse = Mouse(qlearn)
    cat = Cat(filename='resources/world.txt')
    cheese = Cheese()
    world = setup.World(filename='resources/world.txt')
    world.add_agent(mouse)
    world.add_agent(cheese, cell=pick_random_location())
    world.add_agent(cat, cell=pick_random_location())
    world.display.speed = cfg.speed

    wins = []

    for _ in tqdm(range(500_000)):
        world.update(mouse.mouseWin, mouse.catWin)
        wins.append(mouse.mouseWin/(mouse.catWin + mouse.mouseWin +1))

    plt.plot(range(len(wins)), wins, c="red", label="QLeaning")

    mouse = MouseSarsa(sarsa)
    cat = Cat(filename='resources/world.txt')
    cheese = Cheese()
    world = setup.World(filename='resources/world.txt')
    world.add_agent(mouse)
    world.add_agent(cheese, cell=pick_random_location())
    world.add_agent(cat, cell=pick_random_location())
    world.display.speed = cfg.speed

    wins = []

    for _ in tqdm(range(500_000)):
        world.update(mouse.mouseWin, mouse.catWin)
        wins.append(mouse.mouseWin / (mouse.catWin + mouse.mouseWin + 1))

    plt.plot(range(len(wins)), wins, c="blue", label="Sarsa")

    mouse = MouseExpectedSarsa(expected_sarsa)
    cat = Cat(filename='resources/world.txt')
    cheese = Cheese()
    world = setup.World(filename='resources/world.txt')
    world.add_agent(mouse)
    world.add_agent(cheese, cell=pick_random_location())
    world.add_agent(cat, cell=pick_random_location())
    world.display.speed = cfg.speed

    wins = []

    for _ in tqdm(range(500_000)):
        world.update(mouse.mouseWin, mouse.catWin)
        wins.append(mouse.mouseWin / (mouse.catWin + mouse.mouseWin + 1))

    plt.plot(range(len(wins)), wins, c="orange", label="ExpectedSarsa")

    plt.legend()
    plt.savefig("result2.png")
    plt.show()
