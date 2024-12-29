from math import *
import random
import matplotlib.pyplot as plt


class GameState:
    def __init__(self):
        self.player_just_moved = 2

    def clone(self):
        st = GameState()
        st.player_just_moved = self.player_just_moved
        return st

    def do_move(self, move):
        self.player_just_moved = 3 - self.player_just_moved

    def get_moves(self):
        pass

    def get_result(self, playerjm):
        pass

    def __repr__(self):
        pass


class OXOState:
    def __init__(self):
        self.player_just_moved = 2
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def clone(self):
        st = OXOState()
        st.player_just_moved = self.player_just_moved
        st.board = self.board[:]
        return st

    def do_move(self, move):
        assert isinstance(move, int)
        assert 0 <= move <= 8 and not self.board[move]
        self.player_just_moved = 3 - self.player_just_moved
        self.board[move] = self.player_just_moved

    def get_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def get_result(self, playerjm):
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                          (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if len(self.get_moves()) == 0:
            return 0.5
        assert False

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2:
                s += "\n"
        return s


class Node:
    def __init__(self, move=None, parent=None, state=None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_moves()
        self.player_just_moved = state.player_just_moved

    def uct_select_child(self):
        s = sorted(self.childNodes,
                   key=lambda c:
                   c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, m, s):
        n = Node(move=m, parent=self, state=s)
        self.untried_moves.remove(m)
        self.childNodes.append(n)
        return n

    def update(self, result):
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" \
            + str(self.visits) + " U:" + str(self.untried_moves) + "]"

    def tree_to_string(self, indent):
        s = _indent_string(indent) + str(self)
        for c in self.childNodes:
            s += c.tree_to_string(indent + 1)
        return s

    def children_to_string(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def _indent_string(indent):
    s = "\n"
    for i in range(1, indent + 1):
        s += "| "
    return s


def grandfather_heuristic(node, state):
    return sorted(node.untried_moves, key=lambda m: some_grandfather_priority(state, m))


def weighted_move_heuristic(state):
    weighted_moves = [(move, weight_of_move(state, move)) for move in state.get_moves()]
    return sorted(weighted_moves, key=lambda x: x[1], reverse=True)


def rave_heuristic(node, result):
    node.update(result)
    for child in node.childNodes:
        if similar_to_parent(child, node):
            child.update(result)


def uct(root_state, iter_max, verbose=False):
    rootnode = Node(state=root_state)

    for i in range(iter_max):
        node = rootnode
        state = root_state.clone()

        while node.untried_moves == [] and node.childNodes != []:
            node = node.uct_select_child()
            state.do_move(node.move)

        if len(node.untried_moves) > 0:
            m = random.choice(node.untried_moves)
            state.do_move(m)
            node = node.add_child(m, state)

        while len(state.get_moves()) > 0:
            state.do_move(random.choice(state.get_moves()))

        while node is not None:
            node.update(state.get_result(node.player_just_moved))
            node = node.parentNode

    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move


def uct_full_boosted(root_state, iter_max):
    rootnode = Node(state=root_state)

    for i in range(iter_max):
        node = rootnode
        state = root_state.clone()

        while node.untried_moves == [] and node.childNodes != []:
            node = node.uct_select_child()
            state.do_move(node.move)

        if len(node.untried_moves) > 0:
            moves = grandfather_heuristic(node, state)
            m = moves[0]
            state.do_move(m)
            node = node.add_child(m, state)

        while len(state.get_moves()) > 0:
            moves = weighted_move_heuristic(state)
            state.do_move(moves[0][0])

        result = state.get_result(node.player_just_moved)
        while node is not None:
            rave_heuristic(node, result)
            node = node.parentNode

    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move


def uct_play_game(value1, value2):
    state = OXOState()
    while len(state.get_moves()) > 0:
        if state.player_just_moved == 1:
            m = uct(root_state=state, iter_max=value1, verbose=False)
        else:
            m = uct(root_state=state, iter_max=value2, verbose=False)
        state.do_move(m)

    if state.get_result(state.player_just_moved) == 1.0:
        return state.player_just_moved
    elif state.get_result(state.player_just_moved) == 0.0:
        return 3 - state.player_just_moved
    else:
        return -1


def compare_uct_versions(root_state, iter_max):
    print("UCT Classique:")
    move_classic = uct(root_state, iter_max)

    print("UCT Full Boosted:")
    move_boosted = uct_full_boosted(root_state, iter_max)

    if move_classic == move_boosted:
        print("Les deux versions ont choisi le même mouvement.")
    else:
        print("Les mouvements choisis sont différents.")

