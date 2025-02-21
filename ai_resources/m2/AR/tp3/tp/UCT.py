# This is a very simple implementation of the uct Monte Carlo Tree Search
# algorithm in Python 3 (initially in Python 2.7).
# The function uct(rootstate, itermax, verbose = False) is towards the bottom
# of the code.
# It aims to have the clearest and simplest possible code, and for the sake of
# clarity, the code is orders of magnitude less efficient than it could be
# made, particularly by using a state.GetRandomMove() or
# state.DoRandomRollout() function.
#
# Example GameState classes for Nim, OXO and Othello are included to give some
# idea of how you can write your own GameState use uct in your 2-player
# game. Change the game to be played in the uct_play_game() function at the
# bottom of the code.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse
# (University of York, UK) September 2012.
#
# Adaptation from Python 2.7 to Python 3, and pep8 compliance:
# Valentin Emiya, Aix-Marseille University, 2019.
#
# Licence is granted to freely use and distribute for any sensible/legal
# purpose so long as this comment remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at
# www.mcts.ai

from math import *
import random
import matplotlib.pyplot as plt


class GameState:
    """
    A state of the game, i.e. the game board. These are the only functions
    which are absolutely necessary to implement uct in any 2-player complete
    information deterministic zero-sum game, although they can be enhanced
    and made quicker, for example by using a GetRandomMove() function to
    generate a random move during rollout.
    By convention the players are numbered 1 and 2.
    """

    def __init__(self):
        # At the root pretend the player just moved is player 2 - player 1
        # has the first move
        self.player_just_moved = 2

    def clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.player_just_moved = self.player_just_moved
        return st

    def do_move(self, move):
        """ update a state by carrying out the given move.
            Must update player_just_moved.
        """
        self.player_just_moved = 3 - self.player_just_moved

    def get_moves(self):
        """ Get all possible moves from this state.
        """

    def get_result(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """

    def __init__(self):
        # At the root pretend the player just moved is p2 - p1 has the first
        # move
        self.player_just_moved = 2
        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.player_just_moved = self.player_just_moved
        st.board = self.board[:]
        return st

    def do_move(self, move):
        """ update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert isinstance(move, int)
        assert 0 <= move <= 8 and not self.board[move]
        self.player_just_moved = 3 - self.player_just_moved
        self.board[move] = self.player_just_moved

    def get_moves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]

    def get_result(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                          (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if len(self.get_moves()) == 0:
            return 0.5  # draw
        assert False  # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9):
            s += ".XO"[self.board[i]]
            if i % 3 == 2:
                s += "\n"
        return s


class OthelloState:
    """
    A state of the game of Othello, i.e. the game board.

    The board is a 2D array where 0 = empty (.), 1 = player 1 (X),
    2 = player 2 (O).
    In Othello players alternately place pieces on a square board - each piece
    played has to sandwich opponent pieces between the piece played and
    pieces already on the board. Sandwiched pieces are flipped.
    This implementation modifies the rules to allow variable sized square
    boards and terminates the game as soon as the player about to move
    cannot make a move (whereas the standard game allows for a pass move).
    """

    def __init__(self, sz=8):
        # At the root pretend the player just moved is p2 - p1 has the first
        # move
        self.player_just_moved = 5
        self.board = []  # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        assert sz == int(sz) and sz % 2 == 0  # size must be integral and even
        for y in range(sz):
            self.board.append([0] * sz)
        self.board[sz // 2][sz // 2] = self.board[sz // 2 - 1][sz // 2 - 1] = 1
        self.board[sz // 2][sz // 2 - 1] = self.board[sz // 2 - 1][sz // 2] = 2

    def clone(self):
        """ Create a deep clone of this game state.
        """
        st = OthelloState()
        st.player_just_moved = self.player_just_moved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def do_move(self, move):
        """ update a state by carrying out the given move.
            Must update playerToMove.
        """
        (x, y) = (move[0], move[1])
        assert x == int(x) and y == int(y) and self.is_on_board(x, y) and \
               self.board[x][y] == 0
        m = self.get_all_sandwiched_counters(x, y)
        self.player_just_moved = 3 - self.player_just_moved
        self.board[x][y] = self.player_just_moved
        for (a, b) in m:
            self.board[a][b] = self.player_just_moved

    def get_moves(self):
        """ Get all possible moves from this state.
        """
        return [(x, y) for x in range(self.size) for y in range(self.size)
                if self.board[x][y] == 0
                and self.exists_sandwiched_counter(x, y)]

    def adjacent_to_enemy(self, x, y):
        """
        Speeds up get_moves by only considering squares which are adjacent to
        an enemy-occupied square.
        """
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1),
                         (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.is_on_board(x + dx, y + dy) \
                    and self.board[x + dx][y + dy] == self.player_just_moved:
                return True
        return False

    def adjacent_enemy_directions(self, x, y):
        """
        Speeds up get_moves by only considering squares which are adjacent to
        an enemy-occupied square.
        """
        es = []
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1),
                         (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.is_on_board(x + dx, y + dy) \
                    and self.board[x + dx][y + dy] == self.player_just_moved:
                es.append((dx, dy))
        return es

    def exists_sandwiched_counter(self, x, y):
        """
        Does there exist at least one counter which would be flipped if my
        counter was placed at (x,y)?
        """
        for (dx, dy) in self.adjacent_enemy_directions(x, y):
            if len(self.sandwiched_counters(x, y, dx, dy)) > 0:
                return True
        return False

    def get_all_sandwiched_counters(self, x, y):
        """
        Is (x,y) a possible move (i.e. opponent counters are sandwiched between
         (x,y) and my counter in some direction)?
        """
        sandwiched = []
        for (dx, dy) in self.adjacent_enemy_directions(x, y):
            sandwiched.extend(self.sandwiched_counters(x, y, dx, dy))
        return sandwiched

    def sandwiched_counters(self, x, y, dx, dy):
        """
        Return the coordinates of all opponent counters sandwiched between
         (x,y) and my counter.
        """
        x += dx
        y += dy
        sandwiched = []
        while self.is_on_board(x, y) \
                and self.board[x][y] == self.player_just_moved:
            sandwiched.append((x, y))
            x += dx
            y += dy
        if self.is_on_board(x, y) \
                and self.board[x][y] == 3 - self.player_just_moved:
            return sandwiched
        else:
            return []  # nothing sandwiched

    def is_on_board(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def get_result(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        jmcount = len([(x, y)
                       for x in range(self.size) for y in range(self.size)
                       if self.board[x][y] == playerjm])
        notjmcount = len([(x, y)
                          for x in range(self.size) for y in range(self.size)
                          if self.board[x][y] == 3 - playerjm])
        if jmcount > notjmcount:
            return 1.0
        elif notjmcount > jmcount:
            return 0.0
        else:
            return 0.5  # draw

    def __repr__(self):
        s = ""
        for y in range(self.size - 1, -1, -1):
            s += ' '
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s


class Node:
    """
    A node in the game tree. Note wins is always from the viewpoint of
    player_just_moved. Crashes if state not specified.
    """

    def __init__(self, move=None, parent=None, state=None):
        # the move that got us to this node - "None" for the root node
        self.move = move
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_moves()  # future child nodes
        # the only part of the state that the Node needs later
        self.player_just_moved = state.player_just_moved

    def uct_select_child(self):
        """
        Use the UCB1 formula to select a child node. Often a constant UCTK is
        applied so we have
        lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits
        to vary the amount of exploration versus exploitation.
        """
        s = sorted(self.childNodes,
                   key=lambda c:
                   c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untried_moves.remove(m)
        self.childNodes.append(n)
        return n

    def update(self, result):
        """
        update this node - one additional visit and result additional wins.
        result must be from the viewpoint of player_just_moved.
        """
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


def uct(root_state, iter_max, verbose=False):
    """
    Conduct a uct search for itermax iterations starting from rootstate.
    Return the best move from the rootstate.
    Assumes 2 alternating players (player 1 starts), with game results in the
    range [0.0, 1.0].
    """

    rootnode = Node(state=root_state)

    for i in range(iter_max):
        node = rootnode
        state = root_state.clone()

        # Select
        # node is fully expanded and non-terminal
        while node.untried_moves == [] and node.childNodes != []:
            node = node.uct_select_child()
            state.do_move(node.move)

        # Expand
        # if we can expand (i.e. state/node is non-terminal)
        if len(node.untried_moves) > 0:
            m = random.choice(node.untried_moves)
            state.do_move(m)
            node = node.add_child(m, state)  # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a
        # state.GetRandomMove() function
        while len(state.get_moves()) > 0:  # while state is non-terminal
            state.do_move(random.choice(state.get_moves()))

        # Backpropagate
        # backpropagate from the expanded node and work back to the root node
        while node is not None:
            # state is terminal. update node with result from POV of
            # node.player_just_moved
            node.update(state.get_result(node.player_just_moved))
            node = node.parentNode

    '''# Output some information about the tree - can be omitted
    if verbose:
        print(rootnode.tree_to_string(0))
    else: 
        print(rootnode.children_to_string())'''
    # return the move that was most visited
    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move


def uct_play_game(value1, value2):
    """
    Play a sample game between two uct players where each player gets a
    different number of uct iterations (= simulations = tree nodes).
    """
    state = OXOState()  # Exemple avec le jeu OXO
    while len(state.get_moves()) > 0:
        if state.player_just_moved == 1:
            m = uct(root_state=state, iter_max=value1, verbose=False)
        else:
            m = uct(root_state=state, iter_max=value2, verbose=False)
        state.do_move(m)

    if state.get_result(state.player_just_moved) == 1.0:
        return state.player_just_moved  # Victoire pour le joueur actuel
    elif state.get_result(state.player_just_moved) == 0.0:
        return 3 - state.player_just_moved  # Victoire pour l'adversaire
    else:
        return -1


if __name__ == "__main__":
    results = []
    player1 = []
    player2 = []
    nobody = []

    for k in range(1, 10):
        x, y, z = 0, 0, 0

        for i in range(10 * k, 100 * k, 10):
            for j in range(10 * k, 100 * k, 10):
                result = uct_play_game(i, j)
                if result == -1:
                    x += 1
                elif result == 1:
                    y += 1
                elif result == 2:
                    z += 1

        nobody.append(x)
        player1.append(y)
        player2.append(z)

    plt.plot(player1, c="red", label="Player 1")
    plt.plot(player2, c="blue", label="Player 2")
    plt.plot(nobody, c="green", label="Nobody")
    plt.legend()
    plt.savefig("test.png")
    plt.show()
