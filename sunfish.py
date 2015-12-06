#!/usr/bin/env pypy
# -*- coding: utf-8 -*-

from __future__ import print_function
import re
import sys
import os.path
sys.path.append('util')
from itertools import count
from collections import OrderedDict, namedtuple
import set_compiler
set_compiler.install()
import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
import chess
import time
# from minimax import *

# This is the max depth we want our minimax to search
DEPTH = 5

# Our board is represented as a 120 character string. The padding allows for
# fast detection of moves that don't stay within the board.
A1, H1, A8, H8 = 91, 98, 21, 28
initial = (
    '         \n'  #   0 -  9
    '         \n'  #  10 - 19
    ' rnbqkbnr\n'  #  20 - 29
    ' pppppppp\n'  #  30 - 39
    ' ........\n'  #  40 - 49
    ' ........\n'  #  50 - 59
    ' ........\n'  #  60 - 69
    ' ........\n'  #  70 - 79
    ' PPPPPPPP\n'  #  80 - 89
    ' RNBQKBNR\n'  #  90 - 99
    '         \n'  # 100 -109
    '          '   # 110 -119
)

# Integer mapping for converting board to numpy array
char_map = { '\n': -3, ' ': -2, '.': -1, 
            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
            'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11 }

int_map = { -3: '\n', -2: ' ', -1: '.',
            0: 'p', 1: 'n', 2: 'b', 3: 'r', 4: 'q', 5: 'k',
            6: 'P', 7: 'N', 8: 'B', 9: 'R', 10: 'Q', 11: 'K' }

###############################################################################
# Move and evaluation tables
###############################################################################

N, E, S, W = -10, 1, 10, -1
directions = {
    'P': (N, 2*N, N+W, N+E),
    'N': (2*N+E, N+2*E, S+2*E, 2*S+E, 2*S+W, S+2*W, N+2*W, 2*N+W),
    'B': (N+E, S+E, S+W, N+W),
    'R': (N, E, S, W),
    'Q': (N, E, S, W, N+E, S+E, S+W, N+W),
    'K': (N, E, S, W, N+E, S+E, S+W, N+W)
}

###############################################################################
# Chess logic
###############################################################################

class Position(namedtuple('Position', 'board score wc bc ep kp')):
    """ A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights
    bc -- the opponent castling rights
    ep - the en passant square
    kp - the king passant square
    """

    def gen_moves(self):
        # Call cython function to generate moves
        move_count, sources, dests = chess._gen_moves(self.numpyify(), 
                                                np.array(self.wc).astype(np.uint8), 
                                                np.array(self.bc).astype(np.uint8), 
                                                self.ep, 
                                                self.kp,
                                                self.score)
        return zip(sources, dests)[:move_count]

    def rotate(self):
        return Position(
            self.board[::-1].swapcase(), -self.score,
            self.bc, self.wc, 119-self.ep, 119-self.kp)

    def move(self, move):
        pos = chess._make_move(self.numpyify(), 
                                np.array(self.wc).astype(np.uint8), 
                                np.array(self.bc).astype(np.uint8), 
                                self.ep, 
                                self.kp,
                                self.score, 
                                np.array(move).astype(np.int32))
        return Position(self.stringify(pos['board']), pos['score'], tuple(map(lambda x: bool(x), pos['wc'])), tuple(map(lambda x: bool(x), pos['bc'])), pos['ep'], pos['kp'])

    def numpyify(self):
        # convert board to numpy array
        return np.array(map(lambda c: char_map[c], self.board)).astype(np.int32)

    def stringify(self, np_board):
        return ''.join(map(lambda c: int_map[c], np_board))


###############################################################################
# User interface
###############################################################################

# Python 2 compatability
if sys.version_info[0] == 2:
    input = raw_input

def parse(c):
    fil, rank = ord(c[0]) - ord('a'), int(c[1]) - 1
    return A1 + fil - 10*rank


def render(i):
    rank, fil = divmod(i - A1, 10)
    return chr(fil + ord('a')) + str(-rank + 1)

def print_pos(pos):
    print()
    uni_pieces = {'R':'♜', 'N':'♞', 'B':'♝', 'Q':'♛', 'K':'♚', 'P':'♟',
                  'r':'♖', 'n':'♘', 'b':'♗', 'q':'♕', 'k':'♔', 'p':'♙', '.':'·'}
    for i, row in enumerate(pos.board.strip().split('\n ')):
        print(' ', 8-i, ' '.join(uni_pieces.get(p, p) for p in row))
    print('    a b c d e f g h \n\n')

def print_numpy(np_array):
    output = ""
    for i in np_array:
        if i in int_map:
            output += int_map[i]
        else:
            output += str(i)
    print(output)
    #print('[%s]' % (' '.join('%02s' % int_map[i] for i in np_array)))

def main():
    pos = Position(initial, 0, (True, True), (True, True), 0, 0)
    while True:
        print_pos(pos)
        # We query the user until she enters a legal move.
        move = None
        while move not in pos.gen_moves():
            match = re.match('([a-h][1-8])'*2, input('Your move: '))
            if match:
                move = parse(match.group(1)), parse(match.group(2))
            else:
                # Inform the user when invalid input (e.g. "help") is entered
                print("Please enter a move like g8f6")
        pos = pos.move(move)

        # After our move we rotate the board and print it again.
        # This allows us to see the effect of our move.
        print_pos(pos)
        pos = pos.rotate()
        t0 = time.time()
        # Here is our first attempt at a minimax algorithm tree. 
        alpha = -1000000
        bestValue = float("-inf")
        bestAction = None
        for move in pos.gen_moves():
            print("Analyzing")
            new_pos = pos.move(move)
            new_pos = new_pos.rotate()
            # Basic Minimax funcion
            # new_value = chess._minimax_helper(new_pos.numpyify(), 
            #                                 np.array(new_pos.wc).astype(np.uint8), 
            #                                 np.array(new_pos.bc).astype(np.uint8), 
            #                                 new_pos.ep, 
            #                                 new_pos.kp,
            #                                 new_pos.score, 
            #                                 1, 
            #                                 DEPTH)
            # Alpha Beta pruning
            # new_value = chess._alphabeta_helper(new_pos.numpyify(), 
            #                                 np.array(new_pos.wc).astype(np.uint8), 
            #                                 np.array(new_pos.bc).astype(np.uint8), 
            #                                 new_pos.ep, 
            #                                 new_pos.kp,
            #                                 new_pos.score, 
            #                                 1, 
            #                                 DEPTH,
            #                                 alpha,
            #                                 1000000)
            # PVSplit
            new_value = chess._pvsplit_helper(new_pos.numpyify(), 
                                            np.array(new_pos.wc).astype(np.uint8), 
                                            np.array(new_pos.bc).astype(np.uint8), 
                                            new_pos.ep, 
                                            new_pos.kp,
                                            new_pos.score, 
                                            1, 
                                            DEPTH,
                                            alpha,
                                            1000000)
            print(new_value)
            if new_value > bestValue or not bestAction:
                bestAction = move
                bestValue = new_value
                alpha = max(alpha, new_value)
        move = bestAction
        print("Number of Nodes Explored " + str(chess.printEval()))
        t1 = time.time() - t0
        print("Time to move")
        print(str(t1))
        # The black player moves from a rotated position, so we have to
        # 'back rotate' the move before printing it.
        print("My move:", render(119-move[0]) + render(119-move[1]))
        pos = pos.move(move)
        pos = pos.rotate()

if __name__ == '__main__':
    main()
