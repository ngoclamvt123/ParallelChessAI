#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import sys
import os.path
sys.path.append('util')

import argparse
from itertools import count
from collections import OrderedDict, namedtuple

import set_compiler
set_compiler.install()
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)

from chess import print_eval, _make_move, _gen_moves
from minimax import _minimax_serial, _minimax_top_level_parallel
from alphabeta import _alpha_beta_serial, _alpha_beta_bottom_level_parallel, _alpha_beta_top_level_parallel
from pvsplit import _pvsplit
import time

# This is the max depth we want our minimax to search
DEPTH = 3

# This is the number of threads at which we run our strategy
NUM_THREADS = 1

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

# Character and integer mapping for converting board to numpy array
char_map = { '\n': -3, ' ': -2, '.': -1, 
            'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
            'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11 }

int_map = { -3: '\n', -2: ' ', -1: '.',
            0: 'p', 1: 'n', 2: 'b', 3: 'r', 4: 'q', 5: 'k',
            6: 'P', 7: 'N', 8: 'B', 9: 'R', 10: 'Q', 11: 'K' }

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
        move_count, sources, dests = _gen_moves(self.numpyify(), 
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
        pos = _make_move(self.numpyify(), 
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
# Techniques
###############################################################################

def minimax_serial(pos):
    (score, move) = _minimax_serial(pos.numpyify(), 
                        np.array(pos.wc).astype(np.uint8), 
                        np.array(pos.bc).astype(np.uint8), 
                        pos.ep, 
                        pos.kp,
                        pos.score, 
                        0, 
                        DEPTH)
    nodes = print_eval()
    return (score, move, nodes)

def minimax_top_level_parallel(pos):
    (score, move) = _minimax_top_level_parallel(pos.numpyify(), 
                        np.array(pos.wc).astype(np.uint8), 
                        np.array(pos.bc).astype(np.uint8), 
                        pos.ep, 
                        pos.kp,
                        pos.score, 
                        0, 
                        DEPTH,
                        NUM_THREADS)
    nodes = print_eval()
    return (score, move, nodes)

def alpha_beta_serial(pos):
    (score, move) = _alpha_beta_serial(pos.numpyify(), 
                        np.array(pos.wc).astype(np.uint8), 
                        np.array(pos.bc).astype(np.uint8), 
                        pos.ep, 
                        pos.kp,
                        pos.score, 
                        0, 
                        DEPTH,
                        -100000,
                        100000)
    nodes = print_eval()
    return (score, move, nodes)

def alpha_beta_bottom_level_parallel(pos):
    (score, move) = _alpha_beta_bottom_level_parallel(pos.numpyify(), 
                        np.array(pos.wc).astype(np.uint8), 
                        np.array(pos.bc).astype(np.uint8), 
                        pos.ep, 
                        pos.kp,
                        pos.score, 
                        0, 
                        DEPTH,
                        NUM_THREADS,
                        -100000,
                        100000)
    nodes = print_eval()
    return (score, move, nodes)

def alpha_beta_top_level_parallel(pos):
    (score, move) = _alpha_beta_top_level_parallel(pos.numpyify(), 
                        np.array(pos.wc).astype(np.uint8), 
                        np.array(pos.bc).astype(np.uint8), 
                        pos.ep, 
                        pos.kp,
                        pos.score, 
                        0, 
                        DEPTH,
                        NUM_THREADS,
                        -100000,
                        100000)
    nodes = print_eval()
    return (score, move, nodes)

def pvsplit(pos):
    (score, move) = _pvsplit(pos.numpyify(), 
                        np.array(pos.wc).astype(np.uint8), 
                        np.array(pos.bc).astype(np.uint8), 
                        pos.ep, 
                        pos.kp,
                        pos.score, 
                        0, 
                        DEPTH,
                        -1000000,
                        1000000,
                        NUM_THREADS)
    nodes = print_eval()
    return (score, move, nodes)

strategy_map = { 1: ("Minimax Serial", minimax_serial), 
                2: ("Minimax Top Level Parallel", minimax_top_level_parallel), 
                3: ("Alpha Beta Serial", alpha_beta_serial),
                4: ("Alpha Beta Bottom Level Parallel", alpha_beta_bottom_level_parallel),
                5: ("Alpha Beta Top Level Parallel", alpha_beta_top_level_parallel),
                6: ("PVSplit", pvsplit) }

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
    uni_pieces = {'R':'♖', 'N':'♘', 'B':'♗', 'Q':'♕', 'K':'♔', 'P':'♙',
                  'r':'♜', 'n':'♞', 'b':'♝', 'q':'♛', 'k':'♚', 'p':'♟', '.':'·'}
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


def main():
    parser = argparse.ArgumentParser(description="Choose the AI strategy")
    parser.add_argument("-s", "--strategy", type=int, choices=range(1,7), default=1,
                        help="Choose: 1 for Serial Minimax, 2 for Parallel Top Level Minimax, 3 for Serial Alpha Beta, 4 for Parallel Bottom Level Alpha Beta, 5 for Parallel Top Level Alpha Beta, 6 for PVSplit. By default, we use Serial Minimax.")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Choose the number of threads to use. By default, we use 1.")
    parser.add_argument("-d", "--depth", type=int, default=3,
                        help="Choose the depth at which to search the tree. By default, we use 3.")
    args = parser.parse_args()

    # Check if serial algorithm chosen
    if args.threads > 1 and args.strategy in (1, 3):
        print("Cannot run serial strategy with multiple threads")
        return

    # Check if depth valid
    if args.depth <= 0:
        print("Depth must be a positive integer")
        return

    # Check if number of threads valid
    if args.threads <= 0:
        print("Number of threads must be a positive integer")
        return

    global NUM_THREADS
    NUM_THREADS = args.threads

    global DEPTH
    DEPTH = args.depth

    name, func = strategy_map[args.strategy]

    print()
    print('Analyzing with ' + name + ' at depth ' + str(DEPTH) + ' with ' + str(NUM_THREADS) + ' thread(s)')

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
        
        print("Running " + name)
        print()

        t0 = time.time()
        (score, move, nodes) = func(pos)
        t1 = time.time() - t0

        print("Number of board states explored: " + str(nodes))
        print("Time to move: " + str(t1) + " seconds")
        
        # The black player moves from a rotated position, so we have to
        # 'back rotate' the move before printing it.
        print("My move:", render(119-move[0]) + render(119-move[1]))
        pos = pos.move(move)
        pos = pos.rotate()

if __name__ == '__main__':
    main()
