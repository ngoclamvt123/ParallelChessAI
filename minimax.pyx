#cython: boundscheck=False,
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from chess cimport *


cpdef int score_board(np.int32_t[:, :] board, np.int32_t[:] pos_values) nogil:
	cdef:
		int i, j, total

	total = 0
	for i in range(8):
		for j in range(8):
				total += pos_values[board[i][j]]
	return total

cpdef int minimax_helper(Position pos, int agentIndex, int depth) nogil:
	# Right now this is all within the GIL. The only way I can see this getting fixed
	# is if we rewrite all the methods as cython functions on numpy arrays
	if depth == 0:
		if agentIndex == 0:
			return evaluate(pos.board)
		else:
			return -1 * evaluate(pos.board)
	# Agent index 0 is the computer, trying to maximize the scoreboard
	if agentIndex == 0:
		bestValue = float("-inf")
		for move in gen_moves(pos.board):
			bestValue = max(bestValue, minimax_helper(rotate(make_move(pos.board)), 1, depth -1))

	# Agend index 1 is the human, trying to minimize the scoreboard
	elif agentIndex == 1:
		bestValue = float("inf")
		for move in gen_moves(pos.board):
			bestValue = min(bestValue, minimax_helper(rotate(make_move(pos.board)), 0, depth -1, pos_values))



	
