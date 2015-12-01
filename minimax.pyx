#cython: boundscheck=False,
cimport numpy as np
import numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython

cpdef void testing(np.int32_t[:, :] board) nogil:
	with gil:
		print(np.array_str(board))
		print("ayy")
		print("HEYY")
		raw_input()

cpdef int score_board(np.int32_t[:, :] board, np.int32_t[:] pos_values) nogil:
	cdef:
		int i, j, total

	total = 0
	for i in range(8):
		for j in range(8):
				total += pos_values[board[i][j]]
	return total

cpdef int minimax_helper(curr_position, int agentIndex, int depth, np.int32_t[:] pos_values) nogil:
	# Right now this is all within the GIL. The only way I can see this getting fixed
	# is if we rewrite all the methods as cython functions on numpy arrays
	with gil:
		if depth == 0:
			return score_board(curr_position.numpyify(), pos_values)
		# Agent index 0 is the computer, trying to maximize the scoreboard
		if agentIndex == 0:
			bestValue = float("-inf")
			for move in curr_position.gen_moves():
				bestValue = max(bestValue, minimax_helper(curr_position.move(move).rotate(), 1, depth -1, pos_values))

		# Agend index 1 is the human, trying to minimize the scoreboard
		elif agentIndex == 1:
			bestValue = float("inf")
			for move in curr_position.gen_moves():
				bestValue = min(bestValue, minimax_helper(curr_position.move(move).rotate(), 0, depth -1, pos_values))



	
