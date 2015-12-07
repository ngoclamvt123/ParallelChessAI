#cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t, uint8_t
from libc.stdlib cimport abs

from cython.parallel import parallel, prange

import pyximport
pyximport.install()

from openmp cimport omp_lock_t, \
     omp_init_lock, omp_destroy_lock, \
     omp_set_lock, omp_unset_lock, omp_get_thread_num

from constants cimport *
from chess cimport Position, init_position, gen_moves, make_move, evaluate, rotate

cpdef int _alpha_beta_serial(np.int32_t[:] board,
									np.uint8_t[:] wc,
									np.uint8_t[:] bc,
									np.int32_t ep,
									np.int32_t kp,
									np.int32_t score,
									int agentIndex,
									int depth,
									int alpha,
									int beta):
	cdef:
		Position pos = init_position(board, wc, bc, ep, kp, score)

	return alpha_beta_serial(pos, agentIndex, depth, alpha, beta)

cpdef int alpha_beta_serial(Position pos, int agentIndex, int depth, int alpha, int beta) nogil:
	cdef:
		int32_t sources[MAX_MOVES]
		int32_t dests[MAX_MOVES]
		int32_t move_count
		int i, ret, bestValue, v, num_moves, j

	if depth == 0:
		if agentIndex == 0:
			return evaluate(pos.board)
		else:
			return -1 * evaluate(pos.board)

	# Agent 0 is the computer, trying to maximize
	if agentIndex == 0:
		v = -100000
		move_count = gen_moves(pos, sources, dests)
		for i in range(move_count):
			new_pos = make_move(pos, sources[i], dests[i])
			rotate(&new_pos)
			v = max(
				v,
				alpha_beta_serial(new_pos, 1, depth - 1, alpha, beta)
			)
			# Prune the rest of the children, don't need to look
			if v > beta:
				return v
			alpha = max(alpha, v)
		return v

	# Agent 1 is the human, trying to minimize
	elif agentIndex == 1:
		v = 100000
		move_count = gen_moves(pos, sources, dests)
		for i in range(move_count):
			new_pos = make_move(pos, sources[i], dests[i])
			rotate(&new_pos)
			v = min(
				v,
				alpha_beta_serial(new_pos, 0, depth - 1, alpha, beta)
			)
			# Too negative for max to allow this
			if v < alpha:
				return v
			beta = min(beta, v)
		return v