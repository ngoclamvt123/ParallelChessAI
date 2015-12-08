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

cpdef _alpha_beta_serial(np.int32_t[:] board,
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
		int32_t move[2]
		Position pos = init_position(board, wc, bc, ep, kp, score)
		int bestValue = alpha_beta_serial(pos, agentIndex, depth, alpha, beta, move)

	return (bestValue, move)

cdef int alpha_beta_serial(Position pos, int agentIndex, int depth, int alpha, int beta, int32_t *move) nogil:
	cdef:
		int32_t sources[MAX_MOVES]
		int32_t dests[MAX_MOVES]
		int32_t move_count
		int i, ret, bestValue, v, j

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
				alpha_beta_serial(new_pos, 1, depth - 1, alpha, beta, move)
			)
			# Prune the rest of the children, don't need to look
			if v > beta:
				return v
			if v > alpha:
				alpha = v
				move[0] = sources[i]
				move[1] = dests[i]
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
				alpha_beta_serial(new_pos, 0, depth - 1, alpha, beta, move)
			)
			# Too negative for max to allow this
			if v < alpha:
				return v
			beta = min(beta, v)
		return v

cpdef _alpha_beta_bottom_level_parallel(np.int32_t[:] board,
									np.uint8_t[:] wc,
									np.uint8_t[:] bc,
									np.int32_t ep,
									np.int32_t kp,
									np.int32_t score,
									int agentIndex,
									int depth,
									int num_threads,
									int alpha,
									int beta):
	cdef:
		int32_t move[2]
		Position pos = init_position(board, wc, bc, ep, kp, score)
		int bestValue = alpha_beta_bottom_level_parallel(pos, agentIndex, depth, num_threads, alpha, beta, move)

	return (bestValue, move)

cdef int alpha_beta_bottom_level_parallel(Position pos, int agentIndex, int depth, int num_threads, int alpha, int beta, int32_t *move) nogil:
	cdef:
		int32_t sources[MAX_MOVES]
		int32_t dests[MAX_MOVES]
		int32_t move_count
		int min_val[1], max_val[1]
		int i, ret, bestValue, v, j
		omp_lock_t eval_lock
		Position new_pos

	if depth == 0:
		if agentIndex == 0:
			return evaluate(pos.board)
		else:
			return -1 * evaluate(pos.board)
	# An attempt at parallelization!
	elif depth == 1:
		# Assumes it is an even depth to start with
		# agentIndex 0 right now
		if agentIndex == 0:
			min_val[0] = -100000
			move_count = gen_moves(pos, sources, dests)
			omp_init_lock(&eval_lock)

			# Parallelize over the last level of evaluations
			for i in prange(move_count, num_threads=num_threads, nogil=True):
				new_pos = make_move(pos, sources[i], dests[i])
				rotate(&new_pos)
				j = evaluate(new_pos.board)
				j = -1 * j
				if j > beta:
					return j
				omp_set_lock(&eval_lock)
				min_val[0] = max(j, min_val[0])
				omp_unset_lock(&eval_lock)
			omp_destroy_lock(&eval_lock)
			return min_val[0]

		elif agentIndex == 1:
			max_val[0] = 100000
			move_count = gen_moves(pos, sources, dests)
			omp_init_lock(&eval_lock)

			# Parallelize over the last level of evaluations
			for i in prange(move_count, num_threads=num_threads, nogil=True):
				new_pos = make_move(pos, sources[i], dests[i])
				rotate(&new_pos)
				j = evaluate(new_pos.board)
				if j < alpha:
					return j
				omp_set_lock(&eval_lock)
				max_val[0] = min(j, max_val[0])
				omp_unset_lock(&eval_lock)
			omp_destroy_lock(&eval_lock)
			return max_val[0]

	# Agent 0 is the computer, trying to maximize
	if agentIndex == 0:
		v = -100000
		move_count = gen_moves(pos, sources, dests)
		for i in range(move_count):
			new_pos = make_move(pos, sources[i], dests[i])
			rotate(&new_pos)
			v = max(
				v,
				alpha_beta_bottom_level_parallel(new_pos, 1, depth - 1, num_threads, alpha, beta, move)
			)
			# Prune the rest of the children, don't need to look
			if v > beta:
				return v
			if v > alpha:
				alpha = v
				move[0] = sources[i]
				move[1] = dests[i]
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
				alpha_beta_bottom_level_parallel(new_pos, 0, depth - 1, num_threads, alpha, beta, move)
			)
			# Too negative for max to allow this
			if v < alpha:
				return v
			beta = min(beta, v)
		return v