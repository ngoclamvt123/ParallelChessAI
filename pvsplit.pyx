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
from alphabeta cimport alpha_beta_serial

# Python wrapper for alpha beta helper
cpdef _pvsplit(np.int32_t[:] board,
									np.uint8_t[:] wc,
									np.uint8_t[:] bc,
									np.int32_t ep,
									np.int32_t kp,
									np.int32_t score,
									int agentIndex,
									int depth,
									int alpha,
									int beta,
									int num_threads):
	cdef:
		int32_t move[2]
		Position pos = init_position(board, wc, bc, ep, kp, score)
		int bestValue = pvsplit(pos, agentIndex, depth, alpha, beta, num_threads, move)
	
	return (bestValue, move)

cdef int pvsplit(Position pos, int agentIndex, int depth, int a, int b, int num_threads, int32_t *move) nogil:
	cdef:
		int i, j, max_idx, max_eval, curr_eval
		np.int32_t[:] alpha, beta, res, score
		Position new_pos
		int32_t move_count, temp
		int32_t sources[MAX_MOVES]
		int32_t dests[MAX_MOVES]
		Position positions[MAX_MOVES]
		omp_lock_t eval_lock
		int32_t new_move[2]

	move_count = gen_moves(pos, sources, dests)
	if agentIndex == 0:
		if depth == 0:
			return evaluate(pos.board)
		
		max_eval = -100000
		for i in range(move_count):
			positions[i] = make_move(pos, sources[i], dests[i])
			rotate(&positions[i])
			curr_eval = evaluate(positions[i].board)
			if curr_eval > max_eval:
				max_eval = curr_eval
				max_idx = i

		with gil:
			alpha = np.array([a], dtype=np.int32)
			beta = np.array([b], dtype=np.int32)
			res = np.array([-100000], dtype=np.int32)
			score = np.array([-100000], dtype=np.int32)

		res[0] = max(res[0],
			pvsplit(positions[max_idx], 1, depth-1, alpha[0], beta[0], num_threads, move))
		alpha[0] = max(alpha[0], res[0])
		score[0] = res[0]
		omp_init_lock(&eval_lock)
		for i in prange(move_count, num_threads=num_threads, nogil=True, schedule='guided'):
			if i == max_idx:
				continue
			temp = alpha_beta_serial(positions[i], 1, depth - 1, alpha[0], beta[0], new_move)
			omp_set_lock(&eval_lock)
			if temp > score[0]:
				score[0] = temp
				move[0] = sources[i]
				move[1] = dests[i]
			if temp >= beta[0]:
				omp_unset_lock(&eval_lock)
				omp_destroy_lock(&eval_lock)
				return temp
			alpha[0] = max(alpha[0], temp)
			omp_unset_lock(&eval_lock)
		omp_destroy_lock(&eval_lock)
		return score[0]
	else:
		if depth == 0:
			return -1*evaluate(pos.board)

		max_eval = 100000
		for i in range(move_count):
			positions[i] = make_move(pos, sources[i], dests[i])
			rotate(&positions[i])
			curr_eval = -1*evaluate(positions[i].board)
			if curr_eval < max_eval:
				max_eval = curr_eval
				max_idx = i

		# TODO try to get rid of this
		with gil:
			alpha = np.array([a], dtype=np.int32)
			beta = np.array([b], dtype=np.int32)
			res = np.array([100000], dtype=np.int32)
			score = np.array([100000], dtype=np.int32)

		res[0] = min(res[0],
			pvsplit(positions[max_idx], 0, depth-1, alpha[0], beta[0], num_threads, move))
		beta[0] = min(beta[0], res[0])
		score[0] = res[0]
		omp_init_lock(&eval_lock)
		for i in prange(move_count, num_threads=num_threads, nogil=True, schedule='guided'):
			if i == max_idx:
				continue
			temp = alpha_beta_serial(positions[i], 0, depth - 1, alpha[0], beta[0], new_move)
			omp_set_lock(&eval_lock)
			score[0] = min(temp, score[0])
			if temp <= alpha[0]:
				omp_unset_lock(&eval_lock)
				omp_destroy_lock(&eval_lock)
				return temp
			beta[0] = min(beta[0], temp)
			omp_unset_lock(&eval_lock)
		omp_destroy_lock(&eval_lock)
		return score[0]