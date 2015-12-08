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


# Python wrapper for minimax_serial
cpdef _minimax_serial(np.int32_t[:] board,
									np.uint8_t[:] wc,
									np.uint8_t[:] bc,
									np.int32_t ep,
									np.int32_t kp,
									np.int32_t score,
									int agentIndex,
									int depth):
	cdef:
		int32_t move[2]
		Position pos = init_position(board, wc, bc, ep, kp, score)
		int bestValue = minimax_serial(pos, agentIndex, depth, move)

	return (bestValue, move)

cdef int minimax_serial(Position pos, int agentIndex, int depth, int32_t *move) nogil:
	# Right now this is all within the GIL. The only way I can see this getting fixed
	# is if we rewrite all the methods as cython functions on numpy arrays
	cdef:
		int32_t move_count
		int i, bestValue, value
		int32_t sources[MAX_MOVES]
		int32_t dests[MAX_MOVES]
		Position new_pos
	
	if depth == 0:
		if agentIndex == 0:
			return evaluate(pos.board)
		else:
			return -1*evaluate(pos.board)
	
	move_count = gen_moves(pos, sources, dests)

	# Agent index 0 is the computer, trying to maximize the scoreboard
	if agentIndex == 0:
		bestValue = -100000
		for i in range(move_count):
			new_pos = make_move(pos, sources[i], dests[i])
			# with gil: 
			# 	print("move number", i, " thread id ", thread_id)
			rotate(&new_pos)
			value = minimax_serial(new_pos, 1, depth - 1, move)
			if value > bestValue:
				bestValue = value
				move[0] = sources[i]
				move[1] = dests[i]
	# Agent index 1 is the human, trying to minimize the scoreboard
	elif agentIndex == 1:
		bestValue = 1000000
		for i in range(move_count):
			# with gil: 
			# 	print("move number", i, " thread id ", thread_id)
			new_pos = make_move(pos, sources[i], dests[i])
			rotate(&new_pos)
			value = minimax_serial(new_pos, 0, depth - 1, move)
			if value < bestValue:
				bestValue = value
				move[0] = sources[i]
				move[1] = dests[i]
	
	return bestValue

# Python wrapper for minimax_serial
cpdef _minimax_top_level_parallel(np.int32_t[:] board,
									np.uint8_t[:] wc,
									np.uint8_t[:] bc,
									np.int32_t ep,
									np.int32_t kp,
									np.int32_t score,
									int agentIndex,
									int depth,
									int num_threads):
	cdef:
		int32_t move[2]
		Position pos = init_position(board, wc, bc, ep, kp, score)
		int bestValue = minimax_top_level_parallel(pos, agentIndex, depth, num_threads, move)

	return (bestValue, move)

cdef int minimax_top_level_parallel(Position pos, int agentIndex, int depth, int num_threads, int32_t *move) nogil:
	cdef:
		int32_t sources[MAX_MOVES]
		int32_t dests[MAX_MOVES]
		int32_t move_count
		int bestValue[1]
		int i #, thread_id
		Position new_pos
		omp_lock_t eval_lock
	if depth == 0:
		if agentIndex == 0:
			return evaluate(pos.board)
		else:
			return -1*evaluate(pos.board)
	
	move_count = gen_moves(pos, sources, dests)
	# Agent index 0 is the computer, trying to maximize the scoreboard
	omp_init_lock(&eval_lock)
	if agentIndex == 0:
		bestValue[0] = -100000
		for i in prange(move_count, num_threads=num_threads, nogil=True):
			# new_pos = make_move(pos, moves[i])
			# rotate(&new_pos)
			# with gil:
			# 	print ("computer. range", i)
			# 	print ("thread", threadid())
			# 	thread_id = threadid()
			# omp_set_lock(&eval_lock)
			# bestValue[0] = max(bestValue[0], minimax_serial(new_pos, 1, depth-1, 0))
			# omp_unset_lock(&eval_lock)
			bestValue[0] = max_helper_function(pos, sources[i], dests[i], depth, bestValue[0], move, eval_lock)
	# Agent index 1 is the human, trying to minimize the scoreboard
	elif agentIndex == 1:
		bestValue[0] = 100000
		for i in prange(move_count, num_threads=num_threads, nogil=True):
			# new_pos = make_move(pos, moves[i])
			# rotate(&new_pos)
			# with gil:
			# 	print ("human. range", i)
			# 	print ("thread", threadid())
			# 	thread_id = threadid()
			# omp_set_lock(&eval_lock)
			# bestValue[0] = min(bestValue[0], minimax_serial(new_pos, 0, depth-1, 0))
			# omp_unset_lock(&eval_lock)
			bestValue[0] = min_helper_function(pos, sources[i], dests[i], depth, bestValue[0], move, eval_lock)
		# omp_destroy_lock(&eval_lock)
	omp_destroy_lock(&eval_lock)
	return bestValue[0]

cdef int min_helper_function(Position pos, int32_t source, int32_t dest, int depth, int bestValue, int32_t *move, omp_lock_t eval_lock) nogil: 
	cdef:
		Position new_pos
		int value
		int32_t new_move[2]

	new_pos = make_move(pos, source, dest)
	rotate(&new_pos)
	value = minimax_serial(new_pos, 0, depth-1, new_move)
	omp_set_lock(&eval_lock)
	bestValue = max(bestValue, value)
	omp_unset_lock(&eval_lock)
	return bestValue

cdef int max_helper_function(Position pos, int source, int dest, int depth, int bestValue, int32_t *move, omp_lock_t eval_lock) nogil:
	cdef:
		Position new_pos
		int value
		int32_t new_move[2]

	new_pos = make_move(pos, source, dest)
	rotate(&new_pos)
	value = minimax_serial(new_pos, 1, depth-1, new_move)
	omp_set_lock(&eval_lock)
	if value > bestValue:
		bestValue = value
		move[0] = source
		move[1] = dest
	omp_unset_lock(&eval_lock)
	return bestValue