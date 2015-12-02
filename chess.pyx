#cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from sunfish import print_numpy

###############################################################################
# Globals
###############################################################################
cdef enum:
	# Board length
	n = 120

	# Piece constants
	nline = -3
	space = -2
	empty = -1
	opp_pawn = 0
	opp_knight = 1
	opp_bishop = 2
	opp_rook = 3
	opp_queen = 4
	opp_king = 5
	self_pawn = 6
	self_knight = 7
	self_bishop = 8
	self_rook = 9
	self_queen = 10
	self_king = 11

	# Number of piece types
	pieces = 6

	# Our board is represented as a 120 numpy array. The padding allows for
	# fast detection of moves that don't stay within the board.
	A1 = 91
	H1 = 98
	A8 = 21
	H8 = 28

	# Direction constants
	N = -10 
	E = 1
	S = 10
	W = -1

	MAX_DIRS = 8
	MAXINT = 999999
	MAX_MOVES = 140

np_directions = np.array([
		   [ N, 2*N, N+W, N+E, MAXINT, MAXINT, MAXINT, MAXINT], # Pawn
		   [ 2*N+E, N+2*E, S+2*E, 2*S+E, 2*S+W, S+2*W, N+2*W, 2*N+W], # Knight
		   [ N+E, S+E, S+W, N+W, MAXINT, MAXINT, MAXINT, MAXINT], # Bishop
		   [ N, E, S, W, MAXINT, MAXINT, MAXINT, MAXINT], # Rook
		   [ N, E, S, W, N+E, S+E, S+W, N+W], # Queen
		   [ N, E, S, W, N+E, S+E, S+W, N+W ]
		], dtype=np.int32) # King

np_pst_vals = np.array([
	# Pawn
	[0,  0,  0,  0,  0,  0,  0,  0,
	50, 50, 50, 50, 50, 50, 50, 50,
	10, 10, 20, 30, 30, 20, 10, 10,
	 5,  5, 10, 25, 25, 10,  5,  5,
	 0,  0,  0, 20, 20,  0,  0,  0,
	 5, -5,-10,  0,  0,-10, -5,  5,
	 5, 10, 10,-20,-20, 10, 10,  5,
	 0,  0,  0,  0,  0,  0,  0,  0],
	# Knight
	[-50,-40,-30,-30,-30,-30,-40,-50,
	-40,-20,  0,  0,  0,  0,-20,-40,
	-30,  0, 10, 15, 15, 10,  0,-30,
	-30,  5, 15, 20, 20, 15,  5,-30,
	-30,  0, 15, 20, 20, 15,  0,-30,
	-30,  5, 10, 15, 15, 10,  5,-30,
	-40,-20,  0,  5,  5,  0,-20,-40,
	-50,-40,-30,-30,-30,-30,-40,-50],
	# Bishop
	[-20,-10,-10,-10,-10,-10,-10,-20,
	-10,  0,  0,  0,  0,  0,  0,-10,
	-10,  0,  5, 10, 10,  5,  0,-10,
	-10,  5,  5, 10, 10,  5,  5,-10,
	-10,  0, 10, 10, 10, 10,  0,-10,
	-10, 10, 10, 10, 10, 10, 10,-10,
	-10,  5,  0,  0,  0,  0,  5,-10,
	-20,-10,-10,-10,-10,-10,-10,-20],
	# Rook
	[ 0,  0,  0,  0,  0,  0,  0,  0,
	  5, 10, 10, 10, 10, 10, 10,  5,
	 -5,  0,  0,  0,  0,  0,  0, -5,
	 -5,  0,  0,  0,  0,  0,  0, -5,
	 -5,  0,  0,  0,  0,  0,  0, -5,
	 -5,  0,  0,  0,  0,  0,  0, -5,
	 -5,  0,  0,  0,  0,  0,  0, -5,
	  0,  0,  0,  5,  5,  0,  0,  0],
	# Queen
	[-20,-10,-10, -5, -5,-10,-10,-20,
	-10,  0,  0,  0,  0,  0,  0,-10,
	-10,  0,  5,  5,  5,  5,  0,-10,
	 -5,  0,  5,  5,  5,  5,  0, -5,
	  0,  0,  5,  5,  5,  5,  0, -5,
	-10,  5,  5,  5,  5,  5,  0,-10,
	-10,  0,  5,  0,  0,  0,  0,-10,
	-20,-10,-10, -5, -5,-10,-10,-20],
	# King
	[-30,-40,-40,-50,-50,-40,-40,-30,
	-30,-40,-40,-50,-50,-40,-40,-30,
	-30,-40,-40,-50,-50,-40,-40,-30,
	-30,-40,-40,-50,-50,-40,-40,-30,
	-20,-30,-30,-40,-40,-30,-30,-20,
	-10,-20,-20,-20,-20,-20,-20,-10,
	 20, 20,  0,  0,  0,  0, 20, 20,
	 20, 30, 10,  0,  0, 10, 30, 20]
], dtype=np.int32)

empty_moves = np.array([[0, 0]] * MAX_MOVES)

cdef:
	np.int32_t[:, :] directions = np_directions
	# 20,000 cutoff value derived by Claude Shannon
	np.int32_t[:] piece_vals = np.array([
		100, 320, 330, 500, 900, 20000], dtype=np.int32
	)

	np.int32_t[:, :] pst_vals = np_pst_vals

	# Endgame
	np.int32_t[:] king_end = np.array(
		[-50,-40,-30,-20,-20,-30,-40,-50,
		-30,-20,-10,  0,  0,-10,-20,-30,
		-30,-10, 20, 30, 30, 20,-10,-30,
		-30,-10, 30, 40, 40, 30,-10,-30,
		-30,-10, 30, 40, 40, 30,-10,-30,
		-30,-10, 20, 30, 30, 20,-10,-30,
		-30,-30,  0,  0,  0,  0,-30,-30,
		-50,-30,-30,-30,-30,-30,-30,-50],
		dtype = np.int32
	)

	np.int32_t[:] pawn_end = np.array(
		[0,  0,  0,  0,  0,  0,  0,  0,
		50, 50, 50, 50, 50, 50, 50, 50,
		30, 30, 30, 30, 30, 30, 30, 30,
		20, 20, 20, 25, 25, 20, 20, 20,
		10, 10, 10, 15, 15, 10, 10, 10,
		 0,  0,  0,  0,  0,  0,  0,  0,
	   -15,-20,-20,-20,-20,-20,-20,-15,
		 0,  0,  0,  0,  0,  0,  0,  0],
		dtype = np.int32
	)



ctypedef struct Position:
	np.int32_t[:] board
	np.uint8_t[:] wc
	np.uint8_t[:] bc
	np.int32_t ep
	np.int32_t kp
	np.int32_t score

###############################################################################
# Chess logic
###############################################################################

cpdef np.int32_t[:, :] gen_moves(Position pos) nogil:
	cdef:
		int i, j, k
		np.int32_t d, piece, dest
		np.int32_t arr_idx = 0
		np.int32_t[:, :] result 

	with gil:
		result = np.array([[0, 0]] * MAX_MOVES, dtype=np.int32)
		# np.int32_t[:] result = np.array([[]], dtype = np.int32)
	# For each of our pieces, iterate through each possible 'ray' of moves,
	# as defined in the 'directions' map. The rays are broken e.g. by
	# captures or immediately in case of pieces such as knights.

	# result = []
	for i in range(n):
		piece = pos.board[i]

		# skip if this piece does not belong to player of interest
		if piece < self_pawn:
			continue

		for k in range(MAX_DIRS):
			d = directions[piece % pieces, k]
			if d == MAXINT:
				break

			j = i+d

			while True:
				dest = pos.board[j]

				# Stay inside the board
				if dest == nline or dest == space:
					break

				# Castling
				if i == A1 and dest == self_king and pos.wc[0]:
					result[arr_idx, 0] = j
					result[arr_idx, 1] = j-2
					arr_idx += 1
					# with gil:
					# 	result.append((j, j-2))

				if i == H1 and dest == self_king and pos.wc[1]:
					# result[arr_idx] = np.array([j, j+2])
					result[arr_idx, 0] = j
					result[arr_idx, 1] = j+2
					arr_idx += 1						
					# with gil:
					# 	result.append((j, j+2))

				# No friendly captures
				if dest >= self_pawn:
					break

				# Pawn promotion
				if piece == self_pawn and d in (N+W, N+E) and dest == empty and j not in (pos.ep, pos.kp):
					break

				if piece == self_pawn and d in (N, 2*N) and dest != empty:
					break

				if piece == self_pawn and d == 2*N and (i < A1+N or pos.board[i+N] != empty):
					break

				# Move it
				# result[arr_idx] = np.array([i, j])
				result[arr_idx, 0] = i
				result[arr_idx, 1] = j
				arr_idx += 1
				# with gil:
				# 	result.append((i, j))

				# Stop crawlers from sliding
				if piece in (self_pawn, self_knight, self_king):
					break

				# No sliding after captures
				if dest >= opp_pawn and dest < self_pawn:
					break

				j += d

	return result[:arr_idx]


cdef inline Position rotate(Position pos) nogil:
	
	cdef:
		int i
		Position new_pos

	with gil:
		new_pos.board = pos.board.copy()
		new_pos.wc = pos.wc.copy()
		new_pos.bc = pos.bc.copy()
		new_pos.ep = 0
		new_pos.kp = 0

	for i in range(n):
		if pos.board[i] >= 0:
			new_pos.board[i] = (pos.board[i] + 6) % 12

	new_pos.board = new_pos.board[::-1]
	new_pos.score = pos.score * -1
	new_pos.ep = 119-pos.ep
	new_pos.kp = 119-pos.kp

	return new_pos

cpdef Position make_move(Position pos, np.int32_t[:] move) nogil:
	cdef:
		np.int32_t i, j, piece, dest
		Position new_pos

	with gil:
		# Grab source and destination of move
		i = move[0]
		j = move[1]
		piece = pos.board[i]
		dest = pos.board[j]

		# Create copy of variables and apply 
		new_pos.board = pos.board.copy()
		new_pos.wc = pos.wc.copy()
		new_pos.bc = pos.bc.copy()
		new_pos.ep = 0
		new_pos.kp = 0
		new_pos.board[j] = pos.board[i]
		new_pos.board[i] = empty

		# Castling rights
		if i == A1:
			new_pos.wc[0] = 0
			new_pos.wc[1] = pos.wc[1]

		if i == H1:
			new_pos.wc[0] = pos.wc[0]
			new_pos.wc[1] = 0

		if j == A8:
			new_pos.bc[0] = pos.bc[0]
			new_pos.bc[1] = 0

		if j == H8:
			new_pos.wc[0] = 0
			new_pos.bc[1] = pos.bc[1]

		# Castling
		if piece == self_king:
			new_pos.wc[0] = 0
			new_pos.wc[1] = 0
			if abs(j-i) == 2:
				new_pos.kp = (i+j)//2
				new_pos.board[A1 if j < i else H1] = empty
				new_pos.board[new_pos.kp] = self_rook

		# Pawn promotion
		if piece == self_pawn:
			if A8 <= j and j <= H8:
				new_pos.board[j] = self_queen
			if j - i == 2*N:
				ep = i + N
			if j - i in (N+W, N+E) and dest == empty:
				new_pos.board[j+S] = empty

		# Return result
		new_pos.score = pos.score + evaluate(new_pos.board)
		return new_pos

cdef np.int32_t total_material(np.int32_t[:] board) nogil:
	cdef:
		np.int32_t amt = 0
		np.int32_t piece

	for idx in range(120):
		piece = board[idx]
		if piece >= 0:
			amt += piece_vals[piece % 6]

	return amt


cdef np.int32_t is_endgame(np.int32_t[:] board) nogil:
	cdef np.int32_t ret_val = 1
	# material cutoff
	# roughly 2 Kings, 2 Rooks, 1 Minor, 6 Pawns each
	if total_material(board) > 44000:
		ret_val = 0

	return ret_val


cdef np.int32_t evaluate(np.int32_t[:] board) nogil:
	cdef:
		np.int32_t score = 0
		np.int32_t row, col, pos, piece, endgame_bool, idx

	endgame_bool = is_endgame(board)
	for idx in range(120):
		piece = board[idx]
		
		if piece >= 0:
			row = idx / 10 - 2
			col = idx % 10 - 1
			pos = row * 8 + col

			# opponent's piece
			if piece <= 5:
				pos = 63 - pos

				score -= piece_vals[piece]

				if endgame_bool == 1:
					if piece == 0: score -= pawn_end[pos]
					elif piece == 5: score -= king_end[pos]
					else: score -= pst_vals[piece][pos]
				else:
					score -= pst_vals[piece][pos]

			# my piece
			else:
				score += piece_vals[piece % 6]

				if endgame_bool == 1:
					if piece == 6: score += pawn_end[pos]
					elif piece == 11: score += king_end[pos]
					else: score += pst_vals[piece % 6][pos]
				else:
					score += pst_vals[piece % 6][pos]

	return score


cpdef int minimax_helper(Position pos, int agentIndex, int depth) nogil:
	# Right now this is all within the GIL. The only way I can see this getting fixed
	# is if we rewrite all the methods as cython functions on numpy arrays
	cdef:
		np.int32_t[:] move
		np.int32_t[:,:] moves
		int i, ret, bestValue

	if depth == 0:
		if agentIndex == 0:
			ret = evaluate(pos.board)
			#with gil: print ("agent 0 ", ret)
			with gil:
				print_numpy(pos.board)
				print (ret)
				print ("----------")
			return ret
		else:
			ret = evaluate(pos.board)
			# with gil: print ("agent 1 ", ret)
			with gil:
				print_numpy(pos.board)
				print (ret)
				print ("-----------")
			return -1 * ret
	# Agent index 0 is the computer, trying to maximize the scoreboard
	if agentIndex == 0:
		bestValue = -100000
		moves = gen_moves(pos)
		for i in range(moves.shape[0]):
			move = moves[i]
			bestValue = max(bestValue, minimax_helper(rotate(make_move(pos, move)), 1, depth - 1))
		return bestValue
	# Agent index 1 is the human, trying to minimize the scoreboard
	elif agentIndex == 1:
		# with gil:
		# 	print_numpy(pos.board)
		# 	pos = rotate(pos)
		# 	print_numpy(pos.board)
		# 	raw_input()
		bestValue = 1000000
		moves = gen_moves(pos)
		for i in range(moves.shape[0]):
			move = moves[i]
			bestValue = min(bestValue, minimax_helper(rotate(make_move(pos, move)), 0, depth -1))
		return bestValue


	



