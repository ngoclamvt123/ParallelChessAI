#cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t, uint8_t
from libc.stdlib cimport abs

import pyximport
pyximport.install()

cimport cython
from constants cimport *

from sunfish import print_numpy

###############################################################################
# Globals
###############################################################################

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

cdef:
	# Eval count
	int EVALCOUNT

	# Directions pieces can go
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

cpdef Position init_position(np.int32_t[:] board,
							np.uint8_t[:] wc,
							np.uint8_t[:] bc,
							np.int32_t ep,
							np.int32_t kp,
							np.int32_t score) nogil:
	cdef:
		int i
		Position pos

	for i in range(MAX_BOARD_SIZE):
		pos.board[i] = board[i]

	pos.wc[0] = wc[0]
	pos.wc[1] = wc[1]

	pos.bc[0] = bc[0]
	pos.bc[1] = bc[1]

	pos.ep = ep
	pos.kp = kp
	pos.score = score

	return pos

cdef Position clone_position(np.int32_t *board,
							np.uint8_t *wc,
							np.uint8_t *bc,
							np.int32_t ep,
							np.int32_t kp,
							np.int32_t score) nogil:
	cdef:
		int i
		Position pos

	for i in range(MAX_BOARD_SIZE):
		pos.board[i] = board[i]

	pos.wc[0] = wc[0]
	pos.wc[1] = wc[1]

	pos.bc[0] = bc[0]
	pos.bc[1] = bc[1]

	pos.ep = ep
	pos.kp = kp
	pos.score = score

	return pos


###############################################################################
# Chess logic
###############################################################################

# Python wrapper for gen_moves
cpdef _gen_moves(np.int32_t[:] board,
									np.uint8_t[:] wc,
									np.uint8_t[:] bc,
									np.int32_t ep,
									np.int32_t kp,
									np.int32_t score):

	cdef: 
		int32_t sources[MAX_MOVES]
		int32_t dests[MAX_MOVES]
		int32_t move_count = gen_moves(init_position(board, wc, bc, ep, kp, score), sources, dests)

	return (move_count, sources, dests)


cdef int32_t gen_moves(Position pos, int32_t *sources, int32_t *dests) nogil:
	cdef:
		int i, j, k
		np.int32_t d, piece, dest
		int32_t move_count = 0

	# For each of our pieces, iterate through each possible 'ray' of moves,
	# as defined in the 'directions' map. The rays are broken e.g. by
	# captures or immediately in case of pieces such as knights.

	for i in range(MAX_BOARD_SIZE):
		piece = pos.board[i]

		# skip if this piece does not belong to player of interest
		if piece < SELF_PAWN:
			continue

		for k in range(MAX_DIRS):
			d = directions[piece % PIECES, k]
			if d == MAXINT:
				break

			j = i+d

			while True:
				dest = pos.board[j]

				# Stay inside the board
				if dest == NLINE or dest == SPACE:
					break

				# Castling
				if i == A1 and dest == SELF_KING and pos.wc[0]:
					sources[move_count] = j
					dests[move_count] = j-2
					move_count += 1

				if i == H1 and dest == SELF_KING and pos.wc[1]:
					# result[move_count] = np.array([j, j+2])
					sources[move_count] = j
					dests[move_count] = j+2
					move_count += 1

				# No friendly captures
				if dest >= SELF_PAWN:
					break

				# Pawn promotion
				if piece == SELF_PAWN and d in (N+W, N+E) and dest == EMPTY and j not in (pos.ep, pos.kp):
					break

				if piece == SELF_PAWN and d in (N, 2*N) and dest != EMPTY:
					break

				if piece == SELF_PAWN and d == 2*N and (i < A1+N or pos.board[i+N] != EMPTY):
					break

				# Move it
				sources[move_count] = i
				dests[move_count] = j
				move_count += 1

				# Stop crawlers from sliding
				if piece in (SELF_PAWN, SELF_KNIGHT, SELF_KING):
					break

				# No sliding after captures
				if dest >= OPP_PAWN and dest < SELF_PAWN:
					break

				j += d

	return move_count

# Rotate the board for opponent
cdef void rotate(Position* pos) nogil:
	
	cdef:
		int i, j 
		np.int32_t temp

	for i in range(MAX_BOARD_SIZE/2):
		j = MAX_BOARD_SIZE - i - 1
		if pos.board[i] >= 0:
			pos.board[i] = (pos.board[i] + 6) % 12
		if pos.board[j] >= 0:
			pos.board[j] = (pos.board[j] + 6) % 12

		temp = pos.board[i];
		pos.board[i] = pos.board[j];
		pos.board[j] = temp;

	pos.score = pos.score * -1
	pos.ep = 119-pos.ep
	pos.kp = 119-pos.kp

# Python wrapper for make_move
cpdef Position _make_move(np.int32_t[:] board,
									np.uint8_t[:] wc,
									np.uint8_t[:] bc,
									np.int32_t ep,
									np.int32_t kp,
									np.int32_t score,
									np.int32_t[:] move):
	cdef:
		Position pos = init_position(board, wc, bc, ep, kp, score)

	return make_move(pos, move[0], move[1])

cdef Position make_move(Position pos, int32_t i, int32_t j) nogil:
	# Note that i is the source index, and j is the destination index
	cdef:
		np.int32_t piece, dest
		Position new_pos

	# Grab source and destination of move
	piece = pos.board[i]
	dest = pos.board[j]

	# Copy position and apply move
	new_pos = clone_position(pos.board, 
							pos.wc, 
							pos.bc, 
							0, 0, 0)

	new_pos.board[j] = piece
	new_pos.board[i] = EMPTY

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
	if piece == SELF_KING:
		new_pos.wc[0] = 0
		new_pos.wc[1] = 0
		if abs(j-i) == 2:
			new_pos.kp = (i+j)//2
			new_pos.board[A1 if j < i else H1] = EMPTY
			new_pos.board[new_pos.kp] = SELF_ROOK

	# Pawn promotion
	if piece == SELF_PAWN:
		if A8 <= j and j <= H8:
			new_pos.board[j] = SELF_QUEEN
		if j - i == 2*N:
			ep = i + N
		if j - i in (N+W, N+E) and dest == EMPTY:
			new_pos.board[j+S] = EMPTY

	# Return result
	new_pos.score = pos.score + evaluate(new_pos.board)
	return new_pos

cdef np.int32_t total_material(np.int32_t* board) nogil:
	cdef:
		np.int32_t amt = 0
		np.int32_t piece

	for idx in range(120):
		piece = board[idx]
		if piece >= 0:
			amt += piece_vals[piece % 6]

	return amt

cdef np.int32_t is_endgame(np.int32_t* board) nogil:
	cdef np.int32_t ret_val = 1
	# material cutoff
	# roughly 2 Kings, 2 Rooks, 1 Minor, 6 Pawns each
	if total_material(board) > 44000:
		ret_val = 0

	return ret_val

cdef np.int32_t evaluate(np.int32_t* board) nogil:
	cdef:
		np.int32_t score = 0
		np.int32_t row, col, pos, piece, endgame_bool, idx
	
	endgame_bool = is_endgame(board)
	
	global EVALCOUNT
	EVALCOUNT += 1
	
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

cpdef int print_eval():
	return EVALCOUNT
