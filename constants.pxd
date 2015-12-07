cdef enum:
	# Piece constants
	NLINE = -3
	SPACE = -2
	EMPTY = -1
	OPP_PAWN = 0
	OPP_KNIGHT = 1
	OPP_BISHOP = 2
	OPP_ROOK = 3
	OPP_QUEEN = 4
	OPP_KING = 5
	SELF_PAWN = 6
	SELF_KNIGHT = 7
	SELF_BISHOP = 8
	SELF_ROOK = 9
	SELF_QUEEN = 10
	SELF_KING = 11

	# Number of piece types
	PIECES = 6

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
	MAX_BOARD_SIZE = 120