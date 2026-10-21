import random
import time
from enum import Enum

# Constants
WIN = 7          # Connected five
FLEX4 = 6        # Open four
BLOCK4 = 5       # Blocked four
FLEX3 = 4        # Open three
BLOCK3 = 3       # Blocked three
FLEX2 = 2        # Open two
BLOCK2 = 1       # Blocked two
NTYPE = 8        # Number of pattern types
MAX_SIZE = 20    # Maximum board size
MAX_MOVES = 40   # Maximum number of moves per layer
HASH_SIZE = 1 << 22  # Normal hash table size
PVS_SIZE = 1 << 20   # PVS hash table size
MAX_DEPTH = 20   # Maximum search depth
MIN_DEPTH = 4    # Minimum search depth (increased from 2)

# Hash related constants
HASH_EXACT = 0
HASH_ALPHA = 1
HASH_BETA = 2
UNKNOWN = -20000

# Direction vectors
dx = [1, 0, 1, 1]
dy = [0, 1, 1, -1]

# Piece states
class Pieces(Enum):
    WHITE = 0
    BLACK = 1
    EMPTY = 2
    OUTSIDE = 3

class Pos:
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y

class Point:
    def __init__(self, p=None, val=0):
        self.p = p if p is not None else Pos()
        self.val = val

class Cell:
    def __init__(self):
        self.piece = Pieces.EMPTY.value
        self.is_cand = 0
        self.pattern = [[0, 0, 0, 0], [0, 0, 0, 0]]  # Black and white patterns in 4 directions

class Hashe:
    def __init__(self):
        self.key = 0
        self.depth = 0
        self.hashf = 0
        self.val = 0

class Pv:
    def __init__(self):
        self.key = 0
        self.best = Pos()

class Line:
    def __init__(self):
        self.n = 0
        self.moves = [Pos() for _ in range(MAX_DEPTH)]

class MoveList:
    def __init__(self):
        self.phase = 0
        self.n = 0
        self.index = 0
        self.first = False
        self.hash_move = Pos()
        self.moves = [Pos() for _ in range(64)]

class Board:
    def __init__(self):
        self.step = 0
        self.size = 15
        self.b_start = 0
        self.b_end = 0
        self.zobrist_key = 0
        self.zobrist = [[[0 for _ in range(MAX_SIZE + 4)] for _ in range(MAX_SIZE + 4)] for _ in range(2)]
        self.hash_table = [Hashe() for _ in range(HASH_SIZE)]
        self.pvs_table = [Pv() for _ in range(PVS_SIZE)]
        self.type_table = [[[[0 for _ in range(3)] for _ in range(6)] for _ in range(6)] for _ in range(10)]
        self.pattern_table = [[0, 0] for _ in range(65536)]
        self.pval = [[[[0 for _ in range(8)] for _ in range(8)] for _ in range(8)] for _ in range(8)]
        self.cell = [[Cell() for _ in range(MAX_SIZE + 8)] for _ in range(MAX_SIZE + 8)]
        self.rem_move = [Pos() for _ in range(MAX_SIZE * MAX_SIZE)]
        self.cand = [Point() for _ in range(256)]
        self.is_lose = [[False for _ in range(MAX_SIZE + 4)] for _ in range(MAX_SIZE + 4)]
        self.who = Pieces.BLACK.value
        self.opp = Pieces.WHITE.value
        self.root_move = [Point() for _ in range(64)]
        self.root_count = 0
        self.ply = 0

        self.init_chess_type()
        self.init_zobrist()

    def rand64(self):
        # Generate a 64-bit random number
        return random.randint(0, (1 << 64) - 1)

    def init_zobrist(self):
        random.seed(time.time())
        for i in range(MAX_SIZE + 4):
            for j in range(MAX_SIZE + 4):
                self.zobrist[0][i][j] = self.rand64()
                self.zobrist[1][i][j] = self.rand64()

    def set_size(self, size):
        self.size = size
        self.b_start, self.b_end = 4, size + 4
        for i in range(MAX_SIZE + 8):
            for j in range(MAX_SIZE + 8):
                if i < 4 or i >= size + 4 or j < 4 or j >= size + 4:
                    self.cell[i][j].piece = Pieces.OUTSIDE.value
                else:
                    self.cell[i][j].piece = Pieces.EMPTY.value

    def make_move(self, next_pos):
        x, y = next_pos.x, next_pos.y
        self.ply += 1
        self.cell[x][y].piece = self.who
        self.zobrist_key ^= self.zobrist[self.who][x][y]
        self.who, self.opp = self.opp, self.who
        self.rem_move[self.step] = next_pos
        self.step += 1
        self.update_type(x, y)

        for i in range(x - 2, x + 3):
            for j in range(y - 2, y + 3):
                if self.check_xy(i, j):
                    self.cell[i][j].is_cand += 1

    def del_move(self):
        self.step -= 1
        x, y = self.rem_move[self.step].x, self.rem_move[self.step].y
        self.ply -= 1
        self.who, self.opp = self.opp, self.who
        self.zobrist_key ^= self.zobrist[self.who][x][y]
        self.cell[x][y].piece = Pieces.EMPTY.value
        self.update_type(x, y)

        for i in range(x - 2, x + 3):
            for j in range(y - 2, y + 3):
                if self.check_xy(i, j):
                    self.cell[i][j].is_cand -= 1

    def undo(self):
        if self.step >= 2:
            self.del_move()
            self.del_move()
        elif self.step == 1:
            self.del_move()

    def restart(self):
        self.pvs_table = [Pv() for _ in range(PVS_SIZE)]
        self.hash_table = [Hashe() for _ in range(HASH_SIZE)]
        while self.step:
            self.del_move()

    def update_type(self, x, y):
        for i in range(4):
            # Update in positive direction
            a, b = x + dx[i], y + dy[i]
            for j in range(4):
                if not self.check_xy(a, b):
                    break
                key = self.get_key(a, b, i)
                self.cell[a][b].pattern[0][i] = self.pattern_table[key][0]
                self.cell[a][b].pattern[1][i] = self.pattern_table[key][1]
                a, b = a + dx[i], b + dy[i]

            # Update in negative direction
            a, b = x - dx[i], y - dy[i]
            for j in range(4):
                if not self.check_xy(a, b):
                    break
                key = self.get_key(a, b, i)
                self.cell[a][b].pattern[0][i] = self.pattern_table[key][0]
                self.cell[a][b].pattern[1][i] = self.pattern_table[key][1]
                a, b = a - dx[i], b - dy[i]

    def get_key(self, x, y, i):
        step_x, step_y = dx[i], dy[i]
        key = (self.cell[x - step_x * 4][y - step_y * 4].piece) ^ \
              (self.cell[x - step_x * 3][y - step_y * 3].piece << 2) ^ \
              (self.cell[x - step_x * 2][y - step_y * 2].piece << 4) ^ \
              (self.cell[x - step_x * 1][y - step_y * 1].piece << 6) ^ \
              (self.cell[x + step_x * 1][y + step_y * 1].piece << 8) ^ \
              (self.cell[x + step_x * 2][y + step_y * 2].piece << 10) ^ \
              (self.cell[x + step_x * 3][y + step_y * 3].piece << 12) ^ \
              (self.cell[x + step_x * 4][y + step_y * 4].piece << 14)
        return key

    def line_type(self, role, key):
        line_left = [0] * 9
        line_right = [0] * 9
        
        for i in range(9):
            if i == 4:
                line_left[i] = role
                line_right[i] = role
            else:
                line_left[i] = key & 3
                line_right[8 - i] = key & 3
                key >>= 2
        
        # Check patterns from left to right and right to left
        p1 = self.short_line(line_left)
        p2 = self.short_line(line_right)
        
        # If both directions show blocked three, check for open three
        if p1 == BLOCK3 and p2 == BLOCK3:
            return self.check_flex3(line_left)
        # If both directions show blocked four, check for open four
        elif p1 == BLOCK4 and p2 == BLOCK4:
            return self.check_flex4(line_left)
        # Return the stronger pattern
        else:
            return max(p1, p2)

    def check_flex3(self, line):
        role = line[4]
        for i in range(9):
            if line[i] == Pieces.EMPTY.value:
                line[i] = role
                type_val = self.check_flex4(line)
                line[i] = Pieces.EMPTY.value
                if type_val == FLEX4:
                    return FLEX3
        return BLOCK3

    def check_flex4(self, line):
        role = line[4]
        five = 0
        
        for i in range(9):
            if line[i] == Pieces.EMPTY.value:
                count = 0
                # Count pieces to the left
                for j in range(i - 1, -1, -1):
                    if line[j] == role:
                        count += 1
                    else:
                        break
                # Count pieces to the right
                for j in range(i + 1, 9):
                    if line[j] == role:
                        count += 1
                    else:
                        break
                if count >= 4:
                    five += 1
        
        # If there are two ways to win, it's an open four
        return FLEX4 if five >= 2 else BLOCK4

    def short_line(self, line):
        kong = 0
        block = 0
        len_val = 1
        len2 = 1
        count = 1
        
        who = line[4]
        
        # Look right
        for k in range(5, 9):
            if line[k] == who:
                if kong + count > 4:
                    break
                count += 1
                len_val += 1
                len2 = kong + count
            elif line[k] == Pieces.EMPTY.value:
                len_val += 1
                kong += 1
            else:
                if line[k - 1] == who:
                    block += 1
                break
        
        # Calculate middle space
        kong = len2 - count
        
        # Look left
        for k in range(3, -1, -1):
            if line[k] == who:
                if kong + count > 4:
                    break
                count += 1
                len_val += 1
                len2 = kong + count
            elif line[k] == Pieces.EMPTY.value:
                len_val += 1
                kong += 1
            else:
                if line[k + 1] == who:
                    block += 1
                break
        
        return self.type_table[len_val][len2][count][block]

    def generate_assist(self, len_val, len2, count, block):
        if len_val >= 5 and count > 1:
            if count == 5:
                return WIN
            if len_val > 5 and len2 < 5 and block == 0:
                if count == 2:
                    return FLEX2
                if count == 3:
                    return FLEX3
                if count == 4:
                    return FLEX4
            else:
                if count == 2:
                    return BLOCK2
                if count == 3:
                    return BLOCK3
                if count == 4:
                    return BLOCK4
        return 0

    def get_pval(self, a, b, c, d):
        type_count = [0] * 8
        type_count[a] += 1
        type_count[b] += 1
        type_count[c] += 1
        type_count[d] += 1
        
        if type_count[WIN] > 0:
            return 5000
        if type_count[FLEX4] > 0 or type_count[BLOCK4] > 1:
            return 1200
        if type_count[BLOCK4] > 0 and type_count[FLEX3] > 0:
            return 1000
        if type_count[FLEX3] > 1:
            return 200
        
        val = [0, 2, 5, 5, 12, 12]
        score = 0
        for i in range(1, BLOCK4 + 1):
            score += val[i] * type_count[i]
        
        return score

    def init_chess_type(self):
        # Chess type judgment auxiliary table
        for i in range(10):
            for j in range(6):
                for k in range(6):
                    for l in range(3):
                        self.type_table[i][j][k][l] = self.generate_assist(i, j, k, l)
        
        # Pattern table
        for key in range(65536):
            self.pattern_table[key][0] = self.line_type(0, key)
            self.pattern_table[key][1] = self.line_type(1, key)
        
        # Move evaluation table
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    for l in range(8):
                        self.pval[i][j][k][l] = self.get_pval(i, j, k, l)

    # Helper methods
    def color(self, step):
        return step & 1

    def check_xy(self, x, y):
        return self.cell[x][y].piece != Pieces.OUTSIDE.value

    def last_move(self):
        return self.cell[self.rem_move[self.step - 1].x][self.rem_move[self.step - 1].y]

    def type_count(self, c, role, type_count_arr):
        type_count_arr[c.pattern[role][0]] += 1
        type_count_arr[c.pattern[role][1]] += 1
        type_count_arr[c.pattern[role][2]] += 1
        type_count_arr[c.pattern[role][3]] += 1

    def is_type(self, c, role, type_val):
        return (c.pattern[role][0] == type_val or
                c.pattern[role][1] == type_val or
                c.pattern[role][2] == type_val or
                c.pattern[role][3] == type_val)

    def check_win(self):
        c = self.last_move()
        return (c.pattern[self.opp][0] == WIN or
                c.pattern[self.opp][1] == WIN or
                c.pattern[self.opp][2] == WIN or
                c.pattern[self.opp][3] == WIN)