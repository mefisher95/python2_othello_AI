# This file is meant to be uploaded to the class AI website. 
# Any serious code changes should be done in their respective files
# and transfered here only once they are finalized.

import numpy as np
import platform
import math
import copy

# ***********************************************************************************************
# Bitmap
# ***********************************************************************************************
class Bitmap:
    def __init__(self, input=None, architecture=None):
        if architecture is None:
            self.architecture = platform.architecture()[0]
        else:
            self.architecture = architecture.lower()

        if self.architecture == "64bit":
            self.dtype = np.int64
            self.int_size = np.int64()
        elif self.architecture == "32bit":
            self.dtype = np.uint32
            self.int_size = np.uint32()
        elif self.architecture == "16bit":
            self.dtype = np.uint16
            self.int_size = np.uint16()
        elif self.architecture == "8bit":
            self.dtype = np.uint8
            self.int_size = np.uint8()
        else: 
            raise TypeError("invalid range for bit size")
            
        self.int_bytes = self.int_size.itemsize
        self.int_size = self.int_bytes * 8
        self.num_ints = 0

        if input is None:
            self.map =  np.array([0], dtype=self.dtype)
            self.num_ints = len(self.map)
            if self.num_ints == 0: self.num_ints = 1

        elif isinstance(input, list):
            input = np.array(input)
            self.map = []
            self.input(input, True)
            self.num_ints = len(self.map)
            if self.num_ints == 0: 
                self.num_ints = 1
        elif isinstance(input, np.ndarray):
            self.map = input.copy()
            self.num_ints = len(self.map)
        else:
            raise ValueError('Can only insert a list or numpy.ndarray into a Bitmap') 
        
    def input(self, input_array, expand=False):
        if isinstance(input_array, np.ndarray):
            input = input_array
        elif isinstance(input_array, list):
            input = np.array(input_array)
        else:
            raise ValueError('Can only insert a list or numpy.ndarray into a Bitmap')

        x = self.pack_bits_(input)
        i = 0
        new_list =[]
        while i < len(x) and i < self.num_ints:
            new_list.append(x[i])
            i += 1
        if expand:
            if len(x) > self.num_ints:
                while i < len(x):
                    new_list.append(x[i])
                    i += 1
                self.num_ints = len(x)
            elif len(x) < self.num_ints:
                while i < self.num_ints:
                    new_list.append(self.dtype(0))
                    i += 1
        new_list.reverse()
        self.map = np.array(new_list, dtype=self.dtype)

    def is_empty(self):
        for i in self.map:
            if i: return False
        return True

    def pop_count(self):
        if self.architecture == '64bit':
            return self.pop_count64()
        elif self.architecture == '32bit':
            return self.pop_count32()
        elif self.architecture == '16bit':
            return self.pop_count16()
        elif self.architecture == '8bit':
            return self.pop_count8()
        else:
            raise TypeError('Unsupported architecture for Population Count')

    def pop_count8(self):
        count = 0
        for n in self.map:
            n = (n & 0x55) + ((n & 0xAA) >> 1)
            n = (n & 0x33) + ((n & 0xCC) >> 2)
            n = (n & 0x0F) + ((n & 0xF0) >> 4)
            count += n
        return count

    def pop_count16(self):
        count = 0
        for n in self.map:
            n = (n & 0x5555) + ((n & 0xAAAA) >> 1)
            n = (n & 0x3333) + ((n & 0xCCCC) >> 2)
            n = (n & 0x0F0F) + ((n & 0xF0F0) >> 4)
            n = (n & 0x00FF) + ((n & 0xFF00) >> 8)
            count += n
        return count

    def pop_count32(self):
        count = 0
        for n in self.map:
            n = (n & 0x55555555) + ((n & 0xAAAAAAAA) >> 1)
            n = (n & 0x33333333) + ((n & 0xCCCCCCCC) >> 2)
            n = (n & 0x0F0F0F0F) + ((n & 0xF0F0F0F0) >> 4)
            n = (n & 0x00FF00FF) + ((n & 0xFF00FF00) >> 8)
            n = (n & 0x0000FFFF) + ((n & 0xFFFF0000) >> 16)
            count += n
        return count

    def pop_count64(self):
        count = 0
        for n in self.map:
            n = int(n)
            n = (n & 0x5555555555555555) + ((n & 0xAAAAAAAAAAAAAAAA) >> 1)
            n = (n & 0x3333333333333333) + ((n & 0xCCCCCCCCCCCCCCCC) >> 2)
            n = (n & 0x0F0F0F0F0F0F0F0F) + ((n & 0xF0F0F0F0F0F0F0F0) >> 4)
            n = (n & 0x00FF00FF00FF00FF) + ((n & 0xFF00FF00FF00FF00) >> 8)
            n = (n & 0x0000FFFF0000FFFF) + ((n & 0xFFFF0000FFFF0000) >> 16)
            n = (n & 0x00000000FFFFFFFF) + ((n & 0xFFFFFFFF00000000) >> 32) # This
            count += n
        return count

    def bytes_size(self):
        return len(self.map.tobytes())

    def get_bitlist(self):
        return [bool(int(i)) for i in str(self) if i == '0' or i == '1']

    def lshift(self, n=1):
        if isinstance(n, int):
            self = self << n
        else:
            raise ValueError("Can only lshift by an int, was provided a {0}".format(type(n)))

    def rshift(self, n=1):
        if isinstance(n, int):
            self = self >> n
        else:
            raise ValueError("Can only rshift by an int, was provided a {0}".format(type(n)))

    def invert(self):
        self.map = ~self.map

    def pack_bits_(self, bit_list):
        num_indices = int(math.ceil(len(bit_list) / float(self.int_size)))
        map = [self.dtype(0) for i in range(num_indices)]
        def lshift(map, n):     
            if isinstance(n, int):
                jump = n / self.int_size
                for i in range(len(map) - 1, jump - 1, -1):
                    map[i] = map[i - jump]
                for i in range(jump - 1, -1, -1):
                    map[i] = 0

                pos = n % self.int_size
                for i in range(jump):
                    map[i] =  map[i] << pos
                    
                for i in range(len(map) - 1, jump - 1, -1):
                    if i == 0:
                        map[i] = self.dtype(map[i] << pos)
                    else:
                        map[i] = self.dtype(map[i] << pos) | self.dtype(map[i - 1] >> self.int_size - pos)

                return map 
            else:
                raise ValueError("Can only lshift by an int, was provided a {0}".format(type(n)))

        for i in range(len(bit_list)):
            map[0] =  lshift(map, 1)[0] | bit_list[i]
        map.reverse()
        return np.array(map, dtype=self.dtype)

    # def __str__(self):
    #     ret = ''
    #     for i in range(self.num_ints - 1, -1, -1):
    #         ret += '(' + np.binary_repr(self.dtype(self.map[i])).rjust(self.int_size,'0') +')'
    #     return '[' + ret + ']'

    def __len__(self):
        return self.bytes_size() * self.int_bytes 

    def __and__(self, bm):
        if isinstance(bm, Bitmap) or isinstance(bm, int):
            if isinstance(bm, int):
                bm = Bitmap([1])
            
            i_min = min(bm.num_ints, self.num_ints)
            i_max = max(bm.num_ints, self.num_ints)
            new_list = [(bm.map[k] & self.map[k]) if k < i_min else self.dtype(0) for k in range(i_max)]

            return Bitmap(np.array(new_list, dtype=self.dtype), self.architecture)
        else:
            raise ValueError("Can only compare two bitmaps")
    
    def __or__(self, bm):
        if isinstance(bm, int):
            bm = Bitmap([bm], self.architecture)

        if isinstance(bm, Bitmap):
            i_min = min(bm.num_ints, self.num_ints)
            i_max = max(bm.num_ints, self.num_ints)
            new_list = [(bm.map[k] | self.map[k]) if k < i_min else self.map[k] if bm.num_ints < self.num_ints else bm.map[k]  for k in range(i_max)]

            return Bitmap(np.array(new_list, dtype=self.dtype), self.architecture)
        else:
            raise ValueError("Can only compare two bitmaps")

    def __xor__(self, bm):
        if isinstance(bm, int):
            bm = Bitmap([bm], self.architecture)

        if isinstance(bm, Bitmap):
            i = 0
            new_list =[]
            i_min = min(bm.num_ints, self.num_ints)
            i_max = max(bm.num_ints, self.num_ints)
            while i < i_min:
                new_list.append(bm.map[i] ^ self.map[i])
                i += 1
            while i < i_max:
                if bm.num_ints < self.num_ints:
                    new_list.append(self.map[i])
                else:
                    new_list.append(bm.map[i])
                i += 1
    
            return Bitmap(np.array(new_list, dtype=self.dtype), self.architecture)
        else:
            raise ValueError("Can only compare two bitmaps")

    def __lshift__(self, n):
        new_bm = Bitmap(self.map, self.architecture)
        if isinstance(n, int):
            jump = n / new_bm.int_size
            for i in range(len(new_bm.map) - 1, jump - 1, -1):
                new_bm.map[i] = new_bm.map[i - jump]
            for i in range(jump - 1, -1, -1):
                new_bm.map[i] = 0

            pos = n % new_bm.int_size
            for i in range(jump):
                new_bm.map[i] =  new_bm.map[i] << pos

            for i in range(len(new_bm.map) - 1, jump - 1, -1):
                if i == 0:
                    new_bm.map[i] = self.dtype(new_bm.map[i] << pos)
                else:
                    new_bm.map[i] = self.dtype(new_bm.map[i] << pos) | self.dtype(new_bm.map[i - 1] >> new_bm.int_size - pos)

            return new_bm
        else:
            raise ValueError("Can only lshift by an int, was provided a {0}".format(type(n)))

    def __rshift__(self, n):
        new_bm = Bitmap(self.map, self.architecture)
        if isinstance(n, int):
            jump = n / new_bm.int_size
            if jump is not 0:
                for i in range(len(new_bm.map) - jump):
                    new_bm.map[i] = new_bm.map[i+jump]
                for i in range(len(new_bm.map) - jump, len(new_bm.map)):
                    new_bm.map[i] = 0
            pos = n % new_bm.int_size

            for i in range(0, len(new_bm.map) - jump):
                if i is len(new_bm.map)-1:
                    new_bm.map[i] = self.dtype(new_bm.map[i] >> pos)
                else:
                    new_bm.map[i] = self.dtype(new_bm.map[i] >> pos) | self.dtype(new_bm.map[i + 1] << new_bm.int_size - pos)

            return new_bm
        else:
            raise ValueError("Can only lshift by an int, was provided a {0}".format(type(n)))
  
    def __invert__(self):
        return Bitmap(np.bitwise_not(self.map), self.architecture)

    def __mod__(self, n):
        return self.map[0] % n

# ***********************************************************************************************
# Bitboard
# ***********************************************************************************************
class Bitboard():
    def __init__(self, input, architecture = platform.architecture()[0], player = None):
        self.architecture = architecture.lower()
        if isinstance(input, int):
            self.n = input
            self.map_one = Bitmap([1 if i == self.n * self.n -1 else 0 for i in range(self.n * self.n)], self.architecture)
            self.player0 = Bitmap([0 for i in range(input * input)], self.architecture)
            self.player1 = Bitmap([0 for i in range(input * input)], self.architecture)
        elif isinstance(input, Bitboard):
            self.n = input.n
            self.map_one = Bitmap([1 if i == self.n * self.n -1 else 0 for i in range(self.n * self.n)], self.architecture)
            self.player0 = Bitmap(input.player0.map, input.architecture)
            self.player1 = Bitmap(input.player1.map, input.architecture)
        elif isinstance(input, list):
            self.n = len(input)
            self.map_one = Bitmap([1 if i == self.n * self.n -1 else 0 for i in range(self.n * self.n)], self.architecture)
            self.player0 = Bitmap([0 for i in range(self.n * self.n)], self.architecture)
            self.player1 = Bitmap([0 for i in range(self.n * self.n)], self.architecture)

            black_pieces = []
            white_pieces = []
            for row in range(self.n):
                for col in range(self.n):
                    if input[row][col] == 'B': black_pieces.append(self.n * row + col)
                    elif input[row][col] == 'W': white_pieces.append(self.n * row + col)

            for index in black_pieces:
                self.insert(1 if player == 'B' else 0, index)
            for index in white_pieces:
                self.insert(1 if player == 'W' else 0, index)
        else: raise ValueError("Invalid input for building a Bitboard")

    def insert(self, player, pos):
        if pos >= 0 and pos < self.n * self.n:
            if player:
                self.player1 = self.player1 | (self.map_one << pos)
            else:
                self.player0 = self.player0 | (self.map_one << pos)
        else:
            print 'invalid pos'

    def valid_pos(self, pos): return ((~(self.player0 | self.player1) >> pos) % 2)

    def player0_popcount(self):
        return self.player0.pop_count()

    def player1_popcount(self):
        return self.player1.pop_count()

    def board_popcount(self):
        return self.player0_popcount() + self.player1_popcount()

    def extract_moves(self, move_as_map):
        moves_as_indicies = list()
        pos = 0

        while not move_as_map.is_empty():
            if move_as_map % 2: moves_as_indicies.append(pos)
            pos += 1
            move_as_map = move_as_map >> 1
        return moves_as_indicies
    
    def get_actions(self, player_piece):
        all_moves = []
        empty_bitmap = Bitmap([0 for i in range(self.n * self.n)], self.architecture)
        empty_positions = ~(self.player1 | self.player0)

        player = self.player1 if player_piece else self.player0
        opponent = self.player0 if player_piece else self.player1

        # North
        moves = empty_bitmap
        candidates = opponent & (player << self.n)
        while not candidates.is_empty():
            moves = moves | empty_positions & (candidates << self.n)
            candidates = opponent & (candidates << self.n)
        if not moves.is_empty(): all_moves.append(('N', moves))
        
        # South
        moves = empty_bitmap
        candidates = opponent & (player >> self.n)
        while not candidates.is_empty():
            moves = moves | empty_positions & (candidates >> self.n)
            candidates = opponent & (candidates >> self.n)
        if not moves.is_empty(): all_moves.append(('S', moves))

        # East
        x = Bitmap([1 if i % self.n == self.n-1 else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)
        iters = 0

        moves = empty_bitmap
        candidates = opponent & (player >> 1)
        while iters < self.n and not candidates.is_empty():
            iters += 1
            x = x | (x >> 1)
            moves = moves | empty_positions & (candidates >> 1) & ~(x)
            candidates = opponent & (candidates >> 1)
        if not moves.is_empty(): all_moves.append(('E', moves))

        # West
        x = Bitmap([1 if i % self.n == 0 else 0 for i in range(self.n * self.n - 1, -1, -1)], self.architecture)
        iters = 0

        moves = empty_bitmap
        candidates = opponent & (player << 1)
        while iters < self.n and not candidates.is_empty():
            iters += 1
            x = x | (x << 1)
            moves = moves | empty_positions & (candidates << 1) & ~(x)
            candidates = opponent & (candidates << 1)
        if not moves.is_empty(): all_moves.append(('W', moves))
    
        # North West
        x = Bitmap([1 if (i % self.n == 0) or i < self.n else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)
        iters = 0

        moves = empty_bitmap
        candidates = opponent & (player << self.n + 1)
        while iters < self.n and not candidates.is_empty():
            iters += 1
            x = x | (x << self.n + 1)
            moves = moves | empty_positions & (candidates <<  self.n + 1) & ~(x)
            candidates = opponent & (candidates << self.n + 1)
        if not moves.is_empty(): all_moves.append(('NW', moves))

        # North East
        x = Bitmap([1 if (i % self.n == self.n-1) or i < self.n else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)
        iters = 0

        moves = empty_bitmap
        candidates = opponent & (player << self.n - 1)
        while iters < self.n and not candidates.is_empty():
            iters += 1
            x = x | (x << self.n - 1)
            moves = moves | empty_positions & (candidates << self.n - 1) & ~(x)
            candidates = opponent & (candidates << self.n - 1)
        if not moves.is_empty(): all_moves.append(('NE', moves))

        # South West
        x = Bitmap([1 if (i % self.n == 0) or i >= (self.n * self.n) - self.n else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)
        iters = 0

        moves = empty_bitmap
        candidates = opponent & (player >> self.n - 1)
        while iters < self.n and not candidates.is_empty():
            iters += 1
            x = x | (x >> self.n - 1)
            moves = moves | empty_positions & (candidates >> self.n - 1) & ~(x)
            candidates = opponent & (candidates >> self.n - 1)
        if not moves.is_empty(): all_moves.append(('SW', moves))
        
        # South East
        x = Bitmap([1 if (i % self.n == self.n-1) or i >= (self.n * self.n) - self.n else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)
        iters = 0

        moves = empty_bitmap
        candidates = opponent & (player >> self.n + 1)
        while iters < self.n and not candidates.is_empty():
            iters += 1
            x = x | (x >> self.n + 1)
            moves = moves | empty_positions & (candidates >> self.n + 1) & ~(x)
            candidates = opponent & (candidates >> self.n + 1)
        if not moves.is_empty(): all_moves.append(('SE', moves))

        separate_moves = []
        for move in all_moves:
            moves_as_indicies = self.extract_moves(move[1])

            for index in moves_as_indicies:
                if (index >= 0 and index < self.n * self.n 
                and self.should_add_index(separate_moves, index)):
                    separate_moves.append((move[0], index))

        return separate_moves


    def should_add_index(self, separate_moves, index):
        for i in range(len(separate_moves)):
            if separate_moves[i][1] == index: return False
        return True

    def make_move(self, player_piece, player_move):
        pos = player_move[1]
        empty_bitmap = Bitmap([0 for i in range(self.n * self.n)], self.architecture)
        original_move =  Bitmap([0 for i in range(self.n*self.n)], self.architecture) | (self.map_one << pos) 
        move = original_move
        captured_pieces = empty_bitmap
        
        if player_piece:
            player = self.player1
            opponent = self.player0
        else:
            player = self.player0
            opponent = self.player1
        
        player = player | move
        
        # North
        iters = 0
        move = move >> self.n
        while iters < self.n and not (move & opponent).is_empty():
            captured_pieces = captured_pieces | (move & opponent)
            move = move >> self.n
            iters += 1
        if not (move & player).is_empty():
            player = player | captured_pieces
            opponent = opponent ^ captured_pieces
        
        move = original_move
        captured_pieces = empty_bitmap

        # South
        iters = 0
        move = move << self.n
        while iters < self.n and not (move & opponent).is_empty():
            captured_pieces = captured_pieces | (move & opponent)
            move = move << self.n
            iters += 1
        if not (move & player).is_empty():
            player = player | captured_pieces
            opponent = opponent ^ captured_pieces
        
        move = original_move
        captured_pieces = empty_bitmap

        # East
        iters = 0
        move = move << 1
        x = Bitmap([1 if i % self.n == self.n-1 else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)

        while iters < self.n and not (move & opponent & (~x)).is_empty():
            captured_pieces = captured_pieces | (move & opponent)
            move = move << 1
            x = x | (x << 1)
            iters += 1
        if not (move & player).is_empty():
            player = player | captured_pieces
            opponent = opponent ^ captured_pieces
        
        move = original_move
        captured_pieces = empty_bitmap

        # West
        iters = 0
        move = move >> 1
        x = Bitmap([1 if i % self.n == 0 else 0 for i in range(self.n * self.n - 1, -1, -1)], self.architecture)

        while iters < self.n and not (move & opponent & (~x)).is_empty(): 
            captured_pieces = captured_pieces | (move & opponent)
            move = move >> 1
            x = x | (x >> 1)
            iters += 1
        if not (move & player).is_empty():
            player = player | captured_pieces
            opponent = opponent ^ captured_pieces
        
        move = original_move
        captured_pieces = empty_bitmap

        # North East
        iters = 0
        move = move >> self.n - 1
        x = Bitmap([1 if (i % self.n == self.n - 1) or i < self.n else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)

        while iters < self.n and not (move & opponent & (~x)).is_empty(): 
            captured_pieces = captured_pieces | (move & opponent)
            move = move >> self.n - 1
            x = x | (x >> self.n - 1)
            iters += 1
        if not (move & player).is_empty():
            player = player | captured_pieces
            opponent = opponent ^ captured_pieces
        
        move = original_move
        captured_pieces = empty_bitmap

        # North West
        iters = 0
        move = move >> self.n + 1
        x = Bitmap([1 if (i % self.n == 0) or i < self.n else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)

        while iters < self.n and not (move & opponent & (~x)).is_empty():
            captured_pieces = captured_pieces | (move & opponent)
            move = move >> self.n + 1
            x = x | (x >> self.n + 1)
            iters += 1
        if not (move & player).is_empty():
            player = player | captured_pieces
            opponent = opponent ^ captured_pieces
        
        move = original_move
        captured_pieces = empty_bitmap

        # South East
        iters = 0
        move = move << self.n + 1
        x = Bitmap([1 if (i % self.n == self.n-1) or i >= (self.n * self.n) - self.n else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)

        while iters < self.n and not (move & opponent & (~x)).is_empty(): 
            captured_pieces = captured_pieces | (move & opponent)
            move = move << self.n + 1
            x = x | (x << self.n + 1)
            iters += 1
        if not (move & player).is_empty():
            player = player | captured_pieces
            opponent = opponent ^ captured_pieces
        
        move = original_move
        captured_pieces = empty_bitmap

        # South West
        iters = 0
        move = move << self.n - 1
        x = Bitmap([1 if (i % self.n == 0) or i >= (self.n * self.n) - self.n else 0 for i in range(self.n * self.n-1, -1, -1)], self.architecture)
        
        while iters < self.n and not (move & opponent & (~x)).is_empty(): 
            captured_pieces = captured_pieces | (move & opponent)
            move = move << self.n - 1
            x = x | (x << self.n - 1)
            iters += 1
        if not (move & player).is_empty():
            player = player | captured_pieces
            opponent = opponent ^ captured_pieces
        
        # map boards back to original attributes
        if player_piece:
            self.player1 = player 
            self.player0 = opponent
        else:
            self.player0 = player
            self.player1 = opponent

# ***********************************************************************************************
# Heuristics
# ***********************************************************************************************
class Heuristic:
    def __init__(self, size, architecture='32bit'):
        self.n = size
        self.architecture = architecture
        self.heuristic_regions = self.generate_heuristic_regions()
       
        self.gamestate1_heuristic_weights = {'corner':5, 'diagonal': 3, 'hotspots': 5, 'backedge': 2, 'mid': 100, 'frontedge': -50, 'cornerguard': -20, 'supercornerguard': -50}
        self.gamestate2_heuristic_weights = {'corner':100, 'diagonal': 75, 'hotspots': 10, 'backedge': 5, 'mid': -1, 'frontedge': -2, 'cornerguard': -20, 'supercornerguard': -50}
        self.state = {'gamestate': 1}

    def diagonal(self):
        se_diag = [i for i in range(0, self.n * self.n + 1, self.n + 1)]
        sw_diag = [i for i in range(self.n - 1, (self.n * self.n + 1) - self.n, self.n - 1)]
        
        diags = Bitboard(self.n, self.architecture)
        for i in se_diag + sw_diag:
            diags.insert(0, i)
       
        return diags.player0

    def hotspots(self):
        hs = [2, 
              self.n - 3, 
              2*self.n, 
              3 * self.n - 1, 
              (self.n - 3)*self.n, 
              (self.n - 2)*self.n -1, 
              (self.n * self.n)- self.n + 2, 
              self.n * self.n - 3]
        hsedge = Bitboard(self.n, self.architecture)

        for i in hs:
            hsedge.insert(0, i)

        return hsedge.player0

    def backedge(self):
        len = self.n - 4
        N = [i for i in range(3, self.n - 3)]
        W = [i for i in range(3 * self.n, self.n + self.n * len, self.n)]
        E = [i for i in range(4 * self.n - 1, 2 * self.n -1 + self.n * len, self.n)]
        S = [i for i in range(self.n ** 2 - self.n + 3, self.n**2 - 3)]
        bedge = Bitboard(self.n, self.architecture)
        for i in (N + S + E + W):
            bedge.insert(0, i)

        return bedge.player0

    def frontedge(self):
        len = self.n - 4
        N = [i for i in range(2 + self.n, 2 * self.n - 2)]
        S = [i for i in range(self.n **2 - 2 * self.n + 2, self.n ** 2 - self.n - 2)]
        E = [i for i in range(3 *  self.n - 2, 3 * self.n -2 + self.n * len, self.n)]
        W = [i for i in range(2 * self.n + 1, 2 * self.n + self.n * len, self.n)]

        bedge = Bitboard(self.n, self.architecture)
        for i in (N + S + E + W):
            bedge.insert(0, i)

        return bedge.player0

    def corner(self):
        corner = [0, self.n-1, self.n**2 - self.n, self.n**2 - 1]
        
        bedge = Bitboard(self.n, self.architecture)
        for i in corner:
            bedge.insert(0, i)

        return bedge.player0

    def supercornerguard(self):
        scg = [self.n + 1, 2 * self.n - 2, self.n**2 - self.n - 2, self.n**2 - 2 * self.n + 1]
        supcg = Bitboard(self.n, self.architecture)
        for i in scg:
            supcg.insert(0, i)
    
        return supcg.player0

    def cornerguard(self):
        cg = [self.n, 1, 2 * self.n - 1, self.n - 2, 
              self.n**2 - 2*self.n,
              self.n **2 - self.n + 1, self.n**2 - 2, self.n**2 - self.n - 1]

        bedge = Bitboard(self.n, self.architecture)
        for i in cg:
            bedge.insert(0, i)

        return bedge.player0

    def generate_heuristic_regions(self):
        scg = self.supercornerguard()
        hs = self.hotspots()
        be = self.backedge()
        fe = self.frontedge()
        cg = self.cornerguard()
        cr = self.corner()
        mi = ~(be | fe | cg | cr | scg | hs)
        di = self.diagonal() & ~(cg) & ~(cr) & ~(scg)
        return {'backedge': be, 'frontedge': fe, 'cornerguard': cg, 'corner':cr, 'mid': mi, 'diagonal': di, 'supercornerguard': scg, 'hotspots': hs}

    def score_move(self, board, action):
        if self.state['gamestate'] is 1:
            if not (self.heuristic_regions['corner'] & board.map_one << action).is_empty():
                return self.gamestate1_heuristic_weights['corner']
            elif not (self.heuristic_regions['diagonal'] & board.map_one << action).is_empty():
                return self.gamestate1_heuristic_weights['diagonal']
            elif not (self.heuristic_regions['hotspots'] & board.map_one << action).is_empty():
                return self.gamestate1_heuristic_weights['hotspots']
            elif not (self.heuristic_regions['backedge'] & board.map_one << action).is_empty():
                return self.gamestate1_heuristic_weights['backedge']
            elif not (self.heuristic_regions['mid'] & board.map_one << action).is_empty(): 
                return self.gamestate1_heuristic_weights['mid']
            elif not (self.heuristic_regions['frontedge'] & board.map_one << action).is_empty():
                return self.gamestate1_heuristic_weights['frontedge']
            elif not (self.heuristic_regions['cornerguard'] & board.map_one << action).is_empty():
                return self.gamestate1_heuristic_weights['cornerguard']
            elif not (self.heuristic_regions['supercornerguard'] & board.map_one << action).is_empty():
                return self.gamestate1_heuristic_weights['supercornerguard']
            else:
                return 0
        if self.state['gamestate'] is 2:
            if not (self.heuristic_regions['corner'] & board.map_one << action).is_empty():
                return self.gamestate2_heuristic_weights['corner']
            if not (self.heuristic_regions['diagonal'] & board.map_one << action).is_empty():
                return self.gamestate2_heuristic_weights['diagonal']
            if not (self.heuristic_regions['hotspots'] & board.map_one << action).is_empty():
                return self.gamestate2_heuristic_weights['hotspots']
            if not (self.heuristic_regions['backedge'] & board.map_one << action).is_empty():
                return self.gamestate2_heuristic_weights['backedge']
            if not (self.heuristic_regions['mid'] & board.map_one << action).is_empty(): 
                return self.gamestate2_heuristic_weights['mid']
            if not (self.heuristic_regions['frontedge'] & board.map_one << action).is_empty():
                return self.gamestate2_heuristic_weights['frontedge']
            if not (self.heuristic_regions['cornerguard'] & board.map_one << action).is_empty():
                return self.gamestate2_heuristic_weights['cornerguard']
            if not (self.heuristic_regions['supercornerguard'] & board.map_one << action).is_empty():
                return self.gamestate2_heuristic_weights['supercornerguard']
            return 0
        else:
            print 'ERROR', 'gamestate', self.state['gamestate']

    def score_board(self, board, player):
        score = 0
        if self.state['gamestate'] is 1:
            if player:
                score += (board.player1 & self.heuristic_regions['mid']).pop_count()
            else:
                score += (board.player0 & self.heuristic_regions['mid']).pop_count()
            return score
        elif self.state['gamestate'] is 2:
            if player:
                score += (board.player1 & self.heuristic_regions['corner']).pop_count() * self.gamestate2_heuristic_weights['corner']
                score += (board.player1 & self.heuristic_regions['diagonal']).pop_count() * self.gamestate2_heuristic_weights['diagonal']
                score += (board.player1 & self.heuristic_regions['backedge']).pop_count() * self.gamestate2_heuristic_weights['backedge']
                score += (board.player1 & self.heuristic_regions['hotspots']).pop_count() * self.gamestate2_heuristic_weights['hotspots']
                score += (board.player1 & self.heuristic_regions['mid']).pop_count() #* self.gamestate2_heuristic_weights['mid']
                score += (board.player1 & self.heuristic_regions['cornerguard']).pop_count() #* self.gamestate2_heuristic_weights['cornerguard']
                score += (board.player1 & self.heuristic_regions['frontedge']).pop_count() #* self.gamestate2_heuristic_weights['frontedge']
                score += (board.player1 & self.heuristic_regions['supercornerguard']).pop_count() #* self.gamestate2_heuristic_weights['supercornerguard']
            else:
                score += (board.player0 & self.heuristic_regions['corner']).pop_count() * self.gamestate2_heuristic_weights['corner']
                score += (board.player0 & self.heuristic_regions['diagonal']).pop_count() * self.gamestate2_heuristic_weights['diagonal']
                score += (board.player0 & self.heuristic_regions['backedge']).pop_count() * self.gamestate2_heuristic_weights['backedge']
                score += (board.player1 & self.heuristic_regions['hotspots']).pop_count() * self.gamestate2_heuristic_weights['hotspots']
                score += (board.player0 & self.heuristic_regions['mid']).pop_count() #* self.gamestate2_heuristic_weights['mid']
                score += (board.player0 & self.heuristic_regions['cornerguard']).pop_count() #* self.gamestate2_heuristic_weights['cornerguard']
                score += (board.player0 & self.heuristic_regions['frontedge']).pop_count() #* self.gamestate2_heuristic_weights['frontedge']
                score += (board.player1 & self.heuristic_regions['supercornerguard']).pop_count() #* self.gamestate2_heuristic_weights['supercornerguard']
            return score 

    def evaluate_gamestate(self, board):
        if (~self.heuristic_regions['mid'] & (board.player0 | board.player1)).is_empty():
            self.state['gamestate'] = 1
        else:
            self.state['gamestate'] = 2

    def __getitem__(self, n):
        return self.heuristic_regions[n]

# ***********************************************************************************************
# Queue
# ***********************************************************************************************
class Binary_Heap_Queue:
    def __init__(self):
        self.queue = [] 
        self.__root__ = None
    
    def __str__(self):
        return str(self.queue)
    def __len__(self):
        return len(self.queue)

    def len(self):
        return self.queue.__len__()

    def root(self):
        return self.queue[0][0]

    def left(self, i):
        num = 2 * i + 1
        if num >=  self.queue.__len__():
            return None
        else:
            return self.queue[num][1]

    def right(self, i):
        num = 2 * i + 2
        if num >=  self.queue.__len__():
            return None
        else:
            return self.queue[num][1]

    def parent(self, i):
        num = (i - 1) // 2
        if num < 0:
            return None
        else:
            return self.queue[num][1]

    def reheapify(self, i):
        if i == 0:
            return
        if self.queue[i][1] > self.parent(i):
            x = self.queue[i]
            self.queue[i] = self.queue[(i - 1) // 2]
            self.queue[(i - 1) // 2] = x
            self.reheapify((i - 1) // 2)
        else: return

    def pop_reheapify(self, i):
        if i >= len(self.queue) - 1:
            return

        local_left = self.left(i)
        local_right = self.right(i)
        if local_left is not None and local_right is not None:
            if local_left > local_right:
                branch_max = (2 * i) + 1
            else:
                branch_max = (2 * i) + 2

            if self.queue[i][1] < self.queue[branch_max][1]:
                x = self.queue[i]
                self.queue[i] = self.queue[branch_max]
                self.queue[branch_max] = x
                self.pop_reheapify(branch_max)
        elif local_left is not None:
            if self.queue[i][1] < self.queue[(2 * i) + 1][1]:
                x = self.queue[i]
                self.queue[i] = self.queue[(2 * i) + 1]
                self.queue[(2 * i) + 1] = x
                self.pop_reheapify((2 * i) + 1)
            
    def pop(self):
        self.queue[0], self.queue[-1] = self.queue[-1], self.queue[0]
        ret = self.queue.pop()
        self.pop_reheapify(0)
        return ret
        

    def insert(self, obj, pri_key):
        self.queue.append((obj, pri_key))
        self.reheapify(self.queue.__len__() - 1)

        
# ***********************************************************************************************
# Minimax
# ***********************************************************************************************

MAX = 1
MIN = 0
INF = 1000000000

def MINIMAX(player, bitboard, heuristic, temp_recursive_depth=0, maxdepth=0, a=-INF, b=INF):
    temp_recursive_depth += 1

    # CASE: base case
    if temp_recursive_depth == maxdepth:
        return (None, heuristic.score_board(bitboard, player))
   
    # CASE: recursive case
    bestAction = None
    if player == MAX:
        maximum = -INF
        queue = Binary_Heap_Queue()
        
        for action in bitboard.get_actions(MAX):
            queue.insert(action, heuristic.score_move(bitboard, action[1]))

        for action in range(len(queue)):
            resulting_state = Bitboard(bitboard, bitboard.architecture)
            act = queue.pop()

            resulting_state.make_move(MAX, act[0])
            v = MINIMAX(MIN, resulting_state, heuristic, temp_recursive_depth, maxdepth, a, b)

            if v[0] is None:
                maximum = act[1]
                bestAction = act[0]
            elif v[1] > maximum:
                maximum = v[1]
                bestAction = act[0]

            a = max(a, maximum)
            if a >= b:
                break
        return (bestAction, int(maximum))
        
    elif player == MIN:
        minimum = +INF
        queue = Binary_Heap_Queue()

        for action in bitboard.get_actions(MIN):
            queue.insert(action, -heuristic.score_move(bitboard, action[1]))
        
        for action in range(len(queue)):
            resulting_state = Bitboard(bitboard, bitboard.architecture)
            act = queue.pop()

            resulting_state.make_move(MIN, act[0])
            v = MINIMAX(MAX, resulting_state, heuristic, temp_recursive_depth, maxdepth, a, b)

            if v[0] is None:
                minimum = act[1]
                bestAction = act[0]
            elif v[1] < minimum:
                minimum = v[1]
                bestAction = act[0]

            b = min(b, minimum)
            if b <= a:
                break
        return (bestAction, int(minimum))   

# *********************************************************************************************
# Get Move
# *********************************************************************************************
def get_move(board_size, board_state, turn, time_left=180000, opponent_time_left=180000):
    board = Bitboard(board_state, architecture="32bit", player=turn)
    heuristic = Heuristic(board_size, board.architecture)

    heuristic.evaluate_gamestate(board)
    numStates = 0
    if heuristic.state['gamestate'] == 1:
        depth = 2
    else:
        depth = 5
    if time_left < 45000:
        depth = 4
    elif time_left < 10000:
        depth = 2

    result = MINIMAX(MAX, board, heuristic, temp_recursive_depth=0, maxdepth=depth)

    if result[0] is None:
        return result[0]
    else:
        row = result[0][1] / board_size
        col = result[0][1] - (board_size * row)
        return [row, col]


# ********************************************************************************************
# Testing
# ********************************************************************************************

def should_flip_N(board_size, board_state, action, turn):
    col = action[1]
    for row in range(action[0] - 1, -1, -1):
        if board_state[row][col] == ' ': return False
        if board_state[row][col] == turn: return True
    return False

def should_flip_S(board_size, board_state, action, turn):
    col = action[1]
    for row in range(action[0] + 1, board_size, 1):
        if board_state[row][col] == ' ': return False
        if board_state[row][col] == turn: return True
    return False

def should_flip_E(board_size, board_state, action, turn):
    row = action[0]
    for col in range(action[1] + 1, board_size, 1):
        if board_state[row][col] == ' ': return False
        if board_state[row][col] == turn: return True
    return False

def should_flip_W(board_size, board_state, action, turn):
    row = action[0]
    for col in range(action[1] - 1, -1, -1):
        if board_state[row][col] == ' ': return False
        if board_state[row][col] == turn: return True
    return False

def should_flip_NE(board_size, board_state, action, turn):
    col = action[1]
    for row in range(action[0] - 1, -1, -1):
        col += 1
        if col >= board_size or board_state[row][col] == ' ': return False
        if board_state[row][col] == turn: return True
    return False

def should_flip_NW(board_size, board_state, action, turn):
    col = action[1]
    for row in range(action[0] - 1, -1, -1):
        col -= 1
        if col < 0 or board_state[row][col] == ' ': return False
        if board_state[row][col] == turn: return True
    return False

def should_flip_SE(board_size, board_state, action, turn):
    col = action[1]
    for row in range(action[0] + 1, board_size, 1):
        col += 1
        if col >= board_size or board_state[row][col] == ' ': return False
        if board_state[row][col] == turn: return True
    return False

def should_flip_SW(board_size, board_state, action, turn):
    col = action[1]
    for row in range(action[0] + 1, board_size, 1):
        col -= 1
        if col < 0 or board_state[row][col] == ' ': return False
        if board_state[row][col] == turn: return True
    return False



def apply_action(board_size, board_state, action, turn):
    board_state[action[0]][action[1]] = turn
    
    # flip N pieces
    if should_flip_N(board_size, board_state, action, turn):
        col = action[1]
        for row in range(action[0] - 1, -1, -1):
            if board_state[row][col] == turn: break
            board_state[row][col] = turn
    
    # flip S pieces
    if should_flip_S(board_size, board_state, action, turn):
        col = action[1]
        for row in range(action[0] + 1, board_size, 1):
            if board_state[row][col] == turn: break
            board_state[row][col] = turn
    
    # flip E pieces
    if should_flip_E(board_size, board_state, action, turn):
        row = action[0]
        for col in range(action[1] + 1, board_size, 1):
            if board_state[row][col] == turn: break
            board_state[row][col] = turn
    
    # flip W pieces
    if should_flip_W(board_size, board_state, action, turn):
        row = action[0]
        for col in range(action[1] - 1, -1, -1):
            if board_state[row][col] == turn: break
            board_state[row][col] = turn

    # flip NE pieces
    if should_flip_NE(board_size, board_state, action, turn):
        col = action[1]
        for row in range(action[0] - 1, -1, -1):
            col += 1
            if col >= board_size or board_state[row][col] == turn: break
            board_state[row][col] = turn
    
    # flip NW pieces
    if should_flip_NW(board_size, board_state, action, turn):
        col = action[1]
        for row in range(action[0] - 1, -1, -1):
            col -= 1
            if col < 0 or board_state[row][col] == turn: break
            board_state[row][col] = turn
    
    # flip SE pieces
    if should_flip_SE(board_size, board_state, action, turn):
        col = action[1]
        for row in range(action[0] + 1, board_size, 1):
            col += 1
            if col >= board_size or board_state[row][col] == turn: break
            board_state[row][col] = turn
    
    # flip SW pieces
    if should_flip_SW(board_size, board_state, action, turn):
        col = action[1]
        for row in range(action[0] + 1, board_size, 1):
            col -= 1
            if col < 0 or board_state[row][col] == turn: break
            board_state[row][col] = turn

    return board_state


def get_winner(board_state):
    black_score = 0
    white_score = 0
    for row in board_state:
        for col in row:
            if col == 'W':
                white_score += 1
            elif col == 'B':
                black_score += 1
    if black_score > white_score:
        winner = 'B'
    elif white_score > black_score:
        winner = 'W'
    else:
        winner = None
    return (winner, white_score, black_score)


def prepare_next_turn(turn, white_get_move, black_get_move):
    next_turn = 'W' if turn == 'B' else 'B'
    next_move_function = white_get_move if next_turn == 'W' else black_get_move
    return next_turn, next_move_function


def print_board(board_state):
    for row in board_state:
        print row
    

def simulate_game(board_state, board_size, white_get_move, black_get_move):
    player_blocked = False
    turn = 'B'
    get_move = black_get_move
    print_board(board_state)
    
    while True:
        ## GET ACTION ##
        next_action = get_move(board_size, board_state, turn, time_left=0, opponent_time_left=0)

        print "turn: ", turn, "next action: ", next_action
        # _ = raw_input()

        ## CHECK FOR BLOCKED PLAYER ##
        if next_action is None:
            if player_blocked:
                print "Both players blocked!"
                break
            else:
                player_blocked = True
                turn, get_move = prepare_next_turn(turn, white_get_move, black_get_move)
                continue
        else:
            player_blocked = False

        ## APPLY ACTION ##
        board_state = apply_action(board_size, board_state, next_action, turn)
        print_board(board_state)
        turn, get_move = prepare_next_turn(turn, white_get_move, black_get_move)

    winner, white_score, black_score = get_winner(board_state)

    print "Winner: ", winner
    print "White score: ", white_score
    print "Black score: ", black_score
    

if __name__ == "__main__":
    import MiniMax
    ## Replace with whatever board size you want to run on
    board_state = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
               	  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
               	  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
               	  [' ', ' ', ' ', 'W', 'B', ' ', ' ', ' '],
    			  [' ', ' ', ' ', 'B', 'W', ' ', ' ', ' '],
    			  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    			  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    			  [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]

    # board_state = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    #             [' ', ' ', 'B', ' ', ' ', 'B', 'B', 'B'],
    #             ['W', ' ', ' ', 'B', 'B', 'B', 'B', 'B'],
    #             ['W', ' ', 'W', 'B', 'B', 'B', 'B', 'B'],
    #             ['W', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    #             ['W', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    #             [' ', ' ', 'B', 'B', 'B', 'B', 'B', 'B'],
    #             [' ', 'B', 'B', 'B', 'B', 'B', 'B', 'B']]

    # board_state = [['B', ' ', 'W', 'W', 'W', 'W', 'W', ' '],
    #             [' ', 'B', 'W', 'W', 'B', 'B', 'B', 'B'],
    #             ['W', 'W', 'B', 'W', 'B', 'B', 'B', 'B'],
    #             ['W', 'W', 'W', 'B', 'B', 'B', 'B', 'B'],
    #             ['W', 'W', 'B', 'B', 'B', 'B', 'B', 'B'],
    #             ['W', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    #             ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    #             ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']]
    # board_state =  [[' ', ' ', ' ', ' ', ' ', ' '],
	# 				[' ', ' ', ' ', ' ', ' ', ' '],
	# 				[' ', ' ', 'W', 'B', ' ', ' '],
	# 				[' ', ' ', 'B', 'W', ' ', ' '],
	# 				[' ', ' ', ' ', ' ', ' ', ' '],
	# 				[' ', ' ', ' ', ' ', ' ', ' ']]
    board_size = 8

    ## Give these the get_move functions from whatever ais you want to test
    white_get_move = get_move
    black_get_move = MiniMax.get_move
    
    # white_get_move = get_move
    # black_get_move = MiniMax.get_move
    # print 'here', white_get_move, black_get_move
    # import sys
    # sys.exit()
    simulate_game(board_state, board_size, white_get_move, black_get_move)

    # import MiniMax

    # board = Bitboard(board_state, '32bit', player="W")
    # MiniMax.printboard(board)

    # for action in board.get_actions(1):
    #     print action
        # board.make_move(1, action)
        # MiniMax.printboard(board)
        # break