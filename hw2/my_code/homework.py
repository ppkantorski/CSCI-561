#=======================================================================#
# Homework #2: Checkers AI                                              #
# Course:     CSCI 561                                                  #
# Programmer: Patrick Kantorski                                         #
# Date:       03/22/2021                                                #
#=======================================================================#

import time
import os, sys
import random
#import heapq
#import json
from queue import Queue, PriorityQueue
from copy import deepcopy
from pprint import pprint

# Global variables
ROWS = 8
COLS = 8
WHITE = 'w'
BLACK = 'b'
STARTING_BOARD = [['.', 'b', '.', 'b', '.', 'b', '.', 'b'],\
                  ['b', '.', 'b', '.', 'b', '.', 'b', '.'],\
                  ['.', 'b', '.', 'b', '.', 'b', '.', 'b'],\
                  ['.', '.', '.', '.', '.', '.', '.', '.'],\
                  ['.', '.', '.', '.', '.', '.', '.', '.'],\
                  ['w', '.', 'w', '.', 'w', '.', 'w', '.'],\
                  ['.', 'w', '.', 'w', '.', 'w', '.', 'w'],\
                  ['w', '.', 'w', '.', 'w', '.', 'w', '.']]
SHORTEST_DURATION = 0.05
PLAY_DATA_CUTOFF_DURATION = 40

# Main function that will be called upon when grading / not using sys arguments
def main(test=False, test_time= None, test_depth=None, test_board=None):
    
    if test == True:
        time_in = time.time()
    
    input_data = load_input_data()
    game_type = input_data['game_type']
    ai_color = input_data['ai_color']
    if test_time == None:
        game_time = input_data['game_time']
    else:
        game_time = test_time
    
    if test_board == None:
        game_board = input_data['game_board']
    else:
        game_board = test_board
    #game_board = STARTING_BOARD
    
    if game_board == STARTING_BOARD:
        is_starting_board = True
    else:
        is_starting_board = False
    
    
    if game_time > PLAY_DATA_CUTOFF_DURATION:
        try:
            play_data = load_play_data()
            turn = int(play_data['turn'])
        except:
            play_data = {}
            turn = 1
    else:
        turn = None
    
    checkers_board = Game_Board()
    checkers_board.load_board(game_board)
    
    
    if game_time < SHORTEST_DURATION:
        all_moves = get_all_moves(checkers_board, WHITE, WHITE, shuffle=True)
        #pprint(all_moves)
        (value, new_checkers_board) = max(all_moves, key=len)
    else:
        
        
        if ai_color == 'WHITE':
            ai_color = WHITE
        elif ai_color == 'BLACK':
            ai_color = BLACK
        else:
            print("Invalid play color specified.")
            return
        
        if test_depth == None:
            if game_type == 'SINGLE':
                depth = 1
            else:
                depth = compute_depth(game_time, isAlphaBeta=True)
                print('Depth:', depth)
        else:
            depth = test_depth
        
        
        if is_starting_board:
            value, new_checkers_board = None, opening_move(checkers_board, ai_color)
            is_starting_board = False
            shuffle = False
        else:
            if turn != None and turn < 10:
                shuffle = False
            else:
                shuffle = True
            value, new_checkers_board = minimax(checkers_board, depth, ai_color, isAlphaBeta=True, shuffle=shuffle)
        
        
        # This should fix conditions for if minimax ever fails (SAFEGUARD)
        if new_checkers_board == None:
            all_moves = get_all_moves(checkers_board, ai_color, ai_color, shuffle)
            (value, new_checkers_board) = max(all_moves, key=len)
        
    
    
    # update checkers board with new checkers board
    checkers_board = new_checkers_board
    
    output_data = {'formatted_path':format_path(checkers_board.last_path)}
    write_output_data(output_data)
    
    if game_time > 30:
        play_data['turn'] = turn+1
        write_play_data(play_data)
    
    
    if test == True:
        game_time -= time.time()-time_in
        
        elapsed_time = time.time()-time_in
        print(f'Elapsed time: {elapsed_time}')
        print(output_data['formatted_path'])
        return elapsed_time
    
    
    return

# Compute the minimax depth based upon the current game time
def compute_depth(game_time, isAlphaBeta):
    
    calibration = load_calibration_data()
    
    MIN_NUMBER_OF_MOVES = 20
    
    if isAlphaBeta:
        
        for i in range(len(calibration)):
            depth = len(calibration)-i
            if game_time / calibration[i] > MIN_NUMBER_OF_MOVES:
                break
    else:
        if game_time > 80:
            depth = 5
        elif game_time > 5:
            depth = 4
        elif game_time > 1:
            depth = 3
        elif game_time > 0.1:
            depth = 2
        else:
            depth = 1
    
    return depth

# For testing the minimax algorithm and time estinmations
def test_ai(new_game=True, game_time=None):
    
    input_data = load_input_data()
    
    game_type = input_data['game_type']
    ai_color = input_data['ai_color']
    if game_time == None:
        game_time = input_data['game_time']
    game_board = input_data['game_board']
    
    if new_game == True:
        ai_color = 'WHITE'
        game_board = STARTING_BOARD
        is_starting_board = True
    else:
        if game_board == STARTING_BOARD:
            is_starting_board = True
        else:
            is_starting_board = False
    
    board = Game_Board()
    board.load_board(game_board)
    
    
    game_time_1 = game_time
    game_time_2 = game_time
    
    time_out_1 = float('inf')
    time_out_2 = float('inf')
    
    depth_1 = None
    depth_2 = None
    
    turn = 1
    while True:
        
        if ai_color == 'WHITE':
            #depth = 6
            
            time_in_1 = time.time()
            dummy_load_1 = load_input_data()
            
            if game_time_1 > 30:
                try:
                    play_data = load_play_data()
                    ai_turn = int(play_data['turn'])
                except:
                    play_data = {}
                    ai_turn = 1
            else:
                ai_turn = None
            
            if game_time_1 < 0.10:
                all_moves = get_all_moves(board, WHITE, WHITE, shuffle=True)
                #pprint(all_moves)
                try:
                    (value, new_board) = max(all_moves, key=len)
                    #pprint(new_board)
                except:
                    if board.white_left > board.black_left:
                        return 'WHITE'
                    elif board.white_left < board.black_left:
                        return 'BLACK'
                    else:
                        return 'DRAW'
            else:
                depth_1 = compute_depth(game_time_1, isAlphaBeta=True)
                #value, new_board = minimax_2(board, 5, max_player=WHITE, min_player=BLACK)
                if is_starting_board:
                    value, new_board = None, opening_move(board, WHITE)
                    is_starting_board = False
                else:
                    if ai_turn != None and ai_turn < 10:
                        shuffle = False
                    else:
                        shuffle = True
                    value, new_board = minimax(board, depth_1, WHITE, isAlphaBeta=True, shuffle=shuffle)
                
                
                # This should fix conditions when minimax fails, like only 2 bad moves left.
                if new_board == None:
                    all_moves = get_all_moves(board, WHITE, WHITE, shuffle)
                    #pprint(all_moves)
                    try:
                        (value, new_board) = max(all_moves, key=len)
                        #pprint(new_board)
                    except:
                        if board.white_left > board.black_left:
                            return 'WHITE'
                        elif board.white_left < board.black_left:
                            return 'BLACK'
                        else:
                            return 'DRAW'
            
            output_data_1 = {}
            output_data_1['formatted_path'] = format_path(new_board.last_path)
            
            write_output_data(output_data_1)
            
            if game_time_1 > 30:
                play_data['turn'] = ai_turn+1
                write_play_data(play_data)
            
            #time_out_1 = time.time()-time_in_1
            game_time_1 -= time.time()-time_in_1
            
            if turn > 1:
                delete_last_line(14)
            print(f'Turn {turn}')
            print(f'Elapsed time remaining for WHITE: {game_time_1}   DEPTH:{depth_1}')
            print(f'Elapsed time remaining for BLACK: {game_time_2}   DEPTH:{depth_2}')
            ai_color = 'BLACK'
            turn += 1
        else:
            #depth = 4
            
            time_in_2 = time.time()
            dummy_load_2 = load_input_data()
            
            if game_time_2 < 0.10:
                all_moves = get_all_moves(board, BLACK, BLACK, shuffle=True)
                #pprint(all_moves)
                try:
                    (value, new_board) = max(all_moves, key=len)
                    #pprint(new_board)
                except:
                    if board.white_left > board.black_left:
                        return 'WHITE'
                    elif board.white_left < board.black_left:
                        return 'BLACK'
                    else:
                        return 'DRAW'
            else:
                depth_2 = compute_depth(game_time_2, isAlphaBeta=False)
                value, new_board = minimax(board, depth_2, BLACK, isAlphaBeta=False, shuffle=True)
            
            # This should fix conditions when minimax fails, like only 2 bad moves left.
            if new_board == None:
                all_moves = get_all_moves(board, BLACK, BLACK, shuffle=True)
                #pprint(all_moves)
                try:
                    (value, new_board) = max(all_moves, key=len)
                    pprint(new_board)
                except:
                    if board.white_left > board.black_left:
                        return 'WHITE'
                    elif board.white_left < board.black_left:
                        return 'BLACK'
                    else:
                        return 'DRAW'
            
            output_data_2 = {}
            output_data_2['formatted_path'] = format_path(new_board.last_path)
            
            write_output_data(output_data_2, 'output2.txt')
            
            
            #time_out_2 = time.time()-time_in_2
            game_time_2 -= time.time()-time_in_2
            
            if turn > 1:
                delete_last_line(14)
            print(f'Turn {turn}')
            print(f'Elapsed time remaining for WHITE: {game_time_1}   DEPTH:{depth_1}')
            print(f'Elapsed time remaining for BLACK: {game_time_2}   DEPTH:{depth_2}')
            ai_color = 'WHITE'
            
        
        
        if game_time_2 <= 0:
            print("Game over.")
            winner = 'WHITE'
            print('White wins!')
            return winner
        
        if game_time_1 <= 0:
            print("Game over.")
            winner = 'BLACK'
            print('Black wins!')
            return winner
        
        try:
            if new_board.winner() != None:
                winner = new_board.winner()
                if winner == 'WHITE':
                    print('White wins!')
                else:
                    print('Black wins!')
                break
        except:
            print("Game over.")
            if board.white_left > board.black_left:
                winner = 'WHITE'
                print('White wins!')
            else:
                winner = 'BLACK'
                print('Black wins!')
            break
        
        if new_board == board:
            print("Game over.")
            if new_board.white_left > new_board.black_left:
                winner = 'WHITE'
                print('White wins!')
            else:
                winner = 'BLACK'
                print('Black wins!')
            break
        board = new_board
        
        # Print turn results
        print(new_board.last_path)
        pprint(format_path(new_board.last_path))
        print(ai_color, value)
        print_game_board(new_board.board)
    
    
    
    return winner



# Hard-coded opening move for first player with new_game board
def opening_move(board, player_color):
    
    if player_color == WHITE:
        start = (5, 2)
        end = (4, 3)
        path = [start, end]
        
        temp_board = deepcopy(board)
        temp_piece = temp_board.get_piece(start[0], start[1])
        
        new_board = make_move(temp_piece, end, path, temp_board, [])
        new_board.last_path = path
        #print('new_board.last_path', new_board.last_path)
    else:
        start = (2, 1)
        end = (3, 2)
        path = [start, end]
        
        temp_board = deepcopy(board)
        temp_piece = temp_board.get_piece(start[0], start[1])
        
        new_board = make_move(temp_piece, end, path, temp_board, [])
        new_board.last_path = path
        #print('new_board.last_path', new_board.last_path)
    
    
    return new_board


# Minimax function, returns the optimal path.  Alpha Beta pruning is optional
def minimax(position, depth, max_player, max_player_color=None, min_player_color=None, alpha=-float('inf'), beta=float('inf'), isAlphaBeta=True, shuffle=True):
    
    if max_player == WHITE:
        max_player_color = WHITE
        min_player_color = BLACK
    elif max_player == BLACK:
        max_player_color = BLACK
        min_player_color = WHITE
    
    if depth == 0 or position.winner() != None:
        return position.evaluate(max_player_color), position
    
    # Maximizing function
    if max_player:
        max_eval = float('-inf')
        best_move = None
        
        all_moves = get_all_moves(position, max_player_color, max_player_color, shuffle)
        # This should reduce path search time for a single forced jump
        if max_player != True and len(all_moves) == 1:
            priority, move = all_moves[0]
            return move.evaluate(max_player_color), move
        
        for (priority, move) in all_moves:
            evaluation = minimax(move, depth-1, False, max_player_color, min_player_color, alpha, beta, isAlphaBeta)[0]
            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
            
            alpha = max(evaluation, alpha)
            
            if isAlphaBeta and alpha >= beta:
                break
        
        
        return max_eval, best_move
    else: # Minimizing function
        min_eval = float('inf')
        best_move = None
        
        all_moves = get_all_moves(position, min_player_color, max_player_color, shuffle)
        for (priority, move) in all_moves:
            evaluation = minimax(move, depth-1, True, max_player_color, min_player_color, alpha, beta, isAlphaBeta)[0]
            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move
            beta = min(evaluation, beta)
            
            if isAlphaBeta and beta <= alpha:
               break
        
        return min_eval, best_move


# Make move, remove pieces, and update last path
def make_move(piece, move, path, board, taken_pieces):
    
    board.move_piece(piece, move[0], move[1])
    board.remove_pieces(taken_pieces)
    
    board.last_path = path
    
    return board


# Get all moves from a board provided a color
def get_all_moves(board, color, max_player, shuffle=True):
    FORWARD_DIRECTION = -1
    BACKWARD_DIRECTION = 1
    
    # Direction makes sure that the queue is ordered properly.
    if max_player == WHITE:
        if color == WHITE:
            direction = FORWARD_DIRECTION
        elif color == BLACK:
            direction = BACKWARD_DIRECTION
    elif max_player == BLACK:
        if color == WHITE:
            direction = BACKWARD_DIRECTION
        elif color == BLACK:
            direction = FORWARD_DIRECTION
    
    
    pop_list = []
    moves_dict = {}
    most_steps = 0
    max_pieces_taken = 0
    for piece in board.get_all_pieces(color):
        moves, num_pieces_taken = board.get_valid_moves(piece)
        #print('num_pieces_taken', num_pieces_taken)
        if num_pieces_taken == 0:
            pop_list.append((piece.row, piece.col))
        
        max_pieces_taken = max(max_pieces_taken, num_pieces_taken)
        moves_dict[(piece.row, piece.col)] = moves
    
    #print(max_pieces_taken)
    if max_pieces_taken > 0:
        for key in set(pop_list):
            moves_dict.pop(key)
    
    #pprint(moves_dict)
    
    #moves = []
    moves = PriorityQueue()
    keys = list(moves_dict.keys())
    if shuffle:
        random.shuffle(keys)
    for key in keys:
        
        valid_moves = moves_dict[key]
        move_indexes = list(range(len(valid_moves)))
        if shuffle:
            random.shuffle(move_indexes)
        
        for i in move_indexes:
            move = valid_moves[i]
            path = move['path']
            start = move['start']
            end = move['end']
            taken_pieces = move['taken_pieces']
            
            temp_board = deepcopy(board)
            temp_piece = temp_board.get_piece(start[0], start[1])
            
            new_board = make_move(temp_piece, end, path, temp_board, taken_pieces)
            evaluation = new_board.evaluate(color)* direction
            #moves.append((evaluation, new_board))
            #heapq.heappush(moves, (evaluation, new_board))
            
            moves.put((evaluation, new_board))
    
    moves = list(moves.queue)
    
    #pprint(moves)
    return moves

# For printing the game_board
def print_game_board(game_board):
    
    game_board = deepcopy(game_board)
    for i in range(len(game_board)):
        for j in range(len(game_board[i])):
            if game_board[i][j] != '.':
                game_board[i][j] = game_board[i][j].color
    
    pprint(game_board)
    
    return


# For returning the correctly formatted path
def format_path(path):
    
    path_strings = []
    if len(path) == 2:
        if abs(path[0][0] - path[1][0]) == 1 and abs(path[0][1] - path[1][1]) == 1:
            path_string = 'E '+chr(97+path[0][1])+str(8-path[0][0]) + ' ' + chr(97+path[1][1])+str(8-path[1][0])
        else:
            path_string = 'J '+chr(97+path[0][1])+str(8-path[0][0]) + ' ' + chr(97+path[1][1])+str(8-path[1][0])
        path_strings.append(path_string)
    else:
        for i in range(len(path)-1):
            path_string = 'J '+chr(97+path[i][1])+str(8-path[i][0]) + ' ' + chr(97+path[i+1][1])+str(8-path[i+1][0])
            path_strings.append(path_string)
    
    return path_strings


# Input: Takes in the file path for input.txt
# Output: Returns a dictionary containing the input parameters 
def load_input_data(file_path='input.txt'):

    # Read input file
    with open(file_path) as f:
        raw_data = f.read()
    split_data = raw_data.split('\n')
    
    # Sparce configurations from split_data
    game_type = split_data[0]
    ai_color = split_data[1]
    game_time = float(split_data[2])
    
    game_board = []
    for line in split_data[3:]:
        entry = []
        entry[:] = line
        game_board.append(entry)
    
    
    # Generate dictionary
    input_data = {}
    input_data['game_type'] = game_type
    input_data['ai_color'] = ai_color
    input_data['game_time'] = game_time
    input_data['game_board'] = game_board
    
    if game_board == STARTING_BOARD:
        input_data['is_starting_board'] = True
    else:
        input_data['is_starting_board'] = False
    
    
    return input_data


# For printing consecutive board changes.
def delete_last_line(num_lines=1):
    for i in range(num_lines):
        sys.stdout.write("\033[F") #back to previous line 
        sys.stdout.write("\033[K") #clear line


# For writing output data.
def write_output_data(output_data, file_path='output.txt'):
    
    formatted_path = output_data['formatted_path']
    output_string = ''
    
    for i in range(len(formatted_path)):
        output_string += formatted_path[i]
        if i != len(formatted_path)-1:
            output_string += '\n'
    
    with open(file_path, 'w') as f:
        f.write(output_string)
    
    return

# For loading calibration data
def load_calibration_data(file_path='calibration.txt'):
    #with open(file_path) as json_file:
    #    calibration = json.load(json_file)
    
    # Read input file
    with open(file_path) as f:
        raw_data = f.read()
    split_data = raw_data.split('\n')
    
    calibration = list(map(float, split_data))
    calibration.reverse()
    
    return calibration


# For loading turn data information.
def load_play_data(file_path='playdata.txt'):
    
    # Read input file
    with open(file_path) as f:
        raw_data = f.read()
    split_data = raw_data.split('\n')
    
    # Generate dictionary
    play_data = {}
    play_data['turn'] = split_data[0]
    
    
    return play_data

# For writing turn data information.  WARNING: this will consume resource time!
def write_play_data(play_data, file_path='playdata.txt'):
    
    turn = play_data['turn']
    output_string = str(turn)
    
    with open(file_path, 'w') as f:
        f.write(output_string)
    
    return


# Game Piece class that contains all information relavent to checker pieces
class Game_Piece:
    # Default initialization
    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        if color.isupper():
            self.king = True
        else:
            self.king = False
        self.became_king = False
    
    # Make a pawn king
    def make_king(self):
        self.color = self.color.upper()
        self.king = True
        self.became_king = True
    
    # Move game piece
    def move(self, row, col):
        if self.king == True:
            self.became_king = False
        self.row = row
        self.col = col
    
    # Object representation
    def __repr__(self):
        return str((self.row, self.col))

# Game Board class that contains all information relavent to the game
class Game_Board:
    def __init__(self):
        self.board = []
        self.last_path = []
        self.white_left = 0
        self.white_kings = 0
        self.white_kings_on_edge = 0
        self.white_pawns = 0
        self.white_pawns_in_back = 0
        self.white_pawns_in_middle = 0
        self.black_left = 0
        self.black_kings = 0
        self.black_kings_on_edge = 0
        self.black_pawns = 0
        self.black_pawns_in_back = 0
        self.black_pawns_in_middle = 0
    
    # Representation
    def __repr__(self):
        return str(self.last_path)
    
    # For allowing comparisons
    def __lt__(self, other):
        return
    def __eq__(self, other):
        return
    
    # Terminal condition
    def winner(self):
        if self.white_left <= 0:
            return 'BLACK'
        elif self.black_left <= 0:
            return 'WHITE'
        
        return None
    
    # For simple heuristic testing against black
    def evaluate_black(self):
        return self.black_left - self.white_left + (self.black_kings - self.white_kings)*0.5
    
    # Evaluation heuristic for minimax
    def evaluate(self, color):
        
        if color == BLACK:
            DIRECTION = -1
        elif color == WHITE:
            DIRECTION = 1
        else:
            print("Invalid color.")
            return
        
        # Heuristic weights
        BACK_ROW_PAWN = 4
        MIDDLE_PAWN = 2
        KING = 9
        PAWN = 5
        EDGE_KING_PLAYER = -3
        EDGE_KING_OPPONENT = 3
        
        evaluation = 0
        
        if self.white_pawns == 0 and self.black_pawns == 0:
            evaluation += (self.white_kings_on_edge) * EDGE_KING_PLAYER
            evaluation += self.black_kings_on_edge * EDGE_KING_OPPONENT
        
        
        
        evaluation += (self.white_pawns_in_back - self.black_pawns_in_back)*BACK_ROW_PAWN + \
                      (self.white_pawns_in_middle - self.black_pawns_in_middle)*MIDDLE_PAWN+ \
                      (self.white_kings - self.black_kings)*KING + \
                      (self.white_pawns - self.black_pawns)*PAWN
        
        
        return evaluation * DIRECTION
    
    def get_all_pieces(self, color):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != '.' and piece.color.lower() == color.lower():
                    pieces.append(piece)
        return pieces

    def move_piece(self, piece, row, col):
        self.board[piece.row][piece.col], self.board[row][col] = self.board[row][col], self.board[piece.row][piece.col]
        piece.move(row, col)
        
        
        if piece.king:
            if piece.row == 0 or piece.col == 0 or piece.row == ROWS-1 or piece.col == COLS-1:
                if piece.became_king == False:
                    if piece.color.lower() == WHITE:
                        self.white_kings_on_edge -= 1
                    elif piece.color.lower() == BLACK:
                        self.black_kings_on_edge -= 1
            
            if row == 0 or col == 0 or row == ROWS-1 or col == COLS-1:
                if piece.became_king == False:
                    if piece.color.lower() == WHITE:
                        self.white_kings_on_edge += 1
                    elif piece.color.lower() == BLACK:
                        self.black_kings_on_edge += 1
        
        
        if piece.king == False:
            # Make king conditions
            if row == ROWS-1 and piece.color == BLACK:
                self.black_kings += 1
                self.black_pawns -= 1
                #self.black_kings_on_edge += 1
                piece.make_king()
            elif row == 0 and piece.color == WHITE:
                self.white_kings += 1
                self.white_pawns -= 1 
                #self.white_kings_on_edge += 1
                piece.make_king()
            
            # Remove back row pawns from count
            if piece.row == 0 and piece.color == BLACK:
                self.black_pawns_in_back -= 1
            elif piece.row == ROWS-1 and piece.color == WHITE:
                self.white_pawns_in_back -= 1
            
            # Add to middle pawns count
            if (row == ROWS/2-1 or row == ROWS/2) and piece.row < 3 and piece.color == BLACK:
                self.black_pawns_in_middle += 1
            elif (row == ROWS/2-1 or row == ROWS/2) and piece.row > 4 and piece.color == WHITE:
                self.white_pawns_in_middle += 1
            
    
    def get_piece(self, row, col):
        return self.board[row][col]
    
    def load_board(self, game_board=None):
        
        if game_board == None:
            game_board = STARTING_BOARD
        
        
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                color = game_board[row][col]
                if color.lower() == WHITE:
                    self.white_left += 1
                    if color == WHITE.upper():
                        self.white_kings += 1
                        if row == 0 or row == ROWS-1 or col == 0 or col == COLS-1:
                            self.white_kings_on_edge += 1
                    else:
                        if row == ROWS-1:
                            self.white_pawns_in_back += 1
                        elif row == ROWS/2 or row == ROWS/2-1:
                            self.white_pawns_in_middle += 1
                    self.board[row].append(Game_Piece(row, col, color))
                elif color.lower() == BLACK:
                    self.black_left += 1
                    if color == BLACK.upper():
                        self.black_kings += 1
                        if row == 0 or row == ROWS-1 or col == 0 or col == COLS-1:
                            self.black_kings_on_edge += 1
                    else:
                        if row == 0:
                            self.black_pawns_in_back += 1
                        elif row == ROWS/2-1 or row == ROWS/2:
                            self.black_pawns_in_middle += 1
                        
                    self.board[row].append(Game_Piece(row, col, color))
                else:
                    self.board[row].append('.')
        
        self.white_pawns = self.white_left - self.white_kings
        self.black_pawns = self.black_left - self.black_kings
    
    
    def remove_pieces(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = '.'
            if piece != '.':
                if piece.color.lower() == WHITE:
                    self.white_left -= 1
                    if piece.king:
                        self.white_kings -= 1
                    else:
                        if piece.row == ROWS-1:
                            self.white_panws_in_back -= 1
                        elif piece.row == ROWS/2 or piece.row == ROWS/2-1:
                            self.white_pawns_in_middle -= 1
                            
                elif piece.color.lower() == BLACK:
                    self.black_left -= 1
                    if piece.king:
                        self.black_kings -= 1
                    else:
                        if piece.row == 0:
                            self.black_pawns_in_back -= 1
                        elif piece.row == ROWS/2-1 or piece.row == ROWS/2:
                            self.black_pawns_in_middle -= 1
        
        self.white_pawns = self.white_left - self.white_kings
        self.black_pawns = self.black_left - self.black_kings
    
    
    # Fix conditions were set to test if some time could be saved by "trimming the fat"
    def get_valid_moves(self, piece, FIX = True, FIX_2 = True):
        
        
        if piece.color.lower() == WHITE:
            step = -1
        elif piece.color.lower() == BLACK:
            step = 1
        
        max_pieces_taken = 0
        index = 0
        pop_list = []
        
        moves = []
        moves_set = set()
        
        start = [(piece.row, piece.col), [], [(piece.row, piece.col)]]
        
        positions_queue = Queue()
        positions_queue.put(start)
        
        while True:
            
            if positions_queue.empty():
                break
            
            position, taken_pieces, path = positions_queue.get()
            
            
            taken_position_1 = (position[0] + step, position[1]-1)
            if position[0] + step >= 0 and position[0] + step <= ROWS-1 and position[1]-1 >= 0 and position[1]-1 <= COLS-1:
                taken_piece_1 = self.get_piece(taken_position_1[0], taken_position_1[1])
                if taken_piece_1 != '.' and taken_piece_1.color.lower() != piece.color.lower():
                    if position[0] + 2*step >= 0 and position[0] + 2*step <= ROWS-1 and position[1]-2 >= 0 and position[1]-2 <= COLS-1:
                        taken_position_1 = (position[0] + 2*step, position[1]-2)
                        check_piece_1 = self.get_piece(taken_position_1[0], taken_position_1[1])
                    else:
                        check_piece_1 = -1
                else:
                    check_piece_1 = -1
            else:
                taken_piece_1 = -1
                check_piece_1 = -1
            
            taken_position_2 = (position[0] + step, position[1]+1)
            if position[0] + step >= 0 and position[0] + step <= ROWS-1 and position[1]+1 >= 0 and position[1]+1 <= COLS-1:
                taken_piece_2 = self.get_piece(taken_position_2[0], taken_position_2[1])
                if taken_piece_2 != '.' and taken_piece_2.color.lower() != piece.color.lower():
                    if position[0] + 2*step >= 0 and position[0] + 2*step <= ROWS-1 and position[1]+2 >= 0 and position[1]+2 <= COLS-1:
                        taken_position_2 = (position[0] + 2*step, position[1]+2)
                        check_piece_2 = self.get_piece(taken_position_2[0], taken_position_2[1])
                    else:
                        check_piece_2 = -1
                else:
                    check_piece_2 = -1
            else:
                taken_piece_2 = -1
                check_piece_2 = -1
            
            if piece.king:
                taken_position_3 = (position[0] - step, position[1]-1)
                if position[0] - step >= 0 and position[0] - step <= ROWS-1 and position[1]-1 >= 0 and position[1]-1 <= COLS-1:
                    taken_piece_3 = self.get_piece(taken_position_3[0], taken_position_3[1])
                    if taken_piece_3 != '.' and taken_piece_3.color.lower() != piece.color.lower():
                        if position[0] - 2*step >= 0 and position[0] - 2*step <= ROWS-1 and position[1]-2 >= 0 and position[1]-2 <= COLS-1:
                            taken_position_3 = (position[0] - 2*step, position[1]-2)
                            check_piece_3 = self.get_piece(taken_position_3[0], taken_position_3[1])
                        else:
                            check_piece_3 = -1
                    else:
                        check_piece_3 = -1
                else:
                    taken_piece_3 = -1
                    check_piece_3 = -1
                
                taken_position_4 = (position[0] - step, position[1]+1)
                if position[0] - step >= 0 and position[0] - step <= ROWS-1 and position[1]+1 >= 0 and position[1]+1 <= COLS-1:
                    taken_piece_4 = self.get_piece(taken_position_4[0], taken_position_4[1])
                    if taken_piece_4 != '.' and taken_piece_4.color.lower() != piece.color.lower():
                        if position[0] - 2*step >= 0 and position[0] - 2*step <= ROWS-1 and position[1]+2 >= 0 and position[1]+2 <= COLS-1:
                            taken_position_4 = (position[0] -2*step, position[1]+2)
                            check_piece_4 = self.get_piece(taken_position_4[0], taken_position_4[1])
                        else:
                            check_piece_4 = -1
                    else:
                        check_piece_4 = -1
                else:
                    taken_piece_4 = -1
                    check_piece_4 = -1
            
            #pprint(taken_pieces)
            
            if piece.king:
                if (taken_piece_1 == '.' or (taken_piece_1 in taken_pieces or (check_piece_1 != '.' and check_piece_1 != piece))) and\
                   (taken_piece_2 == '.' or (taken_piece_2 in taken_pieces or (check_piece_2 != '.' and check_piece_2 != piece))) and\
                   (taken_piece_3 == '.' or (taken_piece_3 in taken_pieces or (check_piece_3 != '.' and check_piece_3 != piece))) and\
                   (taken_piece_4 == '.' or (taken_piece_4 in taken_pieces or (check_piece_4 != '.' and check_piece_4 != piece))):
                    if len(taken_pieces) > 0:
                        if FIX:
                            set_length = len(moves_set)
                            moves_set.add((tuple(start[0]), tuple(position), tuple(taken_pieces)))
                            if set_length != len(moves_set):
                                num_taken_pieces = len(taken_pieces)
                                max_pieces_taken = max(max_pieces_taken, num_taken_pieces)
                                if (max_pieces_taken > 0 and num_taken_pieces != 0) or max_pieces_taken == 0 or FIX_2 == False:
                                    if max_pieces_taken == 0:
                                        pop_list.append(index)
                                    moves.append({'start':start[0], 'end':position, 'taken_pieces':taken_pieces, 'path':path})
                        elif FIX == False:
                            moves.append({'start':start[0], 'end':position, 'taken_pieces':taken_pieces, 'path':path})
                if (taken_piece_1 == '.' or taken_piece_1 == -1 or (taken_piece_1 in taken_pieces or (check_piece_1 != '.' and check_piece_1 != piece))):
                    if taken_piece_1 == '.' and len(taken_pieces) == 0:
                        if FIX:
                            set_length = len(moves_set)
                            moves_set.add((tuple(start[0]), tuple(taken_position_1), tuple(taken_pieces)))
                            if set_length != len(moves_set):
                                num_taken_pieces = len(taken_pieces)
                                max_pieces_taken = max(max_pieces_taken, num_taken_pieces)
                                if (max_pieces_taken > 0 and num_taken_pieces != 0) or max_pieces_taken == 0 or FIX_2 == False:
                                    if max_pieces_taken == 0:
                                        pop_list.append(index)
                                    moves.append({'start':start[0], 'end':taken_position_1, 'taken_pieces':taken_pieces, 'path':path+[taken_position_1]})
                        elif FIX == False:
                            moves.append({'start':start[0], 'end':taken_position_1, 'taken_pieces':taken_pieces, 'path':path+[taken_position_1]})
                if (taken_piece_2 == '.' or taken_piece_2 == -1 or (taken_piece_2 in taken_pieces or (check_piece_2 != '.' and check_piece_2 != piece))):
                    if taken_piece_2 == '.' and len(taken_pieces) == 0:
                        if FIX:
                            set_length = len(moves_set)
                            moves_set.add((tuple(start[0]), tuple(taken_position_2), tuple(taken_pieces)))
                            if set_length != len(moves_set):
                                num_taken_pieces = len(taken_pieces)
                                max_pieces_taken = max(max_pieces_taken, num_taken_pieces)
                                if (max_pieces_taken > 0 and num_taken_pieces != 0) or max_pieces_taken == 0 or FIX_2 == False:
                                    if max_pieces_taken == 0:
                                        pop_list.append(index)
                                    moves.append({'start':start[0], 'end':taken_position_2, 'taken_pieces':taken_pieces, 'path':path+[taken_position_2]})
                        elif FIX == False:
                            moves.append({'start':start[0], 'end':taken_position_2, 'taken_pieces':taken_pieces, 'path':path+[taken_position_2]})
                if (taken_piece_3 == '.' or taken_piece_3 == -1 or (taken_piece_3 in taken_pieces or (check_piece_3 != '.' and check_piece_3 != piece))):
                    if taken_piece_3 == '.' and len(taken_pieces) == 0:
                        if FIX:
                            set_length = len(moves_set)
                            moves_set.add((tuple(start[0]), tuple(taken_position_3), tuple(taken_pieces)))
                            if set_length != len(moves_set):
                                num_taken_pieces = len(taken_pieces)
                                max_pieces_taken = max(max_pieces_taken, num_taken_pieces)
                                if (max_pieces_taken > 0 and num_taken_pieces != 0) or max_pieces_taken == 0 or FIX_2 == False:
                                    if max_pieces_taken == 0:
                                        pop_list.append(index)
                                    moves.append({'start':start[0], 'end':taken_position_3, 'taken_pieces':taken_pieces, 'path':path+[taken_position_3]})
                        elif FIX == False:
                            moves.append({'start':start[0], 'end':taken_position_3, 'taken_pieces':taken_pieces, 'path':path+[taken_position_3]})
                if (taken_piece_4 == '.' or taken_piece_4 == -1 or (taken_piece_4 in taken_pieces or (check_piece_4 != '.' and check_piece_4 != piece))):
                    if taken_piece_4 == '.' and len(taken_pieces) == 0:
                        if FIX:
                            set_length = len(moves_set)
                            moves_set.add((tuple(start[0]), tuple(taken_position_4), tuple(taken_pieces)))
                            if set_length != len(moves_set):
                                num_taken_pieces = len(taken_pieces)
                                max_pieces_taken = max(max_pieces_taken, num_taken_pieces)
                                if (max_pieces_taken > 0 and num_taken_pieces != 0) or max_pieces_taken == 0 or FIX_2 == False:
                                    if max_pieces_taken == 0:
                                        pop_list.append(index)
                                    moves.append({'start':start[0], 'end':taken_position_4, 'taken_pieces':taken_pieces, 'path':path+[taken_position_4]})
                        elif FIX == False:
                            moves.append({'start':start[0], 'end':taken_position_4, 'taken_pieces':taken_pieces, 'path':path+[taken_position_4]})
            else:
                if (taken_piece_1 == '.' or taken_piece_1 == -1 or (taken_piece_1 in taken_pieces or (check_piece_1 != '.' and check_piece_1 != piece))) and\
                   (taken_piece_2 == '.' or taken_piece_2 == -1 or (taken_piece_2 in taken_pieces or (check_piece_2 != '.' and check_piece_2 != piece))):
                    if len(taken_pieces) > 0:
                        if FIX:
                            set_length = len(moves_set)
                            moves_set.add((tuple(start[0]), tuple(position), tuple(taken_pieces)))
                            if set_length != len(moves_set):
                                num_taken_pieces = len(taken_pieces)
                                max_pieces_taken = max(max_pieces_taken, num_taken_pieces)
                                if (max_pieces_taken > 0 and num_taken_pieces != 0) or max_pieces_taken == 0 or FIX_2 == False:
                                    if max_pieces_taken == 0:
                                        pop_list.append(index)
                                    moves.append({'start':start[0], 'end':position, 'taken_pieces':taken_pieces, 'path':path})
                        elif FIX == False:
                            moves.append({'start':start[0], 'end':position, 'taken_pieces':taken_pieces, 'path':path})
                if (taken_piece_1 == '.' or taken_piece_1 == -1 or (taken_piece_1 in taken_pieces or (check_piece_1 != '.' and check_piece_1 != piece))):
                    if taken_piece_1 == '.' and len(taken_pieces) == 0:
                        if FIX:
                            set_length = len(moves_set)
                            moves_set.add((tuple(start[0]), tuple(taken_position_1), tuple(taken_pieces)))
                            if set_length != len(moves_set):
                                num_taken_pieces = len(taken_pieces)
                                max_pieces_taken = max(max_pieces_taken, num_taken_pieces)
                                if (max_pieces_taken > 0 and num_taken_pieces != 0) or max_pieces_taken == 0 or FIX_2 == False:
                                    if max_pieces_taken == 0:
                                        pop_list.append(index)
                                    moves.append({'start':start[0], 'end':taken_position_1, 'taken_pieces':taken_pieces, 'path':path+[taken_position_1]})
                        elif FIX == False:
                            moves.append({'start':start[0], 'end':taken_position_1, 'taken_pieces':taken_pieces, 'path':path+[taken_position_1]})
                if (taken_piece_2 == '.' or taken_piece_2 == -1 or (taken_piece_2 in taken_pieces or (check_piece_2 != '.' and check_piece_2 != piece))):
                    if taken_piece_2 == '.' and len(taken_pieces) == 0:
                        if FIX:
                            set_length = len(moves_set)
                            moves_set.add((tuple(start[0]), tuple(taken_position_2), tuple(taken_pieces)))
                            if set_length != len(moves_set):
                                num_taken_pieces = len(taken_pieces)
                                max_pieces_taken = max(max_pieces_taken, num_taken_pieces)
                                if (max_pieces_taken > 0 and num_taken_pieces != 0) or max_pieces_taken == 0 or FIX_2 == False:
                                    if max_pieces_taken == 0:
                                        pop_list.append(index)
                                    moves.append({'start':start[0], 'end':taken_position_2, 'taken_pieces':taken_pieces, 'path':path+[taken_position_2]})
                        elif FIX == False:
                            moves.append({'start':start[0], 'end':taken_position_2, 'taken_pieces':taken_pieces, 'path':path+[taken_position_2]})
                
            if taken_piece_1 != '.' and taken_piece_1 not in taken_pieces and (check_piece_1 == '.' or check_piece_1 == piece):
                positions_queue.put([taken_position_1, taken_pieces+[taken_piece_1], path+[taken_position_1]])
            if taken_piece_2 != '.' and taken_piece_2 not in taken_pieces and (check_piece_2 == '.' or check_piece_2 == piece):
                positions_queue.put([taken_position_2, taken_pieces+[taken_piece_2], path+[taken_position_2]])
            if piece.king and taken_piece_3 != '.' and taken_piece_3 not in taken_pieces and (check_piece_3 == '.' or check_piece_3 == piece):
                positions_queue.put([taken_position_3, taken_pieces+[taken_piece_3], path+[taken_position_3]])
            if piece.king and taken_piece_4 != '.' and taken_piece_4 not in taken_pieces and (check_piece_4 == '.' or check_piece_4 == piece):
                positions_queue.put([taken_position_4, taken_pieces+[taken_piece_4], path+[taken_position_4]])
            
            index += 1
        # Filter impossible moves that come when a jump path exists.
        if FIX_2:
            if max_pieces_taken > 0:
                for index in sorted(pop_list, reverse=True):
                    #print(index, len(moves))
                    moves.pop(index)
        elif FIX_2 == False:
            max_pieces_taken = 0
            pop_list = []
            for i in range(len(moves)):
                #print(moves[i])
                if len(moves[i]['taken_pieces']) == 0:
                    #print('index:', i)
                    pop_list.append(i)
                #print(max_pieces_taken)
                max_pieces_taken = max(max_pieces_taken, len(moves[i]['taken_pieces']))
            
            #print(pop_list)
            #input()
            if max_pieces_taken > 0:
                for index in sorted(pop_list, reverse=True):
                    #print(index, len(moves))
                    moves.pop(index)
        #print('moves')
        #pprint(moves)
        
        #pprint(moves_set)
        
        return moves, max_pieces_taken


if __name__ == '__main__':
    
    #main()
    
    ### Comment this out for final turn in. ###
    if len(sys.argv) == 1:
        main()
    else:
        if sys.argv[1] == '-test':
            if len(sys.argv) > 2 and sys.argv[2] == '-new':
                new_game = True
            else:
                new_game = False
            black_wins, white_wins = 0, 0
            if len(sys.argv) > 3:
                try:
                    game_time = float(sys.argv[3])
                except:
                    game_time = None
            
            while True:
                winner = test_ai(new_game, game_time)
                #print(winner)
                #print(winner)
                if winner == 'BLACK':
                    black_wins += 1
                else:
                    white_wins += 1
                print(f'Record: BLACK {black_wins} | WHITE {white_wins}')
        elif sys.argv[1] == '-clean':
            play_data_file = 'playdata.txt'
            if os.path.exists(play_data_file):
                os.remove(play_data_file)
        else:
            main()