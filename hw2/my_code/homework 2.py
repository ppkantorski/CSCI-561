from copy import deepcopy
from pprint import pprint

ROWS, COLS = 8, 8
WHITE = 'w'
BLACK = 'b'


def main():
    
    input_data = load_input_data()
    
    play_type = input_data['play_type']
    play_color = input_data['play_color']
    play_time = input_data['play_time']
    game_board = input_data['game_board']
    
    print(play_color)
    board = Game_Board()
    
    pprint(convert_game_board(board.board))
    
    
    while True:
        if play_color == 'WHITE':
            turn = WHITE
            play_color = 'BLACK'
            value, new_board = minimax_white(board, 4, turn)
            #value, new_board, pc = minimax(board, 9, turn, True)
            
        else:
            turn = BLACK
            play_color = 'WHITE'
            value, new_board = minimax_black(board, 4, turn)
            #value, new_board, pc = minimax(board, 4, turn, True)
        
        #print(pc)
        print('value:', value)
        #value, new_board = minimax(board, 4, turn, True)
        
        print(new_board.last_sequence)
        pprint(convert_sequence(new_board.last_sequence))
        
        #game.ai_move(new_board)
        
        print(play_color)
        
        pprint(convert_game_board(new_board.board))
        #input()
        
        if new_board == board:
            print("Game over.")
            if new_board.white_left > new_board.black_left:
                print('White wins!')
            else:
                print('Black wins!')
            break
        board = new_board
    
        input()
    
    return


def convert_game_board(game_board):
    
    game_board = deepcopy(game_board)
    for i in range(len(game_board)):
        for j in range(len(game_board[i])):
            if game_board[i][j] != '.':
                game_board[i][j] = game_board[i][j].color
    
    return game_board


def convert_sequence(sequence):
    
    sequence_strings = []
    if len(sequence) == 2:
        sequence_string = 'E '+chr(97+sequence[0][1])+str(8-sequence[0][0]) + ' ' + chr(97+sequence[1][1])+str(8-sequence[1][0])
        sequence_strings.append(sequence_string)
    else:
        for i in range(len(sequence)-1):
            sequence_string = 'J '+chr(97+sequence[i][1])+str(8-sequence[i][0]) + ' ' + chr(97+sequence[i+1][1])+str(8-sequence[i+1][0])
            sequence_strings.append(sequence_string)
    
    return sequence_strings


class Game_Piece:

    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        if color.isupper():
            self.king = True
        else:
            self.king = False
        self.last_distance = 0

    def make_king(self):
        self.color = self.color.upper()
        self.king = True
    

    def move(self, row, col):
        self.row = row
        self.col = col
        #self.calc_pos()

    #def __repr__(self):
    #    if self.king == True:
    #        return str(self.color.upper())
    #    else:
    #        return str(self.color)

class Game_Board:
    def __init__(self):
        self.board = []
        self.last_sequence = []
        self.white_left = 0
        self.white_kings = 0
        self.black_left = 0
        self.black_kings = 0
        self.load_board()
    
    def winner(self):
        if self.white_left <= 0:
            return BLACK
        elif self.black_left <= 0:
            return WHITE
        
        return None
    
    def evaluate_black(self):
        #if color == WHITE:
        return self.black_left - self.white_left + (self.black_kings * 0.5 - self.white_kings * 0.5)
        #else:
        #    return self.white_left - self.black_left + (self.white_kings * 0.5 - self.black_kings * 0.5)

    def evaluate_white(self):
        #if color == WHITE:
        return self.white_left - self.black_left + (self.white_kings * 0.5 - self.black_kings * 0.5)
    
    def evaluate(self, color):
        #print(color)
        if color.lower() == BLACK:
            evaluation = (self.black_left - self.white_left) + (self.black_kings - self.white_kings)* 0.5
        else:
            evaluation = self.white_left - self.black_left + (self.white_kings - self.black_kings)* 0.5
        
        #print(evaluation)
        return evaluation

    def get_all_pieces(self, color):
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != '.' and piece.color.lower() == color.lower():
                    pieces.append(piece)
        return pieces

    def move(self, piece, row, col):
        self.board[piece.row][piece.col], self.board[row][col] = self.board[row][col], self.board[piece.row][piece.col]
        piece.move(row, col)

        if row == 0 or row == ROWS - 1 and piece.king == False:
            if piece.color == BLACK:
                self.black_kings += 1
            elif piece.color == WHITE:
                self.white_kings += 1 
            piece.make_king()

    def get_piece(self, row, col):
        return self.board[row][col]
    
    '''
    def create_board(self):
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if col % 2 == ((row +  1) % 2):
                    if row < 3:
                        self.board[row].append(Piece(row, col, BLACK))
                    elif row > 4:
                        self.board[row].append(Piece(row, col, WHITE))
                    else:
                        self.board[row].append(0)
                else:
                    self.board[row].append(0)
    '''
    
    def load_board(self):
        input_data = load_input_data()
        
        game_board = input_data['game_board']
        
        
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                color = game_board[row][col]
                if color.lower() == WHITE:
                    self.white_left += 1
                    if color == WHITE.upper():
                        self.white_kings += 1
                    #self.board[row][col] = Game_Piece(row, col, color)          ##. CONTINUE FROM HERE
                    self.board[row].append(Game_Piece(row, col, color))
                elif color.lower() == BLACK:
                    self.black_left += 1
                    if color == BLACK.upper():
                        self.black_kings += 1
                    #self.board[row][col] = Game_Piece(row, col, color)
                    self.board[row].append(Game_Piece(row, col, color))
                else:
                    self.board[row].append('.')
    
    
    def remove(self, pieces):
        for piece in pieces:
            self.board[piece.row][piece.col] = '.'
            if piece != '.':
                if piece.color.lower() == WHITE:
                    self.white_left -= 1
                elif piece.color.lower() == BLACK:
                    self.black_left -= 1
                
                if piece.king and piece.color.lower() == WHITE:
                    self.white_kings -= 1
                elif piece.king and piece.color.lower() == BLACK:
                    self.black_kings -= 1
    
    
    def get_valid_moves(self, piece):
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row
        
        if piece.king:
            moves.update(self.cross_left(row -1, max(row-3, -1), -1, piece.color, left, piece.king))
            moves.update(self.cross_right(row -1, max(row-3, -1), -1, piece.color, right, piece.king))
            moves.update(self.cross_left(row +1, min(row+3, ROWS), 1, piece.color, left, piece.king))
            moves.update(self.cross_right(row +1, min(row+3, ROWS), 1, piece.color, right, piece.king))
        else:
            if piece.color == WHITE:
                moves.update(self.cross_left(row -1, max(row-3, -1), -1, piece.color, left, piece.king))
                moves.update(self.cross_right(row -1, max(row-3, -1), -1, piece.color, right, piece.king))
            if piece.color == BLACK:
                moves.update(self.cross_left(row +1, min(row+3, ROWS), 1, piece.color, left, piece.king))
                moves.update(self.cross_right(row +1, min(row+3, ROWS), 1, piece.color, right, piece.king))
        
        # Isolate the moves with the most amount of steps
        if len(moves.values()) > 0:
            most_steps = len(max(list(moves.values()), key=len))
        else:
            most_steps = 0
        
        pop_list = []
        for move in moves.keys():
            #print(moves[move])
            if len(moves[move]) < most_steps:
                pop_list.append(move)
        
        #print(pop_list)
        for k in pop_list:
            moves.pop(k)
        #print(moves)
        #pprint(moves)
        return moves
    
    
    def filter_moves_by_length(self, moves, temp_moves):
        for key in temp_moves.keys():
            if key in moves.keys():
                if len(moves[key]) < len(temp_moves[key]):
                    moves[key] = temp_moves[key]
            else:
                moves[key] = temp_moves[key]
        return moves
    
    def filter_list_as_set(self, L):
        return [x for x in L if not any(set(x)<=set(y) for y in L if x is not y)]
    
    def get_valid_moves(self, piece):
        #os.system('clear')
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row
        
        temp_moves = {}
        if piece.king:
            moves = self.cross_left(row -1, max(row-3, -1), -1, piece, left)
            temp_moves = self.cross_right(row -1, max(row-3, -1), -1, piece, right)
            moves = self.filter_moves_by_length(moves, temp_moves)
            temp_moves = self.cross_left(row +1, min(row+3, ROWS), 1, piece, left)
            moves = self.filter_moves_by_length(moves, temp_moves)
            temp_moves = self.cross_right(row +1, min(row+3, ROWS), 1, piece, right)
            moves = self.filter_moves_by_length(moves, temp_moves)
            
        else:
            if piece.color == WHITE:
                moves.update(self.cross_left(row -1, max(row-3, -1), -1, piece, left))
                moves.update(self.cross_right(row -1, max(row-3, -1), -1, piece, right))
            if piece.color == BLACK:
                moves.update(self.cross_left(row +1, min(row+3, ROWS), 1, piece, left))
                moves.update(self.cross_right(row +1, min(row+3, ROWS), 1, piece, right))
        
        # For handling returns to original position and beyond
        if piece.king:
            pieces = []
            temp_key = (-1,-1)
            continue_condition = False
            for key in moves.keys():
                if len(moves[key]) == 3:
                    if temp_key == (-1,-1) or ((temp_key[0] == key[0] and abs(temp_key[1] - key[1]) == 4) or\
                        (temp_key[1] == key[1] and abs(temp_key[0] - key[0]) == 4)):
                        pieces += list(moves[key])
                    #print(pieces)
                    
                    temp_key = key
            pieces = set(pieces)
            
            if len(pieces) > 3:
                moves[(piece.row, piece.col)] = list(pieces)
                combining_condition = True
            else:
                combining_condition = False
        else:
            combining_condition = False
        
        moves_set = set(map(tuple, moves.values()))
        moves_set = self.filter_list_as_set(moves_set)
        
        if len(moves.values()) > 0:
            most_steps = len(max(list(moves.values()), key=len))
        else:
            most_steps = 0
        
        remove_moves = []
        for move in moves.keys():
            if most_steps > 0 and tuple(moves[move]) not in moves_set:
                remove_moves.append(move)
        
        for k in remove_moves:
            moves.pop(k)
        
        # For when additional moves after returning to original position are available
        if combining_condition:
            if len(moves.keys()) > 1:
                for move in moves.keys():
                    if move != (piece.row, piece.col):
                        moves[move] += list(pieces)
                moves.pop((piece.row, piece.col))
            pprint(moves)
        
        return moves

    def cross_left(self, start, stop, step, piece, left, taken_pieces=[]):
        moves = {}
        last = []
        for i in range(start, stop, step):
            if left < 0:
                break
            
            current = self.board[i][left]
            if current == '.':
                if taken_pieces and not last:
                    break
                elif taken_pieces:
                    moves[(i, left)] = last + taken_pieces
                else:
                    moves[(i, left)] = last
                
                if last:
                    if step == -1:
                        row = max(i-3, -1)
                    else:
                        row = min(i+3, ROWS)
                    
                    if piece.king and step == 1:
                        moves.update(self.cross_left(i-1, max(i-3, -1), -1, piece, left-1, taken_pieces=taken_pieces+last))
                    elif piece.king and step == -1:
                        moves.update(self.cross_left(i+1, min(i+3, ROWS), 1, piece, left-1, taken_pieces=taken_pieces+last))
                    
                    moves.update(self.cross_left(i+step, row, step, piece, left-1, taken_pieces=taken_pieces+last))
                    moves.update(self.cross_right(i+step, row, step, piece, left+1, taken_pieces=taken_pieces+last))
                break
            elif current.color.lower() == piece.color.lower():
                break
            else:
                last = [current]
            
            left -= 1
        
        #print('taken_pieces:', taken_pieces)
        #print('last:', last)
        return moves

    def cross_right(self, start, stop, step, piece, right, taken_pieces=[]):
        moves = {}
        last = []
        for i in range(start, stop, step):
            if right >= COLS:
                break
            
            current = self.board[i][right]
            if current == '.':
                if taken_pieces and not last:
                    break
                elif taken_pieces:
                    moves[(i, right)] = last + taken_pieces
                else:
                    moves[(i, right)] = last
                
                if last:
                    if step == -1:
                        row = max(i-3, -1)
                    else:
                        row = min(i+3, ROWS)
                    
                    if piece.king and step == 1:
                        moves.update(self.cross_right(i-1, max(i-3, -1), -1, piece, right+1, taken_pieces=taken_pieces+last))
                    elif piece.king and step == -1:
                        moves.update(self.cross_right(i+1, min(i+3, ROWS), 1, piece, right+1, taken_pieces=taken_pieces+last))
                    
                    moves.update(self.cross_left(i+step, row, step, piece, right-1, taken_pieces=taken_pieces+last))
                    moves.update(self.cross_right(i+step, row, step, piece, right+1, taken_pieces=taken_pieces+last))
                break
            elif current.color.lower() == piece.color.lower():
                break
            else:
                last = [current]
                #last = []

            right += 1
        
        
        #print('taken_pieces:', taken_pieces)
        #for val in last:
        #    print('last:', val.row, val.col)
        return moves

def minimax_white(position, depth, max_player):
    if depth == 0 or position.winner() != None:
        return position.evaluate_white(), position
    
    if max_player:
        maxEval = float('-inf')
        best_move = None
        
        all_moves = get_all_moves(position, WHITE)
        #pprint(all_moves)
        for move in all_moves:
            evaluation = minimax_white(move, depth-1, False)[0]
            maxEval = max(maxEval, evaluation)
            if maxEval == evaluation:
                best_move = move
        
        return maxEval, best_move
    else:
        minEval = float('inf')
        best_move = None
        all_moves = get_all_moves(position, BLACK)
        for move in all_moves:
            evaluation = minimax_white(move, depth-1, True)[0]
            minEval = min(minEval, evaluation)
            if minEval == evaluation:
                best_move = move
        
        return minEval, best_move

def minimax_black(position, depth, max_player):
    if depth == 0 or position.winner() != None:
        return position.evaluate_black(), position
    
    if max_player:
        maxEval = float('-inf')
        best_move = None
        all_moves = get_all_moves(position, BLACK)
        #pprint(all_moves)
        for move in all_moves:
            evaluation = minimax_black(move, depth-1, False)[0]
            maxEval = max(maxEval, evaluation)
            if maxEval == evaluation:
                best_move = move
        
        return maxEval, best_move
    else:
        minEval = float('inf')
        best_move = None
        all_moves = get_all_moves(position, WHITE)
        for move in all_moves:
            evaluation = minimax_black(move, depth-1, True)[0]
            minEval = min(minEval, evaluation)
            if minEval == evaluation:
                best_move = move
        
        return minEval, best_move

def minimax(position, depth, player_color, is_max_player, A = float('-inf'), B = float('inf'), alpha_beta = False):
    
    if player_color == WHITE:
        opponent_color = BLACK
    else:
        opponent_color = WHITE
    
    if depth == 0 or position.winner() != None or (A >= B and alpha_beta):
        #print(player_color)
        return position.evaluate(player_color), position, player_color
    
    if is_max_player:
        #A = float('-inf')
        best_move = None
        for move in get_all_moves(position, player_color):
            #if alpha_beta:
            #    if A >= B:
            #        #print('PRUNED')
            #        return position.evaluate(player_color), position, player_color
            evaluation = minimax(move, depth-1, player_color, False, A, B, alpha_beta)[0]
            A = max(A, evaluation)
                #A = max(A, maxEval)
            if A == evaluation:
                best_move = move
                #best_move.last_sequence = sequence
        #print("maxEval:", maxEval)
        return A, best_move, player_color
    else:
        #B = float('inf')
        best_move = None
        for move in get_all_moves(position, opponent_color):
            #if alpha_beta:
            #    if B <= A:
            #        #print('PRUNED')
            #        return position.evaluate(player_color), position, player_color
            evaluation = minimax(move, depth-1, opponent_color, True, A, B, alpha_beta)[0]
            B = min(B, evaluation)
                #B = min(B, minEval)
            if B == evaluation:
                best_move = move
                #best_move.last_sequence = sequence
        #print("minEval:", minEval)
        return B, best_move, player_color


def simulate_move(piece, move, board, taken_piece):
    board.move(piece, move[0], move[1])
    if taken_piece:
        board.remove(taken_piece)

    return board


def sort_pieces_by_last_distance(piece):
    # For handling a return to initial position
    #if piece.last_distance == 0:
    #    piece.last_distance = float('inf')
    return piece.last_distance


def get_all_moves(board, color):
    
    
    move_pieces = []
    
    most_steps = 0
    for piece in board.get_all_pieces(color):
        valid_moves = board.get_valid_moves(piece)
        
        for move in valid_moves.keys():
            most_steps = max(most_steps, len(valid_moves[move]))
    
    for piece in board.get_all_pieces(color):
        valid_moves = board.get_valid_moves(piece)
        
        for move in valid_moves.keys():
            if len(valid_moves[move]) >= 1 or len(valid_moves[move]) == most_steps:
                move_pieces.append(piece)
    
    #print('move_pieces:', move_pieces)
    moves = []
    #move_sequence = []
    for piece in move_pieces:
        valid_moves = board.get_valid_moves(piece)
        
        #print(valid_moves)
        
        #print(valid_moves.keys())
        #print('move_sequence:', move_sequence)
        #pprint(valid_moves)
        #print(valid_moves.items())
        for move, taken_pieces in valid_moves.items():
            #print('(move, taken_piece):', (move, taken_piece))
            
            if len(taken_pieces) > 0:
                
                for taken_piece in taken_pieces:
                    taken_piece.last_distance = (abs(taken_piece.row - move[0])**2 + abs(taken_piece.col - move[1])**2)**.5
                taken_pieces = sorted(taken_pieces, key=sort_pieces_by_last_distance)
                #taken_pieces.reverse()
                #pprint(taken_pieces)
                #taken_pieces.reverse()
                # Populate the last distance values for each piece
                #temp = move
                #for taken_piece in taken_pieces:
                #    taken_piece.last_distance = (abs(taken_piece.row - move[0])**2 + abs(taken_piece.col - move[1])**2)**.5
                #    temp = (taken_piece.row+(taken_piece.row - temp[0]), taken_piece.col+(taken_piece.col - temp[1]))
                
                # Sort taken_pieces pieces by last distance from previous
                #taken_pieces = sorted(taken_pieces, key=sort_pieces_by_last_distance)
                
                loop = True
                while loop:
                    taken_positions = []
                    #print(move, ' ', end='')
                    taken_position = move
                    pop_index = -1
                    for taken_piece in taken_pieces:
                        print((taken_piece.row, taken_piece.col), ' ', end='')
                        taken_position = (taken_piece.row-taken_position[0] + taken_piece.row, taken_piece.col-taken_position[1] + taken_piece.col)
                        #print(taken_position, ' ', end='')
                        #print(taken_piece.row, taken_piece.col, ' ', end='')
                        if taken_position[0] < 0 or taken_position[0] > 7 or taken_position[1] < 0 or taken_position[1] > 7:
                            pop_index = i
                            break
                        taken_positions.append(taken_position)
                    
                    if pop_index != -1:
                        taken_positions.pop(pop_index)
                    else:
                        break
                
                print('\n')
                taken_positions.reverse()
                move_sequence = taken_positions+[move]
            else:
                #print(piece.row, piece.col)
                move_sequence = [(piece.row, piece.col), move]
            
            #print(valid_moves[(move, taken_pieces)])
            #draw_moves(game, board, piece)
            temp_board = deepcopy(board)
            temp_piece = temp_board.get_piece(piece.row, piece.col)
            #print(temp_piece.row, temp_piece.col, move, temp_board, game)
            #for val in taken_pieces:
            #    print(valid_moves[(move,taken_pieces)])
            new_board = simulate_move(temp_piece, move, temp_board, taken_pieces)
            new_board.last_sequence = move_sequence
            
            moves.append(new_board)
    
    #for move in moves:
    #    print('move.last_sequence', move.last_sequence)
    return moves




# Input: Takes in the file path for input.txt
# Output: Returns a dictionary containing the input parameters 
def load_input_data(file_path='input.txt'):

    # Read input file
    with open(file_path) as f:
        raw_data = f.read()
    split_data = raw_data.split('\n')
    
    # Sparce configurations from split_data
    play_type = split_data[0]
    play_color = split_data[1]
    play_time = float(split_data[2])
    
    game_board = []
    for line in split_data[3:]:
        entry = []
        entry[:] = line
        game_board.append(entry)
    
    
    # Generate dictionary
    input_data = {}
    input_data['play_type'] = play_type
    input_data['play_color'] = play_color
    input_data['play_time'] = play_time
    input_data['game_board'] = game_board

    return input_data




if __name__ == '__main__':
    main()