import time
import copy
from pprint import pprint

def main():
    N = 8
    inputFile = open('input.txt', 'r')
    
    
    input_data = load_input_data()
    
    play_type = input_data['play_type']
    play_color = input_data['play_color']
    game_board = input_data['game_board']
    
    #Read input File
    mainPlayer = play_color
    algorithm = 'ALPHABETA'
    globals()['maxDepth'] = 3
    
    initialBoard = game_board
    
    #Initial state of game
    checkerObj = GameState(initialBoard,mainPlayer)
    
    #Start the algorithm
    nextMove = playCheckers(checkerObj,globals()['MIN_VALUE'],globals()['MAX_VALUE'],0,mainPlayer,False,False,algorithm == 'ALPHABETA')
    
    #Move calculation
    
    print(globals()['result']["move"])
    output_move = globals()['result']["move"]
    output_move = chr(97+output_move[1])+str(output_move[0]+1) + ' ' + chr(97+output_move[3])+str(output_move[2]+1)
    print(output_move)
    
    #Write the output file
    result = globals()['result']
    output_file = open("output.txt", "w")
    output_file.write(output_move+"\n")
    output_file.write(str(result["myopicUtility"])+"\n")
    output_file.write(str(result["farsightedUtility"])+"\n")
    output_file.write(str(result["nodes"]))
    output_file.close()
    
    return



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
    pprint(game_board)
    #input()
    
    
    # Generate dictionary
    input_data = {}
    input_data['play_type'] = play_type
    input_data['play_color'] = play_color
    input_data['game_board'] = game_board
    
    return input_data


# Input: data = dictionary containing each sequence to goal,
#               each sequence contains a list of coordinate dictionaries.
def write_output_data(output_data, file_path='output.txt'):
    
    output_string = ''
    for i in range(len(output_data)):
        data = output_data[i]
        for j in range(len(data)):
            output_string += data[j]
            if j != len(data)-1:
                output_string += ' '
        if i != len(output_data)-1:
            output_string += '\n'
    
    with open(file_path, 'w') as f:
        f.write(output_string)
    
    return output_string



def getMaxValue(checkers,alpha,beta,currDepth,player,isPass1,isPass2,isAlphaBeta):
    playerUtility = {"utility":globals()['MIN_VALUE'], "farsightedUtility":0}
    
    result = globals()['result']
    result["nodes"] += 1
    
    if currDepth >= globals()['maxDepth'] or isTerminal(checkers):
        playerUtility["utility"] = checkers.calculateUtility()
        playerUtility["farsightedUtility"] = playerUtility["utility"]
        return playerUtility

    moves = getAllMoves(checkers,player)
    
    if len(moves) == 0:
        if isPass1:
            if isPass2:
                playerUtility["utility"] = checkers.calculateUtility()
                playerUtility["farsightedUtility"] = playerUtility["utility"]
                return playerUtility
            else:
                isPass2 = True
        isPass1 = True
        return getMinValue(checkers,alpha,beta,currDepth+1,reverse(player),isPass1,isPass2,isAlphaBeta)
    
    for m in moves:
        new_checkers = executeMove(checkers,m)
        minVal = getMinValue(new_checkers,alpha,beta,currDepth+1,reverse(player),False,False,isAlphaBeta)
    
        if minVal["utility"] > playerUtility["utility"]:
            playerUtility["utility"] = minVal["utility"]
            playerUtility["farsightedUtility"] = minVal["farsightedUtility"]
        
        if isAlphaBeta:
            if playerUtility["utility"] >= beta:
                return playerUtility
        
        alpha = max(alpha, playerUtility["utility"])
    
    return playerUtility

def getMinValue(checkers,alpha,beta,currDepth,player,isPass1,isPass2,isAlphaBeta):
    result = globals()['result']
    
    playerUtility = {"utility":globals()['MAX_VALUE'], "farsightedUtility":0}
    
    result["nodes"] += 1
    
    if currDepth >= globals()['maxDepth'] or isTerminal(checkers):
        playerUtility["utility"] = checkers.calculateUtility()
        playerUtility["farsightedUtility"] = playerUtility["utility"]
        return playerUtility

    moves = getAllMoves(checkers,player)
    
    if len(moves) == 0:
        if isPass1:
            if isPass2:
                playerUtility["utility"] = checkers.calculateUtility()
                playerUtility["farsightedUtility"] = playerUtility["utility"]
                return playerUtility
            else:
                isPass2 = True
        isPass1 = True
        return getMaxValue(checkers,alpha,beta,currDepth+1,reverse(player),isPass1,isPass2,isAlphaBeta)
    
    for m in moves:
        new_checkers = executeMove(checkers,m)
        maxVal = getMaxValue(new_checkers,alpha,beta,currDepth+1,reverse(player),False,False,isAlphaBeta)
    
        if maxVal["utility"] < playerUtility["utility"]:
            playerUtility["utility"] = maxVal["utility"]
            playerUtility["farsightedUtility"] = maxVal["farsightedUtility"]
        
        if isAlphaBeta:
            if playerUtility["utility"] <= alpha:
                return playerUtility
        
            beta = min(beta, playerUtility["utility"])
    
    return playerUtility

def playCheckers(checkers,alpha,beta,currDepth,player,isPass1,isPass2,isAlphaBeta):
    result = globals()['result']
    
    moves = getAllMoves(checkers,player)
    
    if len(moves) == 0:
        isPass1 = True
        result["move"] = [-1,-1,-1,-1]
        result["utility"] = checkers.calculateUtility()
        result["myopicUtility"] = result["utility"]
        minVal = getMinValue(checkers,alpha,beta,currDepth+1,reverse(player),isPass1,isPass2,isAlphaBeta)
        result["farsightedUtility"] = minVal["farsightedUtility"]
    
    for m in moves:
        new_checkers = executeMove(checkers,m)
        minVal = getMinValue(new_checkers,alpha,beta,currDepth+1,reverse(player),False,False,isAlphaBeta)
        
        if minVal["utility"] > result["utility"]:
            result["move"] = m
            result["utility"] = minVal["utility"]
            result["myopicUtility"] = new_checkers.calculateUtility()
            result["farsightedUtility"] = minVal["farsightedUtility"]
        
        if isAlphaBeta:
            if result["utility"] >= beta:
                return
            alpha = max(alpha,result["utility"])

def reverse(player):
    if player == 'WHITE':
        return 'BLACK'
    return 'WHITE'

def getAllMoves(checkers,player):
    N = 8
    if player == 'WHITE':
        check = 'w'
    else:
        check = 'b'
    
    moves = []
    for i in range(N):
        for j in range(N):
            currCell = checkers.game_board[i][j]
            if currCell[0] == check:
                getValidMoves(i,j,checkers.game_board,moves,player)
    
    return moves

def getValidMoves(i,j,game_board,moves,player):
    N = len(game_board)
    
    if player == 'WHITE':
        if i > 1 and j > 1:
            jumpcell = game_board[i-2][j-2]
            adjcell = game_board[i-1][j-1]
            if adjcell[0] == 'b' and (jumpcell[0] == '.'):
                moves.append([i,j,i-2,j-2])
        
        if i > 1 and j < N-2:
            jumpcell = game_board[i-2][j+2]
            adjcell = game_board[i-1][j+1]
            if adjcell[0] == 'b' and (jumpcell[0] == '.'):
                moves.append([i,j,i-2,j+2])
        
        if i > 0 and j > 0:
            adjcell = game_board[i-1][j-1]
            if adjcell[0] == '.':
                moves.append([i,j,i-1,j-1])
        
        if i > 0 and j < N-1:
            adjcell = game_board[i-1][j+1]
            if adjcell[0] == '.':
                moves.append([i,j,i-1,j+1])
    else:
        if i < N-1 and j > 0:
            adjcell = game_board[i+1][j-1]
            if adjcell[0] == '.':
                moves.append([i,j,i+1,j-1])
        
        if i < N-1 and j < N-1:
            adjcell = game_board[i+1][j+1]
            if adjcell[0] == '.':
                moves.append([i,j,i+1,j+1])
        
        if i < N-2 and j > 1:
            jumpcell = game_board[i+2][j-2]
            adjcell = game_board[i+1][j-1]
            if adjcell[0] == 'w' and (jumpcell[0] == '.'):
                moves.append([i,j,i+2,j-2])
        
        if i < N-2 and j < N-2:
            jumpcell = game_board[i+2][j+2]
            adjcell = game_board[i+1][j+1]
            if adjcell[0] == 'w' and (jumpcell[0] == '.'):
                moves.append([i,j,i+2,j+2])
    
    return

def executeMove(checkers,move):
    new_game_board = copy.deepcopy(checkers.game_board)
    newPos = new_game_board[move[2]][move[3]]
    oldPos = new_game_board[move[0]][move[1]]
    
    if newPos[0] == '.':
        new_game_board[move[2]][move[3]] = oldPos
    else:
        new_game_board[move[2]][move[3]] = newPos[0]+str(int(newPos[1])+1)
    
    new_game_board[move[0]][move[1]] = '.'
    if int(abs(move[0]-move[2])) == 2:
        new_game_board[int((move[0]+move[2])/2)][int((move[1]+move[3])/2)] = '.'
    
    return GameState(new_game_board,checkers.player)

def isTerminal(checkers):
    if checkers.whiteCount == 0 or checkers.blackCount == 0:
        return True
    return False


class GameState(object):
    
    def __init__(self,game_board,player):
        self.game_board = game_board
        self.game_board_width = len(game_board)
        self.blackWeight = [10,20,30,40,50,60,70,80]
        self.whiteWeight = [80,70,60,50,40,30,20,10]
        self.player = player
        self.whiteCount = 0
        self.blackCount = 0
        self.setCount()
    
    def setCount(self):
        white,black = 0,0
        for i in range(self.game_board_width):
            for j in range(self.game_board_width):
                curr = self.game_board[i][j]
                if curr[0] == 'w':
                    self.whiteCount += 1
                elif curr[0] == 'b':
                    self.blackCount += 1
    
    def calculateUtility(self):
        white,black = 0,0
        for i in range(self.game_board_width):
            for j in range(self.game_board_width):
                curr = self.game_board[i][j]
                if curr == 'w':
                    white += (self.whiteWeight[i])
                elif curr == 'b':
                    black += (self.blackWeight[i])
        if self.player == 'WHITE':
            return white-black
        else:
            return black-white


if __name__ == '__main__':
    
    globals()['MIN_VALUE'] = -float('inf')
    globals()['MAX_VALUE'] = float('inf')
    globals()['result'] = {
        "move":[],
        "utility":MIN_VALUE,
        "myopicUtility":MIN_VALUE,
        "farsightedUtility":MIN_VALUE,
        "nodes":1
    }
    
    main()