from homework import *
import time
#import json

time_in = time.time()

MAX_DEPTH = 8
NUMBER_OF_LOOPS = 4
MAX_TEST_TIME = 100

TEST_BOARD_1 = [['.','b','.','b','.','b','.','b'],
                ['b','.','b','.','.','.','b','.'],
                ['.','b','.','b','.','.','.','b'],
                ['.','.','b','.','.','.','b','.'],
                ['.','.','.','w','.','w','.','.'],
                ['w','.','w','.','w','.','.','.'],
                ['.','.','.','w','.','w','.','w'],
                ['w','.','w','.','w','.','w','.']]

TEST_BOARD_2 = [['.','.','.','.','.','.','.','.'],
                ['.','.','B','.','.','.','.','.'],
                ['.','B','.','B','.','.','.','b'],
                ['.','.','B','.','.','.','.','.'],
                ['.','b','.','.','.','W','.','.'],
                ['.','.','.','.','W','.','.','.'],
                ['.','.','.','.','.','W','.','w'],
                ['W','.','.','.','w','.','w','.']]

TEST_BOARD_3 = [['.','.','.','.','.','.','.','W'],
                ['.','.','B','.','.','.','.','.'],
                ['.','B','.','.','.','B','.','.'],
                ['.','.','.','.','.','.','.','.'],
                ['.','B','.','.','.','.','.','.'],
                ['.','.','.','.','W','.','.','.'],
                ['.','b','.','W','.','W','.','.'],
                ['.','.','.','.','W','.','w','.']]

calibration = []
print('-----------------')
print('Testing board #1.')
print('-----------------')
for depth in range(1, MAX_DEPTH+1):
    elapsed_time = 0
    for i in range(NUMBER_OF_LOOPS):
        print(f"Depth {depth} test #{i+1}...")
        elapsed_time += float(main(test=True, test_time=MAX_TEST_TIME, test_depth=depth, test_board=TEST_BOARD_1))
    elapsed_time = elapsed_time / NUMBER_OF_LOOPS
    
    calibration.append(elapsed_time)

print('-----------------')
print('Testing board #2.')
print('-----------------')
for depth in range(1, MAX_DEPTH+1):
    elapsed_time = 0
    for i in range(NUMBER_OF_LOOPS):
        print(f"Depth {depth} test #{i+1}...")
        elapsed_time += float(main(test=True, test_time=MAX_TEST_TIME, test_depth=depth, test_board=TEST_BOARD_2))
    elapsed_time = elapsed_time / NUMBER_OF_LOOPS
    
    calibration[depth-1] = (calibration[depth-1] + elapsed_time)/2


print('-----------------')
print('Testing board #3.')
print('-----------------')
for depth in range(1, MAX_DEPTH+1):
    elapsed_time = 0
    for i in range(NUMBER_OF_LOOPS):
        print(f"Depth {depth} test #{i+1}...")
        elapsed_time += float(main(test=True, test_time=MAX_TEST_TIME, test_depth=depth, test_board=TEST_BOARD_3))
    elapsed_time = elapsed_time / NUMBER_OF_LOOPS
    
    calibration[depth-1] = (calibration[depth-1]*2 + elapsed_time)/3


#calibration[8] = float(main(test=True, test_time=MAX_TEST_TIME, test_depth=9, test_board=TEST_BOARD_2))

# For writing turn data information.  WARNING: this will consume resource time!
def write_calibration_data(calibration, file_path='calibration.txt'):
    #with open(file_path, 'w') as json_file:
    #    json.dump(calibration, json_file)
    
    output_string = ''
    for i in range(len(calibration)):
        if i != len(calibration)-1:
            output_string += str(calibration[i])+'\n'
        else:
            output_string += str(calibration[i])
    
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
    
    return calibration


write_calibration_data(calibration)
#pprint(calibration)
a = load_calibration_data()
pprint(a)
print(f'Total calibration time:{time.time()-time_in}')