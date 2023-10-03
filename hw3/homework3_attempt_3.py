import operator
from itertools import repeat
from functools import reduce
from pprint import pprint
from threading import Thread
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import time


def main(multiprocess=True, debug=False):
    
    SEARCH_ALGORITHMS = ['A*', 'BFS', 'UCS']
    
    input_data = load_input_data()
    
    #for line in input_data['game_board']:
    #    print(line)
    #
    #print(input_data['end_positions'])
    
    search_type = input_data['search_type']
    if search_type not in SEARCH_ALGORITHMS:
        print('Invalid search algorithm.')
        output_string = write_output_data([['FAIL'] for i in range(len(end_positions))])
        return
    
    game_board = input_data['game_board']
    presets = input_data['presets']
    max_rock_height = presets['max_rock_height']
    map_size = presets['map_size']
    num_end_positions = presets['num_end_positions']
    
    start_position = input_data['start_position']
    end_positions = input_data['end_positions']
    
    if debug == True:
        if multiprocess == True and num_end_positions > 1:
            print("Multiprocessing is enabled.\n")
        else:
            print("Multiprocessing is disabled.\n")
    
    
    time_in = time.time()
    if multiprocess == True and num_end_positions > 1:
        with ProcessPoolExecutor(max_workers=min(num_end_positions, cpu_count())) as pool:
            output_data = list(pool.map(bi_directional_search, repeat(start_position), end_positions, repeat(presets), repeat(game_board), repeat(search_type)))
            #output_data = list(pool.map(bi_path_search, repeat(input_data), end_positions, repeat(search_type)))
    else:
        output_data = []
        for end_position in end_positions:
            path_sequence = bi_directional_search(start_position, end_position, presets, game_board, search_type)
            output_data.append(path_sequence)
            #print(path_sequence)
            #input()
    
    #input()
    
    '''
    time_in = time.time()
    if multiprocess == True and num_end_positions > 1:
        print("Multiprocessing is enabled.")
        with ProcessPoolExecutor(max_workers=min(num_end_positions, cpu_count())) as pool:
            output_data = list(pool.map(path_search, repeat(start_position), end_positions, repeat(primary_dict), repeat(presets), repeat(game_board), repeat(search_type)))
            #output_data = list(pool.map(bi_path_search, repeat(input_data), end_positions, repeat(search_type)))
    else:
        print("Multiprocessing is disabled.")
        output_data = []
        for end_position in end_positions:
            output_data.append(path_search(start_position, end_position, {}, presets, game_board, search_type))
            globals()['forward_dict'] = {}
            globals()['backward_dict']= {}
            globals()['path_found'] = ''
            globals()['continue_comparator'] = True
            #output_data.append(bi_path_search(input_data, end_position, search_type))
    time_out = time.time() - time_in
    '''
    
    
    print(f"Elapsed time: {time.time()-time_in} sec")
    
    output_string = write_output_data(output_data)
    print(output_string)
    
    return


# Direction is 0 for forward, 1 for backward
def path_search(start_position, end_position, primary_dict, presets, game_board, search_type, direction=0, deep=False, debug=False):
    
    globals()['max_forward_length'] = -1
    globals()['max_backward_length'] = -1
    
    FORWARD_DIRECTION = 0
    BACKWARD_DIRECTION = 1
    
    if direction == BACKWARD_DIRECTION:
        start_position_copy = start_position.copy()
        start_position = end_position.copy()
        end_position = start_position_copy
    
    # Load dictionary tools class
    d_tools = Dict_Tools()
    
    
    max_rock_height = presets['max_rock_height']
    map_size = presets['map_size']
    
    start_position_id = f"{start_position['X']},{start_position['Y']}"
    start_height = max(0, -start_position['M'])
    start_mud = max(0, start_position['M'])
    end_position_id = f"{end_position['X']},{end_position['Y']}"
    end_height = max(0, -end_position['M'])
    end_mud = max(0, end_position['M'])
    
    
    # If already at settling site, return start_position
    if start_position_id == end_position_id:
        globals()['continue_comparator'] = False
        globals()['path_found'] = [start_position_id]
        return [start_position_id]
    
    
    # For immediate handling inaccessible settling destinations
    final_state = get_state(end_position, game_board, map_size, search_type)
    final_positions = sorted(final_state.keys())
    inaccessible = []
    for position in final_positions:
        position_height = max(0, -final_state[position]['M'])
        if abs(end_height - position_height) > max_rock_height:
            inaccessible.append(position)
    if len(inaccessible) == len(final_positions):
        globals()['path_found'] = ['FAIL']
        globals()['continue_comparator'] = False
        return ['FAIL']
    
    
    # Reverse cost for reverse directional path
    if search_type == 'A*' and direction == BACKWARD_DIRECTION:
        start_position['C'] = start_mud #Initialize cost for starting position
    else:
        start_position['C'] = 0
    
    primary_dict[start_position_id] = start_position.copy()
    dict_id = [start_position_id]
    current_state = primary_dict[dict_id[0]]
    
    counter = 0
    while globals()['continue_comparator']:
        #process_nodes = d_tools.get_all_keys(primary_dict, black_list=['X', 'Y', 'M'], white_list=['NODE'], map='C')
        
        process_queue = d_tools.get_all_keys(primary_dict, black_list=['X', 'Y', 'M', 'NODE'], map='C')
        
        if deep == True:
            process_queue.sort(key=len)
            process_queue = process_queue[::-1] # Reverse b
            
        
        if debug == True:
            print('--------------')
            print('process_queue:')
            print('--------------')
            pprint(process_queue[0:15])
            input('\n')
        
        # If no more processes are in queue, break from loop
        if len(process_queue) == 0:
            break
        
        process = process_queue[0]
        priority = process[0]
        dict_id = process[1:]
        
        if direction == FORWARD_DIRECTION:
            globals()['max_forward_length'] = max(globals()['max_forward_length'], len(dict_id))
        elif direction == BACKWARD_DIRECTION:
            globals()['max_backward_length'] = max(globals()['max_backward_length'], len(dict_id))
        
        if dict_id[len(dict_id)-1] == end_position_id and globals()['continue_comparator'] == True:
            if direction == BACKWARD_DIRECTION:
                dict_id.reverse()
            if debug == True:
                pprint(dict_id)
                pprint(process)
                pprint(process_queue[0:15])
                print('--------------------')
                print('    RETURN PATH ')
                print('--------------------')
                pprint(dict_id)
                #input('\n')
            
            if deep == True:
                globals()['deepest_cost'] = priority
                print("deepest_cost", globals()['deepest_cost'])
                pprint(dict_id)
                print(dict_id[0])
                print('deepest_length', len(dict_id))
                globals()['deepest_length'] = len(dict_id)
                return
            
            globals()['continue_comparator'] = False
            if globals()['path_found'] == '':
                globals()['path_found'] = dict_id
                if direction == BACKWARD_DIRECTION:
                    final_cost = priority-end_mud
                    print(f'Backward finished first with cost {final_cost}.')
                else:
                    final_cost = priority
                    print(f'Forward finished first with cost {final_cost}.')
                return dict_id
            else:
                return
        
        current_state = d_tools.get_from_dict(primary_dict, dict_id)
        current_mud = max(0, current_state['M'])
        current_height = max(0, -current_state['M'])
        current_cost = current_state['C']
        
        if debug == True:
            print('--------------')
            print('current_state:')
            print('--------------')
            pprint(current_state)
            input('\n')
        
        future_state = get_state(current_state, game_board, map_size, search_type, current_cost, dict_id)
        
        if debug == True:
            print('----------------')
            print('future_state PRE')
            print('----------------')
            pprint(future_state)
            input('\n')
        
        apply_state_filter(primary_dict, current_state, future_state, dict_id, current_height, max_rock_height)
        
        if debug == True:
            print('-------------')
            print('future_state ')
            print('-------------')
            pprint(future_state)
            input('\n')
            print('--------------')
            print('primary_dict:')
            print('--------------')
            pprint(primary_dict)
            input('\n')
        
        counter += 1
    
    if globals()['continue_comparator'] == True:
        globals()['continue_comparator'] = False
        globals()['path_found'] = ['FAIL']
        
    
    return ['FAIL']

# Executes 3 simultaneous threads.  Each thread has the ability to end the other.
# Utilizes 'node_comparator' to make a comparison between forwards and backwards nodes.
def bi_directional_search(start_position, end_position, presets, game_board, search_type):
    
    # Re-initialize global variables
    globals()['deep_dict'] = {}
    globals()['forward_dict'] = {}
    globals()['backward_dict']= {}
    globals()['path_found'] = ''
    globals()['continue_comparator'] = True
    globals()['deepest_length'] = float('inf')
    globals()['deepest_cost'] = 0
    globals()['max_forward_length'] = -1
    globals()['max_backward_length'] = -1
    
    path_search(start_position, end_position, globals()['deep_dict'], presets, game_board, search_type, 0, deep=True)
    print('done')
    background_thread(node_comparator, [start_position, end_position, globals()['forward_dict'], globals()['backward_dict'], game_board, search_type])
    background_thread(path_search, [start_position, end_position, globals()['backward_dict'], presets, game_board, search_type, 1])
    #background_thread(path_search, [start_position, end_position, globals()['backward_dict'], presets, game_board, search_type, 1])
    path_search(start_position, end_position, globals()['forward_dict'], presets, game_board, search_type, 0)
    #input()
    #node_comparator(start_position, end_position, globals()['forward_dict'], globals()['backward_dict'])
    globals()['continue_comparator'] = False
    output_data = globals()['path_found'].copy()
    
    
    return output_data


# Compare nodes between two built-in dictionaries for bi_directional_search
def node_comparator(start_position, end_position, forward_dict, backward_dict, game_board, search_type, debug=False):
    
    MIN_NODE_CONNECTIONS = 10000
    #TIME_BUFFER = 0.01
    #MAX_LOOP_COUNT = 10
    LIST_OFFSET = 2
    min_distance = LIST_OFFSET
    #in_distance = int((((start_position['X']-end_position['X'])**2 + (start_position['Y']-end_position['Y'])**2)**.5)/2) + LIST_OFFSET
    #max_distance = abs(start_position['X']-end_position['X']) + abs(start_position['Y']-end_position['Y'])
    max_distance = int((((start_position['X']-end_position['X'])**2 + (start_position['Y']-end_position['Y'])**2)**.5)) + LIST_OFFSET
    #max_distance = min_distance +1
    
    d_tools = Dict_Tools()
    
    node_connector = []
    node_connection_length = 0
    forward_nodes_length = -1
    backward_nodes_length = -1
    max_forward_node_length = 0
    max_backward_node_length = 0
    
    # Overlap cost is set to account for the mud on the intersecting squares.  Is used for only A* search.
    overlap_cost = 0
    
    #time.sleep(TIME_BUFFER)
    #time_in = time.time()
    loop_counter = 0
    while globals()['continue_comparator']:
        
        #time.sleep((time.time()-time_in))
        #time_in = time.time()
        #max_distance += 1
        
        #forward_nodes = d_tools.get_all_keys(forward_dict, black_list=['X', 'Y', 'M'], white_list=['NODE'], map='C')
        #backward_nodes = d_tools.get_all_keys(backward_dict, black_list=['X', 'Y', 'M'], white_list=['NODE'], map='C')
        forward_nodes = d_tools.get_all_keys(forward_dict, black_list=['X', 'Y', 'M'], map='C')
        backward_nodes = d_tools.get_all_keys(backward_dict, black_list=['X', 'Y', 'M'], map='C')
        
        if len(forward_nodes) != 0 and len(backward_nodes) != 0:
            max_forward_node_length = len(max(forward_nodes, key=len))
            max_backward_node_length = len(max(backward_nodes, key=len))
        
        #pprint(forward_nodes)
        new_forward_nodes = []
        new_backward_nodes = []
        max_node_length = 0
        for i in range(max(len(forward_nodes), len(backward_nodes))):
            if i < len(forward_nodes) and (len(forward_nodes[i]) <= max_distance and len(forward_nodes[i]) >= min_distance):
                new_forward_nodes.append(forward_nodes[i])
                max_node_length = max(max_node_length, len(forward_nodes[i]))
            if i < len(backward_nodes) and (len(backward_nodes[i]) <= max_distance and len(backward_nodes[i]) >= min_distance):
                new_backward_nodes.append(backward_nodes[i][::-1])
                max_node_length = max(max_node_length, len(backward_nodes[i]))
                
        forward_nodes = new_forward_nodes
        backward_nodes = new_backward_nodes
        
        #pprint(forward_nodes)
        #pprint(backward_nodes)
        
        
        if (len(forward_nodes) == 0 and len(backward_nodes) == 0):
            if debug == True:
                print("Forward and backward length are of size 0...")
            continue
        
        #forward_nodes = [k for k in forward_nodes if (len(k) <= max_distance and len(k) > min_distance)]
        # Reverse order of each entry in backward nodes
        #backward_nodes = [k[::-1] for k in backward_nodes if (len(k) <= max_distance and len(k) > min_distance)]
        
        
        for i in range(len(forward_nodes)):
            forward_node = forward_nodes[i]
            for j in range(len(backward_nodes)):
                backward_node = backward_nodes[j]
                if forward_node[-1] == 'NODE' and backward_node[0] == 'NODE':
                    if forward_node[-2] == backward_node[1]:
                        [forward_x, forward_y] = forward_node[-2].split(',')
                        path = forward_node[:-1]+backward_node[2:]
                        if search_type == 'A*':
                            overlap_cost = max(0, game_board[int(forward_y)][int(forward_x)])
                        path[0] += path[-1] - overlap_cost
                        path.pop(-1)
                        node_connector.append(path)
                elif forward_node[-1] != 'NODE' and backward_node[0] != 'NODE':
                    if forward_node[-1] == backward_node[0]:
                        [forward_x, forward_y] = forward_node[-1].split(',')
                        path = forward_node[:-1]+backward_node
                        if search_type == 'A*':
                            overlap_cost = max(0, game_board[int(forward_y)][int(forward_x)])
                        path[0] += path[-1] - overlap_cost
                        path.pop(-1)
                        node_connector.append(path)
        
        # Convert node_connector list into a unique set
        node_connector = sorted([list(x) for x in set(tuple(x) for x in node_connector)])
        
        print( globals()['max_forward_length'],  globals()['max_backward_length'] )
        #node_connection_length == len(node_connector) and 
        if len(node_connector) > MIN_NODE_CONNECTIONS and globals()['continue_comparator'] == True and \
           globals()['max_forward_length'] + globals()['max_backward_length'] >= globals()['deepest_length']:
            #forward_nodes_length == len(forward_nodes) and backward_nodes_length == len(backward_nodes):
            #print(node_connector)
            #cost = node_connector[0][0]
            #
            #if cost < globals()['forward_cost'] and cost < globals()['backward_cost']:
            if globals()['path_found'] == '':
                cost = node_connector[0][0]
                if cost < globals()['deepest_cost']:
                    globals()['path_found'] = node_connector[0][1:]
                    print(f'Node comparator finished first with cost {cost}.')
                    #print(f'Cost: {cost}')
                    #print(node_connector[0][1:])
                    #input()
                    globals()['continue_comparator'] = False
                    
                    return node_connector[0][1:]
            else:
                return
        
        elif forward_nodes_length == len(forward_nodes) and backward_nodes_length == len(backward_nodes) and \
               max_node_length == max_distance and len(forward_nodes) != 0 and len(backward_nodes) != 0:
            if debug == True:
                print("Forward and backward length are not growing...", len(forward_nodes), len(backward_nodes))
            max_distance += 1
            loop_counter = 0
        
        
        forward_nodes_length = len(forward_nodes)
        backward_nodes_length = len(backward_nodes)
        
        if debug == True:
            print('--node_connector--')
            pprint(node_connector)
            print('------------------')
            print('--forward_nodes--')
            pprint(forward_nodes)
            print('------------------')
            print('--backward_nodes--')
            pprint(backward_nodes)
            print('------------------\n')
        
        loop_counter += 1
        
    
    if globals()['continue_comparator'] == True:
        if globals()['path_found'] == '':
            globals()['path_found'] = ['FAIL']
        return ['FAIL']
    
    globals()['continue_comparator'] = False
    
    return


# Remove inaccessible paths from dictionary tree otherwise extend tree
def apply_state_filter(primary_dict, current_state, future_state, dict_id, current_height, max_rock_height):
    d_tools = Dict_Tools()
    
    # Remove inaccessible paths
    future_keys = list(future_state.keys())
    for key in future_keys:
        future_M = future_state[key]['M']
        future_mud = max(0, future_M)
        future_height = max(0, -future_M)
        if abs(future_height - current_height) > max_rock_height:
            future_state.pop(key)
    # If no future_future branches from state exist, delete tree
    if len(future_state.keys()) == 0 and len(dict_id) > 1:
        d_tools.del_from_dict(primary_dict, dict_id)
    else: # Otherwise extend tree
        future_state['NODE'] = current_state
        d_tools.set_in_dict(primary_dict, dict_id, future_state)


# Apply formatting to the dictioanry state
def format_state(state, start_position, game_board, x, y, cost, prev_cost, search_type, unique_keys_list):
    if search_type == 'A*':
        height = max(0, -start_position['M'])
        future_height = max(0, -game_board[y][x])
        future_mud = max(0, game_board[y][x])
        cost += abs(height - future_height) + future_mud
    
    if f'{x},{y}' not in unique_keys_list:
        state[f'{x},{y}'] = {'C':prev_cost+cost, 'M':game_board[y][x], 'X':x, 'Y':y}


# Populate a given state/node by mapping positions to the game_board
def get_state(start_position, game_board, map_size, search_type, prev_cost=0, unique_keys_list=[]):
    
    map_width = map_size['W']
    map_height = map_size['H']
    
    x_position = start_position['X']
    y_position = start_position['Y']
    
    
    y_center = y_position
    y_top = y_position-1
    y_bottom = y_position+1
    x_center = x_position
    x_left = x_position-1
    x_right = x_position+1
    
    
    if search_type == 'BFS':
        cost_map = [1]*8
    elif search_type == 'UCS' or search_type == 'A*':
        cost_map = [10, 10, 10, 10, 14, 14, 14, 14]
    else:
        return {}
    
    
    state = {}
    if x_center < 0 or x_center >= map_width or y_center < 0 or y_center >= map_height:
        pass
    else:
        #state[f'{x_center},{y_center}'] = {'M':game_board[y_center][x_center], 'X':x_center, 'Y':y_center}
        if (y_top >= 0 and y_bottom < map_height):
            if (x_left >= 0 and x_right < map_width):
                format_state(state, start_position, game_board, x_center, y_top,    cost_map[0], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_center, cost_map[1], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_center, y_bottom, cost_map[2], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_center, cost_map[3], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_top,    cost_map[4], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_bottom, cost_map[5], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_bottom, cost_map[6], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_top,    cost_map[7], prev_cost, search_type, unique_keys_list)
            elif x_left < 0:
                format_state(state, start_position, game_board, x_center, y_top,    cost_map[0], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_center, cost_map[1], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_center, y_bottom, cost_map[2], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_top,    cost_map[4], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_bottom, cost_map[5], prev_cost, search_type, unique_keys_list)
            elif x_right >= map_width:
                format_state(state, start_position, game_board, x_center, y_top,    cost_map[0], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_center, y_bottom, cost_map[2], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_center, cost_map[3], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_bottom, cost_map[6], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_top,    cost_map[7], prev_cost, search_type, unique_keys_list)
        elif y_top < 0:
            if (x_left >= 0 and x_right < map_width):
                format_state(state, start_position, game_board, x_right,  y_center, cost_map[1], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_center, y_bottom, cost_map[2], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_center, cost_map[3], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_bottom, cost_map[5], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_bottom, cost_map[6], prev_cost, search_type, unique_keys_list)
            elif x_left < 0:
                format_state(state, start_position, game_board, x_right,  y_center, cost_map[1], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_center, y_bottom, cost_map[2], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_bottom, cost_map[5], prev_cost, search_type, unique_keys_list)
            elif x_right >= map_width:
                format_state(state, start_position, game_board, x_center, y_bottom, cost_map[2], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_center, cost_map[3], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_bottom, cost_map[6], prev_cost, search_type, unique_keys_list)
        elif y_bottom >= map_height:
            if (x_left >= 0 and x_right < map_width):
                format_state(state, start_position, game_board, x_center, y_top,    cost_map[0], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_center, cost_map[1], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_center, cost_map[3], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_top,    cost_map[4], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_top,    cost_map[7], prev_cost, search_type, unique_keys_list)
            elif x_left < 0:
                format_state(state, start_position, game_board, x_center, y_top,    cost_map[0], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_center, cost_map[1], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_right,  y_top,    cost_map[4], prev_cost, search_type, unique_keys_list)
            elif x_right >= map_width:
                format_state(state, start_position, game_board, x_center, y_top,    cost_map[0], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_center, cost_map[3], prev_cost, search_type, unique_keys_list)
                format_state(state, start_position, game_board, x_left,   y_top,    cost_map[7], prev_cost, search_type, unique_keys_list)
        
    
    return state



# Input: Takes in the file path for input.txt
# Output: Returns a dictionary containing the input parameters 
def load_input_data(file_path='input.txt'):
    
    # Read input file
    with open(file_path) as f:
        raw_data = f.read()
    split_data = raw_data.split('\n')
    
    # Sparce configurations from split_data
    search_type = split_data[0]
    map_size = split_data[1].split()
    map_width = int(map_size[0])
    map_height = int(map_size[1])
    start_position = split_data[2].split()
    x_start = int(start_position[0])
    y_start = int(start_position[1])
    
    max_rock_height = int(split_data[3])
    num_end_positions = int(split_data[4])
    
    site_start_index = 5
    site_end_index = len(split_data) - map_height
    
    
    game_board = []
    for i in range(site_end_index, len(split_data)):
        game_board.append(list(map(int, split_data[i].split())))
    
    end_positions = []
    for i in range(site_start_index, site_end_index):
        site_start_position = split_data[i].split()
        x_end = int(site_start_position[0])
        y_end = int(site_start_position[1])
        end_positions.append({'X':x_end, 'Y':y_end, 'M':int(game_board[y_end][x_end])})
    
    
    # Generate dictionary
    input_data = {}
    input_data['search_type'] = search_type
    input_data['start_position'] = {'X':x_start, 'Y':y_start, 'M':int(game_board[y_start][x_start])}
    input_data['end_positions'] = end_positions
    input_data['presets'] = {'max_rock_height':max_rock_height, 'num_end_positions':num_end_positions, 'map_size':{'W':map_width, 'H':map_height}}
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


# For making object run in background
def background_thread(target, args_list):
    args = ()
    for i in range(len(args_list)):
        args = args + (args_list[i],)
    pr = Thread(target=target, args=args)
    pr.daemon = True
    pr.start()
    
    return pr


# Custom toolkit for dictionary manipulation
class Dict_Tools:
    def __init__(self):
        pass
    
    def get_from_dict(self, dictionary, map_list):
        return reduce(operator.getitem, map_list, dictionary)
    
    def del_from_dict(self, dictionary, map_list):
        self.get_from_dict(dictionary, map_list[:-1]).pop(map_list[-1])
    
    def set_in_dict(self, dictionary, map_list, value):
        self.get_from_dict(dictionary, map_list[:-1])[map_list[-1]] = value
    
    def get_keys(self, dictionary, black_list=[], map='', prev_key = None, keys = []):
        
        if type(dictionary) != type({}):
            keys.append(prev_key)
            return keys
        new_keys = []
        for k, v in sorted(dictionary.items()):
            if k not in black_list:
                if k == map:
                    if prev_key != None:
                        new_key = f'{v} {prev_key}'
                    else:
                        new_key = v
                else:
                    if prev_key != None:
                        new_key = f'{prev_key} {k}'
                    else:
                        new_key = k
                new_keys.extend(self.get_keys(v, black_list, map, new_key, []))
        
        return sorted(new_keys)
    
    # Filter will remove values that from keys based upon a list
    # Map will assign the value at a designated key to the front of the keys list
    def get_all_keys(self, dictionary, black_list=[], white_list=[], map=''):
        
        if len(dictionary) == 0:
            return []
        
        keys = self.get_keys(dictionary, black_list, map)
        #keys = [key.split() for key in keys]
        
        
        new_keys = []
        for key in keys:
            new_key = key.split()
            
            if map != '':
                new_key[0] = int(new_key[0])
                
            new_keys.append(new_key)
        
        if len(white_list) > 0:
            new_keys = [k for k in new_keys if set(white_list) & set(k)]
        
        
        return sorted(new_keys)


if __name__ == '__main__':
    # Initialize global variables
    globals()['forward_dict'] = {}
    globals()['backward_dict']= {}
    globals()['path_found'] = ''
    globals()['continue_comparator'] = True
    
    try:
        main()
    except:
        print('Multiprocessing has failed!')
        main(multiprocess=False)