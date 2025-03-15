cur_target = 0

def obs_to_state(obs):
    stations = [[0,0] for _ in range(4)]

    taxi_row, taxi_col, stations[0][0], stations[0][1], stations[1][0], stations[1][1], stations[2][0], stations[2][1], stations[3][0], stations[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    taxi_pos = (taxi_row, taxi_col)

    def get_vector(from_pos, to_pos):
        return (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])

    station_dirs = [get_vector(taxi_pos, stations[0]), get_vector(taxi_pos, stations[1]), get_vector(taxi_pos, stations[2]), get_vector(taxi_pos, stations[3])]

    global cur_target
    if station_dirs[0] == (0, 0):
        cur_target = 1
    elif station_dirs[1] == (0, 0):
        cur_target = 2
    elif station_dirs[2] == (0, 0):
        cur_target = 3
    elif station_dirs[3] == (0, 0):
        cur_target = 0
    target_dir = station_dirs[cur_target]

    # print("Cur Target:", cur_target)

    return (target_dir, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
