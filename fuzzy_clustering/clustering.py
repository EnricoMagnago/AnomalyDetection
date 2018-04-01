#!/usr/bin/python3

import sys
# add path in order to be able to find modules.
sys.path.insert(0, "../data_import/python_interface")

import DataTypes, DataLoader
import numpy as np
import skfuzzy as fuzz # requires scikit-fuzzy
import time


def load_data():
    retval = DataTypes.Data()
    loader = DataLoader.DataLoader("../dataset/")
    #loader.load_subset(retval, 1000)
    loader.load_all(retval)
    return retval

def get_np_arrays(data, window_size, step_size):
    assert(data.measures.size() > window_size)
    tanks_list =[DataTypes.TANK1, DataTypes.TANK2, DataTypes.TANK3]
    window_number = ((data.measures.size() - window_size) // step_size) + 1
    nan_array = np.empty((window_size, window_number))
    nan_array[:] = np.nan
    oxygen = [nan_array]*3
    nitrogen = [nan_array]*3
    sst = [nan_array]*3
    ammonia = [nan_array]*3
    flow = [nan_array]*3
    valve = [nan_array]*3
    for window_index in range(0, window_number):
        if window_index % 10000 == 0:
            print("window {}/{};".format(window_index, window_number))
        for index in range(0, window_size):
            data_index = window_index * window_size + index
            for tank_id in tanks_list:
                tank = data.measures[window_index][0][tank_id]
                oxygen[tank_id][index][window_index] = tank[DataTypes.OXYGEN] if 0 <= tank[DataTypes.OXYGEN] <= 200 else np.nan
                nitrogen[tank_id][index][window_index] = tank[DataTypes.NITROGEN] if 0 <= tank[DataTypes.NITROGEN] <= 200 else np.nan
                sst[tank_id][index][window_index] = tank[DataTypes.SST] if 0 <= tank[DataTypes.SST] <= 200 else np.nan
                ammonia[tank_id][index][window_index] = tank[DataTypes.AMMONIA] if 0 <= tank[DataTypes.AMMONIA] <= 200 else np.nan
                valve[tank_id][index][window_index] = tank[DataTypes.VALVE] if 0 <= tank[DataTypes.VALVE] <= 200 else np.nan
                flow[tank_id][index][window_index] = tank[DataTypes.FLOW] if 0 <= tank[DataTypes.FLOW] <= 200 else np.nan
    print()
    return oxygen, nitrogen, sst, ammonia, valve, flow



def main(argv):
    tanks_list = [DataTypes.TANK1, DataTypes.TANK2, DataTypes.TANK3]
    tanks_names = ["tank1", "tank2", "tank3"]
    sensors_list = [DataTypes.OXYGEN, DataTypes.NITROGEN, DataTypes.SST, DataTypes.AMMONIA, DataTypes.VALVE, DataTypes.FLOW]
    sensors_names = ["oxygen", "nitrogen", "sst", "ammonia", "valve", "flow"]

    loaded_data = load_data()
    window_size = 10
    step_size = 1
    print("reshape data, window_size: {} ; step_size: {}".format(window_size, step_size))
    arrays = get_np_arrays(loaded_data, window_size, step_size)

    features_number = arrays[0][0].shape[0]
    print("features number: {}".format(features_number))

    # clustering parameters
    n_centers = 3
    max_iter = 1000
    error = 0.005
    fuzzyfication = 2
    print("compute clusters, n_centers: {}, max_iter: {}, error: {}, fuzzyfication: {}".format(n_centers, max_iter, error, fuzzyfication))
    centroids = [[np.zeros((features_number, n_centers))]*len(tanks_list)]*len(sensors_list)
    for sensor_id in sensors_list:
        for tank_id in tanks_list:
            begin = time.time()
            centroids[sensor_id][tank_id], _, _, _, _, _, _ = \
                           fuzz.cluster.cmeans(arrays[sensor_id][tank_id],
                                               n_centers, fuzzyfication,
                                               error=error, maxiter=max_iter,
                                               init=None)
            print("computed centroid in: {}".format(time.time() - begin))
    cntr_oxygen, cntr_nitrogen, cntr_sst, cntr_ammonia, cntr_valve, cntr_flow = centroids
    for sensor_id in sensors_list:
        print("\n\n{}".format(sensors_names[sensor_id]))
        for tank_id in tanks_list:
            print("\n{}\n{}".format(tanks_names[tank_id], centroids[sensor_id][tank_id]))

if __name__ == "__main__":
    main(sys.argv)
