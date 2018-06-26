#!/usr/bin/python3

import sys
# add path to find modules.
sys.path.insert(0, "../data_import/python_interface")

import DataTypes, DataLoader
import numpy as np
import skfuzzy as fuzz # requires scikit-fuzzy
import time
import math
import pickle


def load_data():
    retval = DataTypes.Data()
    loader = DataLoader.DataLoader("../dataset/")
    #loader.load_subset(retval, 1000)
    loader.load_all(retval)
    return retval

def data_index_from_window_index(window_index, window_number, step_size):
    return (window_number * step_size) + window_index

def get_np_arrays(data, window_size, step_size):
    assert(data.measures.size() > window_size)
    tanks_list =[DataTypes.TANK1, DataTypes.TANK2, DataTypes.TANK3]
    window_number = ((data.measures.size() - window_size) // step_size) + 1
    nan_array = np.empty((window_size, window_number))
    nan_array[:] = np.nan
    oxygen = [nan_array] * len(tanks_list)
    nitrogen = [nan_array] * len(tanks_list)
    sst = [nan_array] * len(tanks_list)
    ammonia = [nan_array] * len(tanks_list)
    flow = [nan_array] * len(tanks_list)
    valve = [nan_array] * len(tanks_list)
    for window_index in range(0, window_number):
        #if window_index % 10000 == 0:
            #print("window {}/{};".format(window_index, window_number))
        avg_oxygen = [0] * len(tanks_list)
        oxygen_counter = [0] * len(tanks_list)
        avg_nitrogen = [0] * len(tanks_list)
        nitrogen_counter = [0] * len(tanks_list)
        avg_sst = [0] * len(tanks_list)
        sst_counter = [0] * len(tanks_list)
        avg_ammonia = [0] * len(tanks_list)
        ammonia_counter = [0] * len(tanks_list)
        avg_flow = [0] * len(tanks_list)
        flow_counter = [0] * len(tanks_list)
        avg_valve = [0] * len(tanks_list)
        valve_counter = [0] * len(tanks_list)
        for index in range(0, window_size):
            # collect sum of values and their number to compute average.
            data_index = data_index_from_window_index(index, window_index, step_size)
            for tank_id in tanks_list:
                # tank is a vector of 6 elements that represents the sensor value
                # for the given measurement (data_index) in the given tank (tank_id)
                # the 0 take the first element of the pair (TankMeasures, PowerMeasures).
                tank = data.measures[data_index][0][tank_id]
                avg_oxygen[tank_id] += tank[DataTypes.OXYGEN] if 0 <= tank[DataTypes.OXYGEN] <= 200 else 0
                oxygen_counter[tank_id] += 1 if 0 <= tank[DataTypes.OXYGEN] <= 200 else 0
                avg_nitrogen[tank_id] += tank[DataTypes.NITROGEN] if 0 <= tank[DataTypes.NITROGEN] <= 200 else 0
                nitrogen_counter[tank_id] += 1 if 0 <= tank[DataTypes.NITROGEN] <= 200 else 0
                avg_sst[tank_id] += tank[DataTypes.SST] if 0 <= tank[DataTypes.SST] <= 200 else 0
                sst_counter[tank_id] += 1 if 0 <= tank[DataTypes.SST] <= 200 else 0
                avg_ammonia[tank_id] += tank[DataTypes.AMMONIA] if 0 <= tank[DataTypes.AMMONIA] <= 200 else 0
                ammonia_counter[tank_id] += 1 if 0 <= tank[DataTypes.AMMONIA] <= 200 else 0
                avg_flow[tank_id] += tank[DataTypes.FLOW] if 0 <= tank[DataTypes.FLOW] <= 200 else 0
                flow_counter[tank_id] += 1 if 0 <= tank[DataTypes.FLOW] <= 200 else 0
                avg_valve[tank_id] += tank[DataTypes.VALVE] if 0 <= tank[DataTypes.VALVE] <= 200 else 0
                valve_counter[tank_id] += 1 if 0 <= tank[DataTypes.VALVE] <= 200 else 0
        for tank_id in tanks_list:
            # compute average
            avg_oxygen[tank_id] = (avg_oxygen[tank_id] / oxygen_counter[tank_id]) if oxygen_counter[tank_id] > 0 else 0
            avg_nitrogen[tank_id] = (avg_nitrogen[tank_id] / nitrogen_counter[tank_id]) if nitrogen_counter[tank_id] > 0 else 0
            avg_sst[tank_id] = (avg_sst[tank_id] / sst_counter[tank_id]) if sst_counter[tank_id] > 0 else 0
            avg_ammonia[tank_id] = (avg_ammonia[tank_id] / ammonia_counter[tank_id]) if ammonia_counter[tank_id] > 0 else 0
            avg_flow[tank_id] = (avg_flow[tank_id] / flow_counter[tank_id]) if flow_counter[tank_id] > 0 else 0
            avg_valve[tank_id] = (avg_valve[tank_id] / valve_counter[tank_id]) if valve_counter[tank_id] > 0 else 0
        # re-iterate through the current window.
        for index in range(0, window_size):
            # collect data, replace nan values with the average.
            data_index = data_index_from_window_index(index, window_index, step_size)
            for tank_id in tanks_list:
                tank = data.measures[data_index][0][tank_id]
                oxygen[tank_id][index][window_index] = tank[DataTypes.OXYGEN] if 0 <= tank[DataTypes.OXYGEN] <= 200 else avg_oxygen[tank_id]
                nitrogen[tank_id][index][window_index] = tank[DataTypes.NITROGEN] if 0 <= tank[DataTypes.NITROGEN] <= 200 else avg_nitrogen[tank_id]
                sst[tank_id][index][window_index] = tank[DataTypes.SST] if 0 <= tank[DataTypes.SST] <= 200 else avg_sst[tank_id]
                ammonia[tank_id][index][window_index] = tank[DataTypes.AMMONIA] if 0 <= tank[DataTypes.AMMONIA] <= 200 else avg_ammonia[tank_id]
                valve[tank_id][index][window_index] = tank[DataTypes.VALVE] if 0 <= tank[DataTypes.VALVE] <= 200 else avg_valve[tank_id]
                flow[tank_id][index][window_index] = tank[DataTypes.FLOW] if 0 <= tank[DataTypes.FLOW] <= 200 else avg_flow[tank_id]
    #print()
    index_to_time = [0] * data.index_to_time.size()
    for i in range(0, data.index_to_time.size()):
        index_to_time[i] = DataLoader.time_to_string(data, i)
    return [oxygen, nitrogen, sst, ammonia, valve, flow], index_to_time

def compute_autocorrelation_vectors(data):
    """ input data: 6 sensors * 3 tanks * (window_length * window_number)
    input vector is a matrix with shape (window_size, N)
    output matrix has shape 3 tanks * ((6 * window_length - 1) * window_number)
    we compute the auto-correlation coefficient for each sensor in each tank
    separately then we concatenate the features of the 6 sensor to get a single
    vector that represents the state of a tank in a particular instant. """
    sensors_number = len(data)
    tanks_number = len(data[0])
    window_length = data[0][0].shape[0]
    windows_number = data[0][0].shape[1]
    #samples_number
    # for each window we compute a vector of features: different lags: lag in [1; window_length*sensors_number)
    output = [np.empty((sensors_number * window_length - 1, windows_number), dtype=np.float64)] * tanks_number
    for tank_id in range(0, tanks_number):
        max_tank_val = float('-Inf')
        for window_id in range(0, windows_number):
            window_mean = 0
            window_squared_difference = 0
            for sensor_id in range(0, sensors_number):
                for window_index in range(0, window_length):
                    window_mean += data[sensor_id][tank_id][window_index][window_id]
            window_mean /= (window_length * sensors_number)
            for sensor_id in range(0, sensors_number):
                for window_index in range(0, window_length):
                    window_squared_difference += (data[sensor_id][tank_id][window_index][window_id] - window_mean) ** 2

            for lag in range(1, window_length * sensors_number):
                curr_lag =  lag % window_length # in [0; window_length -1]
                sensor_id = lag // window_length # in [0; sensors_number -1]
                autocorrelation = 0
                if window_squared_difference == 0:
                    # if window_mean != 0:
                    #     print('mean: {} ; sum of square difference: {}'.format(window_mean, window_squared_difference))
                    autocorrelation = float('Inf')
                else:
                    for window_index in range(0, window_length - curr_lag):
                         autocorrelation += \
                             ((data[sensor_id][tank_id][window_index][window_id] - window_mean) * \
                             (data[sensor_id][tank_id][window_index + curr_lag][window_id] - window_mean))
                    autocorrelation /= window_squared_difference
                    max_tank_val = max(autocorrelation, max_tank_val)
                output[tank_id][lag - 1][window_id] = autocorrelation
        for window_id in range(0, windows_number):
            for lag in range(0, (window_length * sensors_number) - 1):
                if output[tank_id][lag][window_id] == float('Inf'):
                    output[tank_id][lag][window_id] = max_tank_val + 0.001

    return output


def cluster_data(data, n_centers, max_iter, error, fuzzyfication):
    centroids, u, u0, d, _, iterations, fpc = fuzz.cluster.cmeans(data,
                                                                 n_centers,
                                                                 fuzzyfication,
                                                                 error=error,
                                                                 maxiter=max_iter,
                                                                 init=None)
    u = u ** fuzzyfication
    reconstructed = np.empty_like(data.T)
    for k in range(0, u.shape[1]):
        u_sum = np.sum(u[:, k], 0)
        numerator = np.array([np.fmax((u[cluster_index, k]*centroids[cluster_index])/u_sum, \
                                      np.finfo(np.float64).eps) \
                              for cluster_index in range(0, centroids.shape[0])])
        reconstructed[k] = np.sum(numerator, 0)

    return centroids, reconstructed.T


def main(argv):
    # if (len(argv) != 7 and len(argv) != 1):
    #     print("usage: {} window_size step_size clusters fuzzyfication auto_corr_clusters auto_corr_fuzzyfication".format(argv[0]))
    #     exit(1)

    window_size_list = [10, 20, 40, 60, 80, 100]
    step_size_list = [4, 6, 8, 12, 16, 20, 24, 28]
    n_centers_list = [2, 3, 5, 7, 9]
    fuzzyfication_list = [1, 2, 3, 4]

#    window_size = int(argv[1])
#    step_size = int(argv[2])
    fusion_coefficient = 1

    # clustering parameters
#    n_centers = int(argv[3])
    max_iter = 1000
    error = 0.005
#    fuzzyfication = int(argv[4])

#    autocorr_n_centers = int(argv[5])
    autocorr_max_iter = 1000
    autocorr_error = 0.005
#    autocorr_fuzzyfication = int(argv[6])

    loaded_data = load_data()

    parameters_gen = ((w_size, step_size, n_centers, fuzzyf, autocorr_n_centers, autocorr_fuzzyf) \
                      for w_size in window_size_list \
                      for step_size in step_size_list if step_size < w_size \
                      for n_centers in n_centers_list \
                      for fuzzyf in fuzzyfication_list \
                      for autocorr_n_centers in n_centers_list \
                      for autocorr_fuzzyf in fuzzyfication_list)

    for window_size, step_size, n_centers, fuzzyfication, autocorr_n_centers, autocorr_fuzzyfication in parameters_gen:
        print("\n\nparameters:\n\twindow_size: {}\n\tstep_size: {}\n\tn_centers: {}\n\tfuzzyfication: {}".format(window_size, step_size, n_centers, fuzzyfication))
        retval = run(loaded_data, window_size, step_size, n_centers, fuzzyfication, autocorr_n_centers, autocorr_fuzzyfication, fusion_coefficient, max_iter, error, autocorr_max_iter, autocorr_error)
        if not retval:
            print("Error with params: w_size: {}; step: {}; centers: {}; fuzzyf: {}; autoc_centers: {}; autoc_fuzzyf: {};".format(window_size, step_size, n_centers, fuzzyfication, autocorr_n_centers, autocorr_fuzzyfication))

def run(loaded_data, window_size, step_size, n_centers, fuzzyfication, autocorr_n_centers, autocorr_fuzzyfication, fusion_coefficient, max_iter, error, autocorr_max_iter, autocorr_error):
    tanks_list = [DataTypes.TANK1, DataTypes.TANK2, DataTypes.TANK3]
    tanks_names = ["tank1", "tank2", "tank3"]
    sensors_list = [DataTypes.OXYGEN, DataTypes.NITROGEN, DataTypes.SST, DataTypes.AMMONIA, DataTypes.VALVE, DataTypes.FLOW]
    sensors_names = ["oxygen", "nitrogen", "sst", "ammonia", "valve", "flow"]

    dump_file_name = ['./{}_{}_{}_{}_reshaped_data_window{}_step{}.dump'.format(n_centers, max_iter, error, fuzzyfication, window_size, step_size), \
                './{}_{}_{}_{}_centroids_window{}_step{}.dump'.format(n_centers, max_iter, error, fuzzyfication, window_size, step_size), \
                './{}_{}_{}_{}_reconstructed_window{}_step{}.dump'.format(n_centers, max_iter, error, fuzzyfication, window_size, step_size), \
                './{}_{}_{}_{}_anomaly_scores_window{}_step{}.dump'.format(n_centers, max_iter, error, fuzzyfication, window_size, step_size), \
                './{}_{}_{}_{}_auto_correlation_anomaly_scores_window{}_step{}.dump'.format(autocorr_n_centers, autocorr_max_iter, autocorr_error, autocorr_fuzzyfication, window_size, step_size), \
                './{}_{}_{}_{}_final_anomaly_scores_window{}_step{}_fusion_coefficient{}.dump'.format(autocorr_n_centers, autocorr_max_iter, autocorr_error, autocorr_fuzzyfication, window_size, step_size, fusion_coefficient)]

    try:
        with open(dump_file_name[0], 'rb') as f:
            #print("loading data...", end="")
            vectors, arrays, index_to_time = pickle.load(f)
            #print("done")
    except FileNotFoundError:
        #loaded_data = load_data()
        #print("reshape data, window_size: {} ; step_size: {}".format(window_size, step_size))
        """ arrays[sensor_id][tank_id][position_in_window][window_id]
        arrays is a list of sensor measures. It has a list for each sensor.
        each sensor has measures for each tank
        for each tank there are window_number samples
        each sample has window_size features. """
        arrays, index_to_time = get_np_arrays(loaded_data, window_size, step_size)

        features_number = arrays[0][0].shape[0]
        samples_number  = arrays[0][0].shape[1]

        assert(features_number == window_size)
        assert(samples_number == ((loaded_data.measures.size() - window_size) // step_size) + 1)

        # reshape data: concatenate as single vector the samples coming from different sensors at the same index.
        vectors = [np.empty((features_number * len(sensors_list), samples_number))]*3
        for tank_id in tanks_list:
            for sensor_id in sensors_list:
                for feature in range(0, features_number):
                    vectors[tank_id][sensor_id * features_number + feature] = \
                                                arrays[sensor_id][tank_id][feature]

        with open(dump_file_name[0], 'wb') as f:
            pickle.dump((vectors, arrays, index_to_time), f, pickle.HIGHEST_PROTOCOL)

    features_number = vectors[0].shape[0]
    samples_number  = vectors[0].shape[1]

    try:
        with open(dump_file_name[1], 'rb') as centroid_f, open(dump_file_name[2], 'rb') as reconstructed_f:
            #print("loading centroids and reconstructed data...", end="")
            centroids = pickle.load(centroid_f)
            reconstructed_data = pickle.load(reconstructed_f)
            #print("done")
    except FileNotFoundError:
        #print("computing centroids and reconstructed data...", end="")
        centroids = [np.empty(features_number)]*len(tanks_list)
        reconstructed_data = [np.empty(vectors[0].shape)]*len(tanks_list)
        for tank_id in tanks_list:
            try:
                centroids[tank_id], reconstructed_data[tank_id] = \
                                    cluster_data(vectors[tank_id], n_centers,
                                                 max_iter, error, fuzzyfication)
            except ZeroDivisionError:
                print("--- WARNING division by 0 ---")
                return False
        #print("done")
        with open(dump_file_name[1], 'wb') as centroid_f, open(dump_file_name[2], 'wb') as reconstructed_f:
            pickle.dump(centroids, centroid_f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(reconstructed_data, reconstructed_f, pickle.HIGHEST_PROTOCOL)

    try:
        with open(dump_file_name[3], 'rb') as scores_f:
            #print("loading anomaly scores...", end="")
            anomaly_score = pickle.load(scores_f)
            #print("done")
    except FileNotFoundError:
        anomaly_score = [np.empty(samples_number)]*len(tanks_list)
        for tank_id in tanks_list:
            for index in range(0, samples_number):
                anomaly_score[tank_id][index] = \
                        np.linalg.norm(vectors[tank_id][:,index] - reconstructed_data[tank_id][:,index])

        with open(dump_file_name[3], 'wb') as scores_f:
            pickle.dump((anomaly_score), scores_f, pickle.HIGHEST_PROTOCOL)

    for tank_id in tanks_list:
        anomaly_mean = np.mean(anomaly_score[tank_id])
        anomaly_var = np.var(anomaly_score[tank_id])
        anomaly_score[tank_id] = (anomaly_score[tank_id] - anomaly_mean) / math.sqrt(anomaly_var)

    # SECOND PART: compute distances based on auto-correlation.
    #print("\n\n--------- AUTO CORRELATION ---------")
    try:
        with open(dump_file_name[4], 'rb') as f:
            #print("auto-correlation anomaly scores...", end="")
            auto_correlation_centroids, auto_correlation_reconstructed_data, \
                auto_correlation_vectors, auto_correlation_anomaly_score = pickle.load(f)
            #print("done")

    except FileNotFoundError:
        auto_correlation_centroids = [np.empty(features_number)]*len(tanks_list)
        auto_correlation_reconstructed_data = [np.empty(vectors[0].shape)]*len(tanks_list)

        auto_correlation_vectors = compute_autocorrelation_vectors(arrays)
        for tank_id in tanks_list:
            try:
                auto_correlation_centroids[tank_id], auto_correlation_reconstructed_data[tank_id] = \
                             cluster_data(auto_correlation_vectors[tank_id], autocorr_n_centers,
                                          autocorr_max_iter, autocorr_error, autocorr_fuzzyfication)
            except ZeroDivisionError:
                print("--- WARNING division by 0 ---")
                return False

            auto_correlation_anomaly_score = [np.empty(samples_number)]*len(tanks_list)
        for tank_id in tanks_list:
            for index in range(0, samples_number):
                auto_correlation_anomaly_score[tank_id][index] = \
                     np.linalg.norm(auto_correlation_vectors[tank_id][:,index] - \
                                    auto_correlation_reconstructed_data[tank_id][:,index])

        for tank_id in tanks_list:
            anomaly_mean = np.mean(auto_correlation_anomaly_score[tank_id])
            anomaly_var = np.var(auto_correlation_anomaly_score[tank_id])
            auto_correlation_anomaly_score[tank_id] = \
                    (auto_correlation_anomaly_score[tank_id] - anomaly_mean) / \
                    math.sqrt(anomaly_var)

        with open(dump_file_name[4], 'wb') as auto_correlation_scores_f:
            pickle.dump((auto_correlation_centroids, auto_correlation_reconstructed_data, \
                         auto_correlation_vectors, auto_correlation_anomaly_score), \
                        auto_correlation_scores_f, pickle.HIGHEST_PROTOCOL)

    try:
        with open(dump_file_name[5], 'rb') as f:
            #print("loading final scores...", end="")
            final_anomaly_scores = pickle.load(f)
            #print("done")
    except FileNotFoundError:
        final_anomaly_scores = [np.empty(samples_number)]*len(tanks_list)
        for tank in tanks_list:
            tank_min = float('nan')
            tank_max = float('nan')
            for sample in range(0, samples_number):
                final_anomaly_scores[tank][sample] = (anomaly_score[tank][sample] * fusion_coefficient + \
                     auto_correlation_anomaly_score[tank][sample]) / (fusion_coefficient + 1)
                tank_min = min(final_anomaly_scores[tank][sample], tank_min)
                tank_max = max(final_anomaly_scores[tank][sample], tank_max)
            for sample in range(0, samples_number):
                final_anomaly_scores[tank][sample] = (final_anomaly_scores[tank][sample] - tank_min) / (tank_max - tank_min)

        with open(dump_file_name[5], 'wb') as final_scores_f:
            pickle.dump(final_anomaly_scores, final_scores_f, pickle.HIGHEST_PROTOCOL)

    #final_ranking = [np.flip(np.argsort(final_anomaly_scores[tank]), axis=0) for tank in tanks_list]
    return True


if __name__ == "__main__":
    main(sys.argv)
