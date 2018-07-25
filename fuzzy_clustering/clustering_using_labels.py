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
import pathlib

tanks_list = [DataTypes.TANK1, DataTypes.TANK2, DataTypes.TANK3]
tanks_names = ["tank1", "tank2", "tank3"]
sensors_list = [DataTypes.OXYGEN, DataTypes.NITROGEN, DataTypes.SST, DataTypes.AMMONIA, DataTypes.VALVE, DataTypes.FLOW]
sensors_names = ["oxygen", "nitrogen", "sst", "ammonia", "valve", "flow"]

def sigmoid(x):
    return 1/(1 + math.exp(-x))

def load_data():
    retval = DataTypes.Data()
    loader = DataLoader.DataLoader("../dataset/")
    #loader.load_subset(retval, 5000)
    loader.load_all(retval)
    return retval

def data_index_from_window_index(window_index, window_number, step_size):
    return (window_number * step_size) + window_index

def group_anomalies(loaded_data, window_size, window_step):
    tanks_number = 3
    samples_number = loaded_data.measures.size()
    assert(samples_number > window_size)
    windows_number = ((samples_number - window_size) // window_step) + 1
    retval = [[0] * windows_number] * tanks_number # each number represents the number of anomalies found in that window.
    # extract list of window_ids to which each samples belongs.
    map_sample_to_windows = [None] * samples_number
    for sample_id in range(0, samples_number):
        window_ids = []
        # for every possible position of the sample in the window:
        for window_index in range(0, min(sample_id, window_size), window_step):
            window_ids.append((sample_id - window_index) // window_step)
        map_sample_to_windows[sample_id] = window_ids
    anomaly_idx = 0
    for index in range(0, loaded_data.anomaly_indexes.size()):
        sample_id = loaded_data.anomaly_indexes[index][0]
        windows = map_sample_to_windows[sample_id]
        anomalies = DataTypes.AnomaliesList(loaded_data.anomaly_indexes[index][1])
        for anomaly_idx in range(0, anomalies.size()):
            anomaly = anomalies[anomaly_idx]
            for tank_id in range(0, anomaly.tanks.size()):
                tank = anomaly.tanks[tank_id]
                for window in windows:
                    retval[tank][window] += 1
    return retval

def get_np_arrays(data, window_size, step_size):
    assert(data.measures.size() > window_size)
    tanks_list = [DataTypes.TANK1, DataTypes.TANK2, DataTypes.TANK3]
    window_number = ((data.measures.size() - window_size) // step_size) + 1
    nan_array = np.empty((window_size, window_number))
    nan_array[:] = np.nan
    oxygen = [nan_array.copy() for i in range(0, len(tanks_list))]
    nitrogen = [nan_array.copy() for i in range(0, len(tanks_list))]
    sst = [nan_array.copy() for i in range(0, len(tanks_list))]
    ammonia = [nan_array.copy() for i in range(0, len(tanks_list))]
    flow = [nan_array.copy() for i in range(0, len(tanks_list))]
    valve = [nan_array.copy() for i in range(0, len(tanks_list))]
    for window_index in range(0, window_number):
        #if window_index % 10000 == 0:
            #print("window {}/{};".format(window_index, window_number))
        avg_oxygen = [0 for i in range(0, len(tanks_list))]
        oxygen_counter = [0 for i in range(0, len(tanks_list))]
        avg_nitrogen = [0 for i in range(0, len(tanks_list))]
        nitrogen_counter = [0 for i in range(0, len(tanks_list))]
        avg_sst = [0 for i in range(0, len(tanks_list))]
        sst_counter = [0 for i in range(0, len(tanks_list))]
        avg_ammonia = [0 for i in range(0, len(tanks_list))]
        ammonia_counter = [0 for i in range(0, len(tanks_list))]
        avg_flow = [0 for i in range(0, len(tanks_list))]
        flow_counter = [0 for i in range(0, len(tanks_list))]
        avg_valve = [0 for i in range(0, len(tanks_list))]
        valve_counter = [0 for i in range(0, len(tanks_list))]
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
    index_to_time = [0 for i in range(0, data.index_to_time.size())]
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
    output = [np.empty((sensors_number * window_length - 1, windows_number), dtype=np.float64) for i in range(0, tanks_number)]
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

def cluster_data(data, n_centers, max_iter, error, fuzzyfication, window_anomalies):
    centroids, u, u0, d, _, iterations, fpc = fuzz.cluster.cmeans(data,
                                                                 n_centers,
                                                                 fuzzyfication,
                                                                 error=error,
                                                                 maxiter=max_iter,
                                                                 init=None)
    assert u.shape[1] == len(window_anomalies), "u.shape[1] : {}; len(window_anomalies): {}".format(u.shape[1], len(window_anomalies))
    anomalies_per_cluster = [0 for i in range(0, n_centers)]
    for cluster_idx in range(0, n_centers):
        total = 0
        u_sum = np.sum(u[cluster_idx, :], 0)
        for sample_idx in range(0, u.shape[1]):
            anomalies_per_cluster[cluster_idx] = anomalies_per_cluster[cluster_idx] + (window_anomalies[sample_idx] * u[cluster_idx][sample_idx] / u_sum)
            total = total + window_anomalies[sample_idx]
        anomalies_per_cluster[cluster_idx] = anomalies_per_cluster[cluster_idx] / total

    normal_clusters = []
    anomaly_clusters = []
    for cluster_idx in range(0, n_centers):
        if sigmoid(anomalies_per_cluster[cluster_idx]) > 0.7:
            anomaly_clusters.append(cluster_idx)
        else:
            normal_clusters.append(cluster_idx)

    u = u ** fuzzyfication
    reconstructed = np.empty_like(data.T)
    anomaly_reconstructed = np.empty_like(data.T)
    for k in range(0, u.shape[1]):
        u_sum = 0
        for cluster_index in normal_clusters:
            u_sum = u_sum + u[cluster_index][k]
        anomaly_u_sum = 0
        for cluster_index in anomaly_clusters:
            anomaly_u_sum = anomaly_u_sum + u[cluster_index][k]
        anomaly_num = np.array([np.fmax((u[cluster_index, k] * centroids[cluster_index])/anomaly_u_sum, \
                                      np.finfo(np.float64).eps) \
                              for cluster_index in anomaly_clusters])
        numerator = np.array([np.fmax((u[cluster_index, k] * centroids[cluster_index])/u_sum, \
                                      np.finfo(np.float64).eps) \
                              for cluster_index in normal_clusters])
        reconstructed[k] = np.sum(numerator, 0)
        anomaly_reconstructed[k] = np.sum(numerator, 0)

    return centroids, reconstructed.T, anomaly_reconstructed.T

def compute_anomaly_score(vectors, n_centers, max_iter, error, fuzzyfication, window_anomalies):
    features_number = vectors[0].shape[0]
    samples_number  = vectors[0].shape[1]

    centroids = [np.empty(features_number) for i in range(0, len(tanks_list))]
    reconstructed_data = [np.empty(vectors[0].shape) for i in range(0, len(tanks_list))]
    anomaly_reconstructed_data = [np.empty(vectors[0].shape) for i in range(0, len(tanks_list))]
    for tank_id in tanks_list:
        try:
            centroids[tank_id], reconstructed_data[tank_id], anomaly_reconstructed_data[tank_id] = \
                                cluster_data(vectors[tank_id], n_centers,
                                             max_iter, error, fuzzyfication, window_anomalies[tank_id])
        except ZeroDivisionError:
            return None

    anomaly_score = [np.empty(samples_number) for i in range(0, len(tanks_list))]
    for tank_id in tanks_list:
        for index in range(0, samples_number):
            score = np.linalg.norm(vectors[tank_id][:,index] - reconstructed_data[tank_id][:,index])
            score = sigmoid(score)
            an_score = np.linalg.norm(vectors[tank_id][:,index] - anomaly_reconstructed_data[tank_id][:,index])
            an_score = sigmoid(an_score)
            anomaly_score[tank_id][index] = max(score, 1 - an_score)

    for tank_id in tanks_list:
        anomaly_mean = np.mean(anomaly_score[tank_id])
        anomaly_var = np.var(anomaly_score[tank_id])
        anomaly_score[tank_id] = \
                (anomaly_score[tank_id] - anomaly_mean) / math.sqrt(anomaly_var)

    return anomaly_score


def compute_autocorr_anomaly_score(auto_correlation_vectors, shape, autocorr_n_centers, autocorr_max_iter, autocorr_error, autocorr_fuzzyfication, window_anomalies):
    features_number = shape[0]
    samples_number  = shape[1]

    auto_correlation_centroids = [np.empty(features_number) for i in range(0, len(tanks_list))]
    auto_correlation_reconstructed_data = [np.empty(shape) for i in range(0, len(tanks_list))]
    auto_correlation_anomaly_reconstructed_data = [np.empty(shape) for i in range(0, len(tanks_list))]
    for tank_id in tanks_list:
       try:
           auto_correlation_centroids[tank_id], auto_correlation_reconstructed_data[tank_id], auto_correlation_anomaly_reconstructed_data[tank_id] = \
                         cluster_data(auto_correlation_vectors[tank_id], autocorr_n_centers,
                                      autocorr_max_iter, autocorr_error, autocorr_fuzzyfication, window_anomalies[tank_id])
       except ZeroDivisionError:
            return None

    auto_correlation_anomaly_score = [np.empty(samples_number) for i in range(0, len(tanks_list))]
    for tank_id in tanks_list:
        for index in range(0, samples_number):
            score = np.linalg.norm(auto_correlation_vectors[tank_id][:,index] - \
                                   auto_correlation_reconstructed_data[tank_id][:,index])
            score = sigmoid(score)
            an_score = np.linalg.norm(auto_correlation_vectors[tank_id][:,index] - \
                                   auto_correlation_anomaly_reconstructed_data[tank_id][:,index])
            an_score = sigmoid(an_score)
            auto_correlation_anomaly_score[tank_id][index] = max(score, 1 - an_score)

    for tank_id in tanks_list:
        anomaly_mean = np.mean(auto_correlation_anomaly_score[tank_id])
        anomaly_var = np.var(auto_correlation_anomaly_score[tank_id])
        auto_correlation_anomaly_score[tank_id] = \
                (auto_correlation_anomaly_score[tank_id] - anomaly_mean) / \
                math.sqrt(anomaly_var)

    return auto_correlation_anomaly_score

def dump_final_anomaly_scores(dump_file_name, anomaly_score, auto_correlation_anomaly_score, fusion_coefficient, samples_number):
    final_anomaly_scores = [np.empty(samples_number) for i in range(0, len(tanks_list))]
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

    with open(dump_file_name, 'wb') as final_scores_f:
        pickle.dump(final_anomaly_scores, final_scores_f, pickle.HIGHEST_PROTOCOL)
    return True


def main(argv):
    window_size_list = [40, 60, 80, 100]
    step_size_list = [6, 8, 12, 16, 20, 24, 28]
    n_centers_list = [2, 3, 5]
    fuzzyfication_list = [2, 3]

    fusion_coefficient_list = [0.5, 1, 1.5]

    max_iter = 1000
    error = 0.005

    loaded_data = load_data()

    for window_size in window_size_list:
        for step_size in step_size_list:
            dirname = './windowsize_{}/stepsize_{}/'.format(window_size, step_size)
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
            print("\nIn: {}\n\tnp_arrays...".format(dirname), end="", flush=True)
            arrays, _ = get_np_arrays(loaded_data, window_size, step_size)
            print("done", flush=True)

            print("\tload anomalies...", end="", flush=True)
            window_anomalies = group_anomalies(loaded_data, window_size, step_size)
            print("done", flush=True)

            features_number = arrays[0][0].shape[0]
            samples_number  = arrays[0][0].shape[1]

            assert(features_number == window_size)
            assert(samples_number == ((loaded_data.measures.size() - window_size) // step_size) + 1)

            print("\tcreate vectors...", end="", flush=True)
            # reshape data: concatenate as single vector the samples coming from different sensors at the same index.
            vectors = [np.empty((features_number * len(sensors_list), samples_number)) for i in range(0, 3)]
            for tank_id in tanks_list:
                for sensor_id in sensors_list:
                    for feature in range(0, features_number):
                        vectors[tank_id][sensor_id * features_number + feature] = \
                                                    arrays[sensor_id][tank_id][feature]

            auto_correlation_vectors = compute_autocorrelation_vectors(arrays)

            print ("done", flush=True)
            features_number = vectors[0].shape[0]
            samples_number  = vectors[0].shape[1]

            for n_centers in n_centers_list:
                for fuzzyfication in fuzzyfication_list:
                    print("\t{} centers; {} fuzzyfication...".format(n_centers, fuzzyfication), end="", flush=True)

                    anomaly_score = compute_anomaly_score(vectors, n_centers, max_iter, error, fuzzyfication, window_anomalies)
                    auto_correlation_anomaly_score = compute_autocorr_anomaly_score(auto_correlation_vectors, vectors[0].shape, \
                                                                                    n_centers, max_iter, error, fuzzyfication, window_anomalies)

                    if anomaly_score is None:
                        print("issue with anomaly scores")
                        continue
                    if auto_correlation_anomaly_score is None:
                        print("issue with autocorrelation anomaly scores")
                        continue

                    print("done", flush=True)
                    dump_file_name_format = dirname + 'centers{}_fuzzyfication{}_fusion_coefficient{}.dump'
                    for fusion_coefficient in fusion_coefficient_list:
                        name = dump_file_name_format.format(n_centers, fuzzyfication, fusion_coefficient)
                        dump_final_anomaly_scores(name, anomaly_score, \
                                                  auto_correlation_anomaly_score, \
                                                  fusion_coefficient, \
                                                  samples_number)

if __name__ == "__main__":
    main(sys.argv)
