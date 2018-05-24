#!/usr/bin/python3

import sys
# add path to find modules.
sys.path.insert(0, "../data_import/python_interface")

import DataTypes, DataLoader
import numpy as np
import skfuzzy as fuzz # requires scikit-fuzzy
import time
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
    oxygen = [nan_array]*3
    nitrogen = [nan_array]*3
    sst = [nan_array]*3
    ammonia = [nan_array]*3
    flow = [nan_array]*3
    valve = [nan_array]*3
    for window_index in range(0, window_number):
        if window_index % 10000 == 0:
            print("window {}/{};".format(window_index, window_number))
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
                tank = data.measures[window_index][0][tank_id]
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
    print()
    index_to_time = [0] * data.index_to_time.size()
    for i in range(0, data.index_to_time.size()):
        index_to_time[i] = DataLoader.time_to_string(data, i)
    return [oxygen, nitrogen, sst, ammonia, valve, flow], index_to_time

def compute_autocorrelation_vectors(data):
    """ input data: 6 sensors * 3 tanks * (window_size * samples)
    input vector is a matrix with shape (window_size, N)
    output matrix has shape (features, N)
    we compute the auto-correlation coefficient for each sensor in each tank
    separately then we concatenate the features of the 6 sensor to get a single
    vector that represents the state of a tank in a particular instant. """
    sensors_number = len(data)
    tanks_number = len(data[0])
    window_length = data[0][0].shape[0]
    samples_number = data[0][0].shape[1]
    output = [np.empty((window_length * sensors_number, samples_number), dtype=np.float64)] * tanks_number
    for tank_id in range(0, tanks_number):
        for sensor_id in range(0, sensors_number):
            means = np.mean(data[sensor_id][tank_id], axis=0)
            for sample_index in range(0, samples_number):
                mean = means[sample_index]
                for curr_lag in (1, window_length):
                    output_feature_index = curr_lag - 1
                    variance = 0
                    for feature_id in range (0, window_length):
                        variance += ((data[sensor_id][tank_id][feature_id][sample_index] - mean) ** 2)
                    for feature_id in range (0, window_length - curr_lag):
                        lagged_feature_id = feature_id + curr_lag
                        output[tank_id][output_feature_index][sample_index] += \
                                ((data[sensor_id][tank_id][feature_id][sample_index] - mean) * \
                                (data[sensor_id][tank_id][lagged_feature_id][sample_index] - mean))

                    if variance > 0 and \
                       output[tank_id][output_feature_index][sample_index] / variance > np.finfo(np.float64).eps and \
                       output[tank_id][output_feature_index][sample_index] / variance < np.finfo(np.float64).max:
                        output[tank_id][output_feature_index][sample_index] /= variance
                    else:
                        output[tank_id][output_feature_index][sample_index] = -1
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
    tanks_list = [DataTypes.TANK1, DataTypes.TANK2, DataTypes.TANK3]
    tanks_names = ["tank1", "tank2", "tank3"]
    sensors_list = [DataTypes.OXYGEN, DataTypes.NITROGEN, DataTypes.SST, DataTypes.AMMONIA, DataTypes.VALVE, DataTypes.FLOW]
    sensors_names = ["oxygen", "nitrogen", "sst", "ammonia", "valve", "flow"]


    window_size = 20
    step_size = 5
    dump_file_name = ['./reshaped_data_window{}_step_{}.dump'.format(window_size, step_size), \
                      './centroids_window{}_step_{}.dump'.format(window_size, step_size), \
                      './reconstructed_window{}_step_{}.dump'.format(window_size, step_size), \
                      './anomaly_scores_window{}_step_{}.dump'.format(window_size, step_size)]

    try:
        with open(dump_file_name[0], 'rb') as f:
            print("loading data...", end="")
            vectors, arrays, index_to_time = pickle.load(f)
            print("done")
    except FileNotFoundError:
        loaded_data = load_data()
        print("reshape data, window_size: {} ; step_size: {}".format(window_size, step_size))
        arrays, index_to_time = get_np_arrays(loaded_data, window_size, step_size)

        features_number = arrays[0][0].shape[0]
        samples_number  = arrays[0][0].shape[1]

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

    # clustering parameters
    n_centers = 3
    max_iter = 1000
    error = 0.005
    fuzzyfication = 2
    try:
        with open(dump_file_name[1], 'rb') as centroid_f, open(dump_file_name[2], 'rb') as reconstructed_f:
            print("loading centroids and reconstructed data...", end="")
            centroids = pickle.load(centroid_f)
            reconstructed_data = pickle.load(reconstructed_f)
            print("done")
    except FileNotFoundError:
        print("computing centroids and reconstructed data...", end="")
        centroids = [np.empty(features_number)]*len(tanks_list)
        reconstructed_data = [np.empty(vectors[0].shape)]*len(tanks_list)
        for tank_id in tanks_list:
            centroids[tank_id], reconstructed_data[tank_id] = \
                        cluster_data(vectors[tank_id], n_centers,
                                     max_iter, error, fuzzyfication)
        print("done")
        with open(dump_file_name[1], 'wb') as centroid_f, open(dump_file_name[2], 'wb') as reconstructed_f:
            pickle.dump(centroids, centroid_f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(reconstructed_data, reconstructed_f, pickle.HIGHEST_PROTOCOL)

    try:
        with open(dump_file_name[3], 'rb') as scores_f:
            print("loading anomaly scores...", end="")
            anomaly_score, anomalies_ranking = pickle.load(scores_f)
            print("done")
    except FileNotFoundError:
        anomaly_score = [np.empty(samples_number)]*len(tanks_list)
        anomalies_ranking = [np.empty(samples_number)]*len(tanks_list)
        for tank_id in tanks_list:
            for index in range(0, samples_number):
                anomaly_score[tank_id][index] = \
                        np.linalg.norm(vectors[tank_id][:,index] - reconstructed_data[tank_id][:,index])

        with open(dump_file_name[3], 'wb') as scores_f:
            pickle.dump((anomaly_score, anomalies_ranking), scores_f, pickle.HIGHEST_PROTOCOL)

    for tank_id in tanks_list:
        anomaly_mean = np.mean(anomaly_score[tank_id])
        anomaly_var = np.var(anomaly_score[tank_id])
        anomaly_score[tank_id] = (anomaly_score[tank_id] - anomaly_mean) / anomaly_var
        anomalies_ranking[tank_id] = np.flip(np.argsort(anomaly_score[tank_id]), axis=0)

    for index in range(0, samples_number):
        tanks = [index_to_time[anomalies_ranking[tank][index]] for tank in tanks_list]
        print("{};\ttank1: {};\ttank2: {};\ttank3:{};".format(index, *tanks))


    # SECOND PART: compute distances based on auto-correlation.
    print("\n\n--------- AUTO CORRELATION ---------")
    auto_correlation_centroids = [np.empty(features_number)]*len(tanks_list)
    auto_correlation_reconstructed_data = [np.empty(vectors[0].shape)]*len(tanks_list)

    auto_correlation_vectors = compute_autocorrelation_vectors(arrays)
    auto_correlation_centroids[tank_id], auto_correlation_reconstructed_data[tank_id] = \
                        cluster_data(auto_correlation_vectors[tank_id], n_centers,
                                     max_iter, error, fuzzyfication)

    auto_correlation_anomaly_score = [np.empty(samples_number)]*len(tanks_list)
    auto_correlation_anomalies_ranking = [np.empty(samples_number)]*len(tanks_list)
    for tank_id in tanks_list:
        for index in range(0, samples_number):
            auto_correlation_anomaly_score[tank_id][index] = \
                        np.linalg.norm(auto_correlation_vectors[tank_id][:,index] - auto_correlation_reconstructed_data[tank_id][:,index])

    for tank_id in tanks_list:
        anomaly_mean = np.mean(auto_correlation_anomaly_score[tank_id])
        anomaly_var = np.var(auto_correlation_anomaly_score[tank_id])
        auto_correlation_anomaly_score[tank_id] = (auto_correlation_anomaly_score[tank_id] - anomaly_mean) / anomaly_var
        auto_correlation_anomalies_ranking[tank_id] = np.flip(np.argsort(auto_correlation_anomaly_score[tank_id]), axis=0)

    for index in range(0, samples_number):
        tanks = [index_to_time[auto_correlation_anomalies_ranking[tank][index]] for tank in tanks_list]
        print("{};\ttank1: {};\ttank2: {};\ttank3:{};".format(index, *tanks))

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
