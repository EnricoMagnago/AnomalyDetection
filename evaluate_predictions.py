#!/usr/bin/python3

import sys
# add path to find modules.
sys.path.insert(0, "./data_import/python_interface")

import DataTypes, DataLoader
import numpy as np
import skfuzzy as fuzz # requires scikit-fuzzy
import time
import math
import pickle


def load_data():
    retval = DataTypes.Data()
    loader = DataLoader.DataLoader("./dataset/")
    #loader.load_subset(retval, 1000)
    loader.load_all(retval)
    return retval

def confusion_matrices(anomaly_scores, window_anomalies, threshold):
    assert(len(window_anomalies) == len(anomaly_scores) == 3) # tanks number
    assert(len(window_anomalies[0]) == len(anomaly_scores[0]))
    # row, column [0][0]: correctly classified anomalies
    #             [0][1]: classified as anomaly but was not
    #             [1][0]: classified as normal but was anomaly
    #             [1][1]: correctly classified normal
    confusion_matrix = [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]

    for tank_id in range(0, len(window_anomalies)):
        for window_id in range(0, len(window_anomalies[tank_id])):
            # 1 if no anomaly, 0 otherwise
            expected = int(window_anomalies[tank_id][window_id] == 0)
            # 1 if not anomaly, 0 otherwise
            found = int(anomaly_scores[tank_id][window_id] < threshold)
            confusion_matrix[tank_id][expected][found] += 1
    return confusion_matrix

def weighted_mean_squared_error(anomaly_scores, window_anomalies, expected_normal, expected_anomaly, threshold):
    assert(expected_normal >= 0 and expected_anomaly >= 0)
    assert(expected_normal + expected_anomaly > 0)
    assert(0 < threshold < 1)
    mean_errors = [0, 0, 0]
    for tank_id in range(0, len(window_anomalies)):
        error = 0
        for window_id in range(0, len(window_anomalies[tank_id])):
            # 1 if anomaly, 0 if normal.
            expected = int(window_anomalies[tank_id][window_id] != 0)
            # in [0; 1]: 0 normal, 1 anomaly.
            found = anomaly_scores[tank_id][window_id]
            assert(0 <= found <= 1)
            # this is an error
            if expected != found >= threshold:
                # compute squared distance from threshold, normalize to get number between 0 and 1.
                threshold_distance = abs(found - threshold)
                # here we normalize with respect to the size of the interval of the error
                # if we expected 1, we predicted normal -> errors are in [0; threshold]
                # if we expected 0, we predicted anomaly -> errors are in [threshold; 1]
                threshold_distance /= (threshold if expected == 1 else 1 - threshold)
                error_weight = expected_normal if expected == 0 else expected_anomaly
                error += (error_weight * threshold_distance)
        mean_errors[tank_id] = error / (len(window_anomalies[tank_id]) * max(expected_normal, expected_anomaly))

    return mean_errors


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


def main(argv):
    if len(argv) < 4 or len(argv) > 5:
        print("usage: {} window_size window_step anomalies_score_pickle_file " \
              "<decision threshold> <expected_normal_weight> <expected_anomaly_weight>".format(argv[0]))
        exit(1)
    threshold = float(argv[4]) if len(argv) > 4 else 0.5
    window_size = int(argv[1])
    window_step = int(argv[2])
    anomaly_scores_file = argv[3]
    expected_normal_weight = float(argv[5]) if len(argv) > 5 else 0.4
    expected_anomaly_weight = float(argv[6]) if len(argv) > 6 else 0.6

    if threshold >= 1 or threshold <= 0:
        print("threshold must in (0; 1)")
        exit(1)

    try:
        with open(anomaly_scores_file, 'rb') as f:
            anomaly_scores = pickle.load(f)
    except:
        print("could not open `{}`".format(anomaly_scores_file))
        exit(1)

    try:
        with open("anomalies_group_by_window_{}_step{}.dump".format(window_size, window_step), 'rb') as f:
            window_anomalies = pickle.load(f)
    except FileNotFoundError:
        loaded_data = load_data()
        window_anomalies = group_anomalies(loaded_data, window_size, window_step)
        with open("anomalies_group_by_window_{}_step{}.dump".format(window_size, window_step), 'wb') as f:
            pickle.dump(window_anomalies, f, pickle.HIGHEST_PROTOCOL)

    # compute confusion matrices.
    confusion_matrix = confusion_matrices(anomaly_scores, window_anomalies, threshold)

    # dump confusion matrices.
    print("-- confusion matrices:\n\t{}\t\t{}\n\t{}\t{}".format("correct anomaly", "expected normal", "expected anomaly", "correct normal"))
    for tank_id in range(0, len(window_anomalies)):
        print("\ntank: {}".format(tank_id + 1))
        print("  {}\t{}\n  {}\t{}".format(confusion_matrix[tank_id][0][0], confusion_matrix[tank_id][0][1], \
                                          confusion_matrix[tank_id][1][0], confusion_matrix[tank_id][1][1]))


    errors = \
      weighted_mean_squared_error(anomaly_scores, window_anomalies, \
                                  expected_normal_weight, expected_anomaly_weight, \
                                  threshold=threshold)

    print("\n\nerrors, threshold: {}; error expected normal: {}; error expected anomaly: {}".format(threshold, expected_normal_weight, expected_anomaly_weight))
    for tank_id in range(0, len(window_anomalies)):
        print("error for tank {} : {}".format(tank_id + 1, errors[tank_id]))





if __name__ == "__main__":
    main(sys.argv)
