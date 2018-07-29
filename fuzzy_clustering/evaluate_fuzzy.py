#!/usr/bin/python3

import sys
# add path to find modules.
sys.path.insert(0, "../data_import/python_interface")

import DataTypes, DataLoader
import os
import numpy as np
import skfuzzy as fuzz # requires scikit-fuzzy
import time
import math
import pickle
import re

def load_data():
    retval = DataTypes.Data()
    loader = DataLoader.DataLoader("../dataset/")
    #loader.load_subset(retval, 1000)
    loader.load_all(retval)
    return retval

def confusion_matrices(anomaly_scores, window_anomalies, threshold):
    assert(len(window_anomalies) == len(anomaly_scores) == 3) # tanks number
    assert(len(window_anomalies[0]) == len(anomaly_scores[0]))
    # row, column [0][0]: correctly classified anomalies (TP)
    #             [0][1]: classified as anomaly but was not (FP)
    #             [1][0]: classified as normal but was anomaly (FN)
    #             [1][1]: correctly classified normal (TN)
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
    mean_errors = [(0, 0), (0, 0), (0, 0)]
    for tank_id in range(0, len(window_anomalies)):
        l1_error = 0
        l2_error = 0
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
                l1_error += (error_weight * threshold_distance)
                l2_error += (error_weight * (threshold_distance**2))
        mean_errors[tank_id] = (l1_error / (len(window_anomalies[tank_id]) * max(expected_normal, expected_anomaly)), \
                                math.sqrt(l2_error / (len(window_anomalies[tank_id]) * max(expected_normal, expected_anomaly))))

    return mean_errors


def group_anomalies(loaded_data, window_size, window_step):
    tanks_number = 3
    samples_number = loaded_data.measures.size()
    assert(samples_number > window_size)
    windows_number = ((samples_number - window_size) // window_step) + 1
    retval = [[0 for i in range(0, windows_number)] for i in range(0, tanks_number)] # each number represents the number of anomalies found in that window.
    # extract list of window_ids to which each samples belongs.
    map_sample_to_windows = [None for i in range(0, samples_number)]
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


def compute_scores(anomaly_scores, window_anomalies, threshold, expected_normal_weight, expected_anomaly_weight):
    confusion_matrix = confusion_matrices(anomaly_scores, window_anomalies, threshold)
    errors = \
      weighted_mean_squared_error(anomaly_scores, window_anomalies, \
                                  expected_normal_weight, expected_anomaly_weight, \
                                  threshold=threshold)
    retval = []

    assert(len(window_anomalies) == 3)

    for tank_id in range(0, len(window_anomalies)):
        accuracy = confusion_matrix[tank_id][0][0] + confusion_matrix[tank_id][1][1]
        accuracy = accuracy / (accuracy + confusion_matrix[tank_id][0][1] + confusion_matrix[tank_id][1][0])
        precision = confusion_matrix[tank_id][0][0] / (confusion_matrix[tank_id][0][0] + confusion_matrix[tank_id][0][1])
        recall = confusion_matrix[tank_id][0][0] / (confusion_matrix[tank_id][0][0] + confusion_matrix[tank_id][1][0])
        l1 = errors[tank_id][0]
        l2 = errors[tank_id][1]
        element = accuracy, precision, recall, l1, l2
        retval.append(element)


    return retval

def evaluate_predictions(window_anomalies, anomaly_scores, threshold, expected_normal_weight, expected_anomaly_weight):
    confusion_matrix_tank0, confusion_matrix_tank1, confusion_matrix_tank2 = \
                confusion_matrices(anomaly_scores, window_anomalies, threshold)
    error_tank0, error_tank1, error_tank2 = \
                weighted_mean_squared_error(anomaly_scores, window_anomalies, \
                                            expected_normal_weight, expected_anomaly_weight, \
                                            threshold=threshold)
    return (confusion_matrix_tank0, error_tank0), (confusion_matrix_tank1, error_tank1), (confusion_matrix_tank2, error_tank2)


def configs_from_file_name(name):
    pattern = re.compile(r'centers(?P<n_centers>[0-9]+(\.[0-9])?)_fuzzyfication(?P<fuzzyf>[0-9]+(\.[0-9])?)_fusion_coefficient(?P<fusion>[0-9]+(\.[0-9])?)\.dump')
    m = pattern.match(name)
    return int(m['n_centers']), float(m['fuzzyf']), float(m['fusion'])


def main(argv):
    if len(argv) != 2:
        print("usage: {} predictions_dir".format(argv[0]))
        exit(1)

    thresholds = [i / 10 for i in range(1, 10)]
    expected_normal_weight = 0.5
    expected_anomaly_weight = 0.5

    loaded_data = load_data()
    evaluations_tank0 = {}
    evaluations_tank1 = {}
    evaluations_tank2 = {}

    for window_dir in os.listdir(argv[1]):
        assert window_dir[:11] == "windowsize_", "found: {}".format(window_dir[:11])
        window_size = int(window_dir[11:]) #windowsize_
        print("window: {}".format(window_size))
        window_dir_path = os.path.join(argv[1], window_dir)
        for step_dir in os.listdir(window_dir_path):
            assert step_dir[:9] == "stepsize_"
            window_step = int(step_dir[9:])
            print("\tstep: {}".format(window_step))
            window_anomalies = group_anomalies(loaded_data, window_size, window_step)
            step_dir_path = os.path.join(window_dir_path, step_dir)
            for threshold in thresholds:
                for predictions_file in os.listdir(step_dir_path):
                    predictions_file = os.fsdecode(predictions_file)
                    if predictions_file.endswith(".dump"):
                        anomaly_scores_file = os.path.join(step_dir_path, predictions_file)
                        with open(anomaly_scores_file, 'rb') as f:
                         anomaly_scores = pickle.load(f)
                         config = (window_size, window_step, threshold, *configs_from_file_name(predictions_file))
                         assert (config not in evaluations_tank0) and \
                             (config not in evaluations_tank1) and \
                             (config not in evaluations_tank2), "config: {}".format(config)
                         evaluations_tank0[config], evaluations_tank1[config], evaluations_tank2[config] = \
                                evaluate_predictions(window_anomalies, anomaly_scores, \
                                                     threshold, expected_normal_weight, expected_anomaly_weight)


    with open('fuzzy_clustering_eval.dump', 'wb') as f:
        pickle.dump((evaluations_tank0, evaluations_tank1, evaluations_tank2), f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(sys.argv)
