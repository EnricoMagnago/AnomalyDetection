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
    if len(argv) != 4 and len(argv) != 5:
        print("usage: {} window_size window_step anomalies_score_pickle_file <decision threshold>".format(argv[0]))
        exit(1)
    threshold = float(argv[4]) if len(argv) > 4 else 0.5
    window_size = int(argv[1])
    window_step = int(argv[2])
    anomaly_scores_file = argv[3]

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

    print("-- confusion matrices:\n\t{}\t\t{}\n\t{}\t{}".format("corr_anom", "norm_class_as_anom", "anom_class_an_norm", "corr_norm"))
    for tank_id in range(0, len(window_anomalies)):
        print("\ntank: {}".format(tank_id))
        print("  {}\t{}\n  {}\t{}".format(confusion_matrix[tank_id][0][0], confusion_matrix[tank_id][0][1], \
                                          confusion_matrix[tank_id][1][0], confusion_matrix[tank_id][1][1]))





if __name__ == "__main__":
    main(sys.argv)
