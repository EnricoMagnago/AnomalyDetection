#!/usr/bin/python3

import sys
import os
import argparse
# add path to find modules.
sys.path.insert(0, "./data_import/python_interface")

import DataTypes, DataLoader
import numpy as np
import skfuzzy as fuzz # requires scikit-fuzzy
import time
import math
import pickle

def set_parser():
    #Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_dir_sax', type = str, default= '../data_score/sax/', help = "Directory of scores files")
    parser.add_argument('--results_dir', type = str, default= '../results_files/', help = "Directory of results files")
    parser.add_argument('--group_anomalies_dir', type = str, default= '../group_anomalies/', help = "Directory of results")
    parser.add_argument('--threshold', type = float, default= 0.5 , help = "Decision threshold")
    parser.add_argument('--expected_nw', type = float, default = 0.5, help = "Expected_normal_weight")
    parser.add_argument('--expected_aw', type = float, default = 0.5, help = "Expected_anomaly_weight")
    
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed

def create_directory(path_directory):
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)


def do_computations(window_size, window_step, anomaly_scores_file, file_r, FLAGS, loaded_data):
   
    try:
        with open(anomaly_scores_file, 'rb') as f:
            anomaly_scores = pickle.load(f)
            #print(anomaly_scores[1])

    except:
        print("could not open `{}`".format(anomaly_scores_file))
        exit(1)

    try:
        with open(FLAGS.group_anomalies_dir+"anomalies_group_by_window_{}_step{}.dump".format(window_size, window_step), 'rb') as f:
            window_anomalies = pickle.load(f)
    except FileNotFoundError:
        window_anomalies = group_anomalies(loaded_data, window_size, window_step)
        with open(FLAGS.group_anomalies_dir+"anomalies_group_by_window_{}_step{}.dump".format(window_size, window_step), 'wb') as f:
            pickle.dump(window_anomalies, f, pickle.HIGHEST_PROTOCOL)

    # compute confusion matrices.
    confusion_matrix = confusion_matrices(anomaly_scores, window_anomalies, FLAGS.threshold)

    file_r.write("RS\n")
    # dump confusion matrices.
    print("-- confusion matrices:\n\t{}\t\t{}\n\t{}\t{}".format("correct anomaly", "expected normal", "expected anomaly", "correct normal"))
    for tank_id in range(0, len(window_anomalies)):
        print("\ntank: {}".format(tank_id + 1))
        print("  {}\t{}\n  {}\t{}".format(confusion_matrix[tank_id][0][0], confusion_matrix[tank_id][0][1], \
                                          confusion_matrix[tank_id][1][0], confusion_matrix[tank_id][1][1]))


    errors = \
      weighted_mean_squared_error(anomaly_scores, window_anomalies, \
                                  FLAGS.expected_nw, FLAGS.expected_aw, \
                                  threshold=FLAGS.threshold)

    print("\n\nerrors, threshold: {}; error expected normal: {}; error expected anomaly: {}".format(FLAGS.threshold, FLAGS.expected_nw, FLAGS.expected_aw))
    for tank_id in range(0, len(window_anomalies)):
        print("error for tank {} := l1: {}; l2: {}".format(tank_id + 1, errors[tank_id][0], errors[tank_id][1]))
        file_r.write("tank "+str(tank_id)+"\n")
        file_r.write("ca "+str(confusion_matrix[tank_id][0][0])+"\n")
        file_r.write("en "+str(confusion_matrix[tank_id][0][1])+"\n")
        file_r.write("ea "+str(confusion_matrix[tank_id][1][0])+"\n")
        file_r.write("cn "+str(confusion_matrix[tank_id][1][1])+"\n")
        file_r.write("l1 "+str(errors[tank_id][0])+"\n")
        file_r.write("l2 "+str(errors[tank_id][1])+"\n")

def load_data():
    retval = DataTypes.Data()
    loader = DataLoader.DataLoader("./dataset/")
    #loader.load_subset(retval, 1000,100)
    loader.load_all(retval)
    return retval

def confusion_matrices(anomaly_scores, window_anomalies, threshold):
    assert(len(window_anomalies) == len(anomaly_scores) == 3) # tanks number
    assert(len(window_anomalies[0]) == len(anomaly_scores[0]))
    #print(anomaly_scores[0])
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
            #print("tid " + str(tank_id)+ "- wid "+ str(window_id))
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


def main(argv):
    FLAGS, _ = set_parser()
    create_directory(FLAGS.results_dir)
    create_directory(FLAGS.group_anomalies_dir)
    

    if FLAGS.threshold >= 1 or FLAGS.threshold <= 0:
        print("threshold must in (0; 1)")
        exit(1)

    #results_path = FLAGS.results_dir+"decision-threshold_"+str(FLAGS.threshold)+"/expected-normal-weight_"+str(FLAGS.expected_nw)+ \
    #"/expected-anomaly-weight_"+str(FLAGS.expected_aw)+"/"

    #counter used to name filed
    file_counter = 0


        
    #declare thresholds to use
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    windows_size = [25,50,75]
    steps_size = [40,70]
    sensors = [0,1,2,3,4,5]
    alphabets = [3,5]
    substring_size = [5,7]
    paa_size = [10,30]
    prjs_size = [2,3]
    prjs_iter = [3]
    freq_thresholds = [0.7]
    anomaly_thresholds = [0.15]

    loaded_data = load_data()

    for threshold in thresholds:
        FLAGS.threshold = threshold

        path_result_files = FLAGS.results_dir + "/sax/anomaly-threshold_"+str(FLAGS.threshold)+"/expected-nw_"+str(FLAGS.expected_nw)+ \
            "/expected-aw_"+str(FLAGS.expected_aw)+"/"
        
        for window_size in windows_size:
            for step_size in steps_size:
                for sensor in sensors:
                    for alphabet in alphabets:
                        for substring in substring_size:
                            for pa in paa_size:
                                for prj_size in prjs_size:
                                    for prj_iter in prjs_iter:
                                        for freq_threshold in freq_thresholds:
                                            for anomaly_threshold in anomaly_thresholds:
                                                score_file_path = FLAGS.scores_dir_sax+"window_"+str(window_size)+"/step_"+str(step_size)+ \
                                                    "/sensor"+str(sensor)+"_alphabet"+str(alphabet)+"_substring"+str(substring)+"_paa"+str(pa)+\
                                                    "_prjsize"+str(prj_size)+"_prjiter"+str(prj_iter)+"_freqthr"+str(freq_threshold)+\
                                                    "_anomalythr"+str(anomaly_threshold)+".dump"
                                                    
                                                path_result_file = path_result_files+"result_"+str(file_counter)+".txt" 

                                                if os.path.isfile(score_file_path):
                                                    print(path_result_file)
                                                    os.makedirs(os.path.dirname(path_result_file), exist_ok=True)
                                                    file_r= open(path_result_file, 'w')
                                                    file_r.write("PG\n")
                                                    file_r.write("decisionthr "+ str(FLAGS.threshold)+"\n")
                                                    file_r.write("expected_nw " + str(FLAGS.expected_nw)+"\n")
                                                    file_r.write("expected_aw " + str(FLAGS.expected_aw)+"\n")
                                                    file_r.write("PA\n")
                                                    file_r.write("windowsize "+ str(window_size)+"\n")
                                                    file_r.write("stepsize "+ str(step_size)+"\n")
                                                    file_r.write("sensor "+ str(sensor)+"\n")
                                                    file_r.write("alphabet "+ str(alphabet)+"\n")
                                                    file_r.write("substring "+ str(substring)+"\n")
                                                    file_r.write("paa "+ str(pa)+"\n")
                                                    file_r.write("prj_size "+ str(prj_size)+"\n")
                                                    file_r.write("prj_iter "+ str(prj_iter)+"\n")
                                                    file_r.write("freq_threshold "+ str(freq_threshold)+"\n")
                                                    file_r.write("anomaly_threshold "+ str(anomaly_threshold)+"\n")
                                                    
                                                    #results_path_tmp = results_path+window_folder+"/"+step_folder+"/"
                                                    
                                                    #print("WINDOW "+window_split[1]+" STEP "+step_split[1])
                                                    do_computations(window_size, step_size, score_file_path, file_r, FLAGS, loaded_data)
                                                    file_r.close()
                                                    file_counter = file_counter + 1


                                                
                                                






if __name__ == "__main__":
    main(sys.argv)
