#! /usr/bin/python3

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

window_size_list = [40, 60, 80, 100]
step_size_list = [6, 8, 12, 16, 20, 24, 28]
n_centers_list = [2, 3, 5]
fuzzyfication_list = [2, 3]
fusion_coefficient_list = [0.5, 1, 1.5]
threshold_list = [i / 10 for i in range(1, 10)]

tank_color = ['r', 'g', 'b']
tank_marker = ['x', 'o', 's']


def gen_configuration():
    for window in window_size_list:
        for step in step_size_list:
            for n_clusters in n_centers_list:
                for fuzzyf in fuzzyfication_list:
                    for fusion in fusion_coefficient_list:
                        for threshold in threshold_list:
                            yield (window, step, threshold, n_clusters, fuzzyf, fusion)

def dump_confusion_matrix(confusion_matrix):
    print("\t\tE_anomaly\tE_normal")
    print("C_anomaly\t{}\t{}".format(confusion_matrix[0][0], confusion_matrix[0][1]))
    print("C_normal\t{}\t{}".format(confusion_matrix[1][0], confusion_matrix[1][1]))

def compute_accuracy_precision_recall(confusion_matrix):
    # row, column [0][0]: correctly classified anomalies
    #             [0][1]: classified as anomaly but was not
    #             [1][0]: classified as normal but was anomaly
    #             [1][1]: correctly classified normal
    # accuracy:  correctly classified / total
    # precision: correct_anomalies / expected_anomalies
    # recall:    correct_anomalies / predicted_anomalies
    accuracy = confusion_matrix[0][0] + confusion_matrix[1][1]
    accuracy = accuracy / (accuracy + confusion_matrix[0][1] + confusion_matrix[1][0])
    precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    return accuracy, precision, recall

def get_evaluation(evaluations, window_size, step_size, threshold, n_clusters, fuzzyfication, fusion):
    key = (window_size, step_size, threshold, n_clusters, fuzzyfication, fusion)
    assert key in evaluations, "can not find key: {}".format(key)
    return evaluations[key]

def is_score_ge(a, b):
    a_score = a[0]
    b_score = b[0]
    for i in range(0, len(a_score)):
        if b_score[i] > a_score[i]:
            return False
    return True

def find_best(scores):
    best_scores = []
    for config in gen_configuration():
        confusion_matrix, (l1, l2) = get_evaluation(scores, *config)
        acc, prec, recall = compute_accuracy_precision_recall(confusion_matrix)
        curr_score = (acc, prec, recall, l1, l2), config
        to_be_added = True
        for score in best_scores:
            if is_score_ge(score, curr_score):
                to_be_added = False
        if to_be_added:
            best_scores.append(curr_score)
    return best_scores

def dump_best_scores(best_scores, indent=0):
    indent_str = ''
    for i in range(0, indent):
        indent_str += '\t'
    indent = indent_str

    if len(best_scores) == 0:
        print(indent + "None")
        return
    print(indent + "acc\tprec\trecall\tl1\tl2\t:\twindow\tstep\tthreshold\tn_clusters\tfuzzyf\tfusion")
    for score, config in best_scores:
        print(indent + "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}\t{4:.3f}\t:\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}".format(*score, *config))
    print()

def plots_for_each_config(evaluations):
    # plot scores for every configuration
    accuracies = [[] for tank in range(0, 3)]
    precisions = [[] for tank in range(0, 3)]
    recalls = [[] for tank in range(0, 3)]
    l1s = [[] for tank in range(0, 3)]
    l2s = [[] for tank in range(0, 3)]
    for tank_id, evaluation in enumerate(evaluations):
        for index, config in enumerate(gen_configuration()):
            confusion_matrix, (l1, l2) = get_evaluation(evaluation, *config)
            acc, prec, recall = compute_accuracy_precision_recall(confusion_matrix)
            accuracies[tank_id].append(acc)
            precisions[tank_id].append(prec)
            recalls[tank_id].append(recall)
            l1s[tank_id].append(l1)
            l2s[tank_id].append(l2)

    configurations_number = len(accuracies[0])

    x = range(0, configurations_number)

    acc_figure = plt.figure(1)
    plt.title("Accuracy for each configuration")
    for tank_id in range(0, 3):
        plt.scatter(x, accuracies[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    plt.ylim(0, 1)
    plt.legend(['tank1', 'tank2', 'tank3'])
    plt.xlabel('configuration id')
    plt.ylabel('accuracy')
    plt.show()

    prec_figure = plt.figure(2)
    plt.title("Precision for each configuration")
    for tank_id in range(0, 3):
        plt.scatter(x, precisions[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    plt.ylim(0, 1)
    plt.legend(['tank1', 'tank2', 'tank3'])
    plt.xlabel('configuration id')
    plt.ylabel('precision')
    plt.show()

    recall_figure = plt.figure(3)
    plt.title("Recall for each configuration")
    for tank_id in range(0, 3):
        plt.scatter(x, recalls[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    plt.ylim(0, 1)
    plt.legend(['tank1', 'tank2', 'tank3'])
    plt.xlabel('configuration id')
    plt.ylabel('recall')
    plt.show()

    L1_figure = plt.figure(4)
    plt.title("L1 distance for each configuration")
    for tank_id in range(0, 3):
        plt.scatter(x, l1s[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    plt.legend(['tank1', 'tank2', 'tank3'])
    plt.xlabel('configuration id')
    plt.ylabel('L1 distance')
    plt.show()

    L2_figure = plt.figure(5)
    plt.title("L2 distance for each configuration")
    for tank_id in range(0, 3):
        plt.scatter(x, l1s[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    plt.legend(['tank1', 'tank2', 'tank3'])
    plt.xlabel('configuration id')
    plt.ylabel('L2 distance')
    plt.show()



def main(argv):
    if len(argv) !=  2:
        print("usage: {} evaluations_file")
        exit(1)

    with open(argv[1], 'rb') as f:
        evaluations = pickle.load(f)

    plots_for_each_config(evaluations)


    best = find_best(evaluations[0])
    dump_best_scores(best)

if __name__ == "__main__":
    main(sys.argv)
