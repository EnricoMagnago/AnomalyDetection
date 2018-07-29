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


def gen_configuration(window_size_list=window_size_list, step_size_list=step_size_list, \
                      threshold_list=threshold_list,
                      n_centers_list=n_centers_list, fuzzyfication_list=fuzzyfication_list,\
                      fusion_coefficient_list=fusion_coefficient_list):
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
    # row, column [0][0]: correctly classified anomalies (TP)
    #             [0][1]: classified as anomaly but was not (FP)
    #             [1][0]: classified as normal but was anomaly (FN)
    #             [1][1]: correctly classified normal (TN)
    # accuracy:  correctly classified / total
    # precision: correct_anomalies / expected_anomalies
    # recall:    correct_anomalies / predicted_anomalies
    accuracy = confusion_matrix[0][0] + confusion_matrix[1][1]
    accuracy = accuracy / (accuracy + confusion_matrix[0][1] + confusion_matrix[1][0])
    # TP / (TP + FP)
    precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    # TP / (TP + FN)
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
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

    figure_id = 0
    legend = ['tank1', 'tank2', 'tank3']

    x = range(0, configurations_number)
    figure_id += 1
    acc_figure = plt.figure(figure_id)
    plt.title("Accuracy for each configuration")
    for tank_id in range(0, 3):
        plt.scatter(x, accuracies[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    plt.ylim(0, 1)
    plt.legend(legend, loc='lower left', bbox_to_anchor=(0.9, 0.9))
    plt.xlabel('configuration id')
    plt.ylabel('accuracy')
    plt.show()

    figure_id += 1
    prec_figure = plt.figure(figure_id)
    plt.title("Precision for each configuration")
    for tank_id in range(0, 3):
        plt.scatter(x, precisions[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    plt.ylim(0, 1)
    plt.legend(legend, loc='lower left', bbox_to_anchor=(0.9, 0.9))
    plt.xlabel('configuration id')
    plt.ylabel('precision')
    plt.show()

    figure_id += 1
    recall_figure = plt.figure(figure_id)
    plt.title("Recall for each configuration")
    for tank_id in range(0, 3):
        plt.scatter(x, recalls[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    plt.ylim(0, 1)
    plt.legend(legend, loc='lower left', bbox_to_anchor=(0.9, 0.9))
    plt.xlabel('configuration id')
    plt.ylabel('recall')
    plt.show()

    # figure_id += 1
    # L1_figure = plt.figure(figure_id)
    # plt.title("L1 distance for each configuration")
    # for tank_id in range(0, 3):
    #     plt.scatter(x, l1s[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    # plt.legend(['tank1', 'tank2', 'tank3'])
    # plt.xlabel('configuration id')
    # plt.ylabel('L1 distance')
    # plt.show()

    # figure_id += 1
    # L2_figure = plt.figure(figure_id)
    # plt.title("L2 distance for each configuration")
    # for tank_id in range(0, 3):
    #     plt.scatter(x, l1s[tank_id], s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
    # plt.legend(['tank1', 'tank2', 'tank3'])
    # plt.xlabel('configuration id')
    # plt.ylabel('L2 distance')
    # plt.show()

def plots_for_clusters(evaluations, tank_id=1):
    line_fmt = ['r+:', 'rx:', 'r*:', 'g+:', 'gx:', 'g*:', 'b+:', 'bx:', 'b*:']
    legend_format = "w: {}, s: {},"
    legend = []
    figure_id = 0
    window_sizes = [40, 80, 100]
    step_sizes = [6, 16, 28]
    threshold = 0.2
    fuzzyfication = 2
    fusion = 0.5
    acc_scores = [[] for i in range(0, len(window_sizes)*len(step_sizes))]
    prec_scores = [[] for i in range(0, len(window_sizes)*len(step_sizes))]
    recall_scores = [[] for i in range(0, len(window_sizes)*len(step_sizes))]
    evaluation = evaluations[tank_id]
    for window_index, window_size in enumerate(window_sizes):
        for step_index, step_size in enumerate(step_sizes):
            index = window_index * len(step_sizes) + step_index
            for cluster_index, config in enumerate(gen_configuration(window_size_list=[window_size], step_size_list=[step_size], \
                                                                     threshold_list=[threshold],
                                                                     fuzzyfication_list=[fuzzyfication], fusion_coefficient_list=[fusion])):
                assert 0 <= cluster_index < len(n_centers_list), "index: {}, max: {}, config: {}".format(cluster_index, len(n_centers_list), config)
                confusion_matrix, _ = get_evaluation(evaluation, *config)
                acc, prec, recall = compute_accuracy_precision_recall(confusion_matrix)
                acc_scores[index].append(acc)
                prec_scores[index].append(prec)
                recall_scores[index].append(recall)
            legend.append(legend_format.format(window_size, step_size))

    x = n_centers_list
    for scores, name in zip([acc_scores, prec_scores, recall_scores], ["Accuracy", "Precision", "Recall"]):
        assert len(scores) == len(line_fmt) == len(legend)
        figure_id += 1
        clusters_figure = plt.figure(figure_id)
        plt.title("{} for different number of clusters in tank: {}".format(name, tank_id + 1))
        for index, score in enumerate(scores):
            plt.plot(x, score, line_fmt[index])
        plt.xlim(n_centers_list[0] - 0.2, n_centers_list[-1] + 0.7)
        plt.legend(legend, loc='lower left', bbox_to_anchor=(0.82, 0.40))
        plt.xlabel('number of clusters')
        plt.ylabel('{}'.format(name.lower()))
        plt.show()

def plots_for_windows(evaluations, tank_id=1):
    line_fmt = ['r+:', 'rx:', 'r*:', 'g+:', 'gx:', 'g*:', 'b+:', 'bx:', 'b*:']
    legend_format = "s: {}, c: {}"
    legend = []
    figure_id = 0
    n_clusters = n_centers_list
    step_sizes = [6, 16, 28]
    threshold = 0.2
    fuzzyfication = 2
    fusion = 0.5
    acc_scores = [[] for i in range(0, len(step_sizes)*len(n_clusters))]
    prec_scores = [[] for i in range(0, len(step_sizes)*len(n_clusters))]
    recall_scores = [[] for i in range(0, len(step_sizes)*len(n_clusters))]
    evaluation = evaluations[tank_id]
    for step_index, step_size in enumerate(step_sizes):
        for cluster_index, n_cluster in enumerate(n_clusters):
            index = step_index * len(n_clusters) + cluster_index
            for window_index, config in enumerate(gen_configuration(step_size_list=[step_size], n_centers_list=[n_cluster], \
                                                                    threshold_list=[threshold],
                                                                    fuzzyfication_list=[fuzzyfication], fusion_coefficient_list=[fusion])):
                assert 0 <= window_index < len(window_size_list), "index: {}, max: {}, config: {}".format(window_index, len(window_size_list), config)
                confusion_matrix, _ = get_evaluation(evaluation, *config)
                acc, prec, recall = compute_accuracy_precision_recall(confusion_matrix)
                acc_scores[index].append(acc)
                prec_scores[index].append(prec)
                recall_scores[index].append(recall)
            legend.append(legend_format.format(step_size, n_cluster))

    x = window_size_list
    for scores, name in zip([acc_scores, prec_scores, recall_scores], ["Accuracy", "Precision", "Recall"]):
        assert len(scores) == len(line_fmt) == len(legend)
        figure_id += 1
        clusters_figure = plt.figure(figure_id)
        plt.title("{} for different window sizes in tank: {}".format(name, tank_id + 1))
        for index, score in enumerate(scores):
            plt.plot(x, score, line_fmt[index])
        plt.xlim(window_size_list[0] - 2 , window_size_list[-1] + 12)
        plt.legend(legend, loc='lower left', bbox_to_anchor=(0.85, 0.40))
        plt.xlabel('window size')
        plt.ylabel('{}'.format(name.lower()))
        plt.show()

def plots_for_steps(evaluations, tank_id=1):
    line_fmt = ['r+:', 'rx:', 'r*:', 'g+:', 'gx:', 'g*:', 'b+:', 'bx:', 'b*:']
    legend_format = "w: {}, c: {}"
    legend = []
    figure_id = 0
    n_clusters = n_centers_list
    window_sizes = [40, 80, 100]
    threshold = 0.2
    fuzzyfication = 2
    fusion = 0.5
    acc_scores = [[] for i in range(0, len(window_sizes)*len(n_clusters))]
    prec_scores = [[] for i in range(0, len(window_sizes)*len(n_clusters))]
    recall_scores = [[] for i in range(0, len(window_sizes)*len(n_clusters))]
    evaluation = evaluations[tank_id]
    for window_index, window_size in enumerate(window_sizes):
        for cluster_index, n_cluster in enumerate(n_clusters):
            index = window_index * len(n_clusters) + cluster_index
            for step_index, config in enumerate(gen_configuration(window_size_list=[window_size], n_centers_list=[n_cluster], \
                                                                  threshold_list=[threshold],
                                                                  fuzzyfication_list=[fuzzyfication], fusion_coefficient_list=[fusion])):
                assert 0 <= step_index < len(step_size_list), "index: {}, max: {}, config: {}".format(step_index, len(step_size_list), config)
                confusion_matrix, _ = get_evaluation(evaluation, *config)
                acc, prec, recall = compute_accuracy_precision_recall(confusion_matrix)
                acc_scores[index].append(acc)
                prec_scores[index].append(prec)
                recall_scores[index].append(recall)
            legend.append(legend_format.format(window_size, n_cluster))

    x = step_size_list
    for scores, name in zip([acc_scores, prec_scores, recall_scores], ["Accuracy", "Precision", "Recall"]):
        assert len(scores) == len(line_fmt) == len(legend)
        figure_id += 1
        clusters_figure = plt.figure(figure_id)
        plt.title("{} for different step sizes in tank: {}".format(name, tank_id + 1))
        for index, score in enumerate(scores):
            plt.plot(x, score, line_fmt[index])
        plt.xlim(step_size_list[0] - 2 , step_size_list[-1] + 5)
        plt.legend(legend, loc='lower left', bbox_to_anchor=(0.85, 0.40))
        plt.xlabel('step size')
        plt.ylabel('{}'.format(name.lower()))
        plt.show()



def main(argv):
    if len(argv) !=  2:
        print("usage: {} evaluations_file")
        exit(1)

    with open(argv[1], 'rb') as f:
        evaluations = pickle.load(f)

    #plots_for_each_config(evaluations)

    # for tank in range(0, 3):
    #     plots_for_clusters(evaluations, tank)

    # for tank in range(0, 3):
    #     plots_for_windows(evaluations, tank)

    for tank in range(0, 3):
        plots_for_steps(evaluations, tank)

if __name__ == "__main__":
    main(sys.argv)
