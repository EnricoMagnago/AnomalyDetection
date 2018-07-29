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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines



tank_color = ['r', 'g', 'b']
tank_marker = ['x', 'o', 's']
sensors_names =['OXYGEN', 'NITRATE', 'SST', 'AMMONIA', 'VALVE', 'FLOW']
windows_list = [50, 75]
steps_list = [40,70]

def set_parser():
    #Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type = str, default= '../results_files/', help = "Directory of results files")
    parser.add_argument('--plots_dir_fuzzy', type = str, default= '../plots/fuzzy/', help = "Directory of fuzzy plots")
    parser.add_argument('--plots_dir_sax', type = str, default= '../plots/sax/', help = "Directory of sax plots")
    parser.add_argument('--technique', type = int, default= 0 , help = "Choose technique  {0 = fuzzy, 1 = sax})")
    

    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS, unparsed


def create_directory(path_directory):
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)

def create_list_map(*args):

    l_map = []
    for arg in args:
        l_map.append(arg)

    return l_map

                    

def load_configurations(FLAGS):


    
    path_technique  = FLAGS.results_dir + "sax/"


    #list containing files found in path_complete
    files_list = []
    #for each anomaly_threshold folder found
    for anomaly_threshold_folder in os.listdir(path_technique):
        path_tmp = path_technique + anomaly_threshold_folder+"/"
        #for expecter_nw_folder_found
        for expected_nw_folder in os.listdir(path_tmp):
            path_tmp_2 = path_tmp + expected_nw_folder+"/"
             #for expecter_nw_folder_found
            for expected_aw_folder in os.listdir(path_tmp_2):

                #complete path
                path_complete = path_tmp_2 + expected_aw_folder +"/"


                for file_name in os.listdir(path_complete):
                    file_r = open(path_complete+file_name, 'r')
                    
                    #process file
                    file_el = []
                    #tanks[i] contains the followinf list [correct_anomalies, expected_normal, expected_anomalies, correct_normal, l1, l2] of i-th tank
                    tanks = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]

                    index_tank = 0
                    ca = 0
                    en = 0
                    ea = 0
                    cn = 0
                    l1 = 0
                    l2 = 0

                    for line in file_r:
                        line_split = line.split(" ")
                        if line_split[0] == "decisionthr": 
                            file_el.append(float(line_split[1]))
                       
                        elif line_split[0] == "windowsize":
                            file_el.append(int(line_split[1]))
                        elif line_split[0] == "stepsize":
                            file_el.append(int(line_split[1]))
                        elif line_split[0] == "sensor":
                            file_el.append(int(line_split[1]))
                        elif line_split[0] == "alphabet":
                            file_el.append(int(line_split[1]))
                        elif line_split[0] == "substring":
                            file_el.append(int(line_split[1]))
                        elif line_split[0] == "paa":
                            file_el.append(int(line_split[1]))
                        elif line_split[0] == "prj_size":
                            file_el.append(int(line_split[1]))
                        elif line_split[0] == "prj_iter":
                            file_el.append(int(line_split[1]))
                        elif line_split[0] == "freq_threshold":
                            file_el.append(float(line_split[1]))
                        elif line_split[0] == "anomaly_threshold":
                            file_el.append(float(line_split[1]))
                        elif line_split[0] == "tank":
                            index_tank = int(line_split[1])
                        elif line_split[0] == "ca":
                            ca = float(line_split[1])
                            #print("ca "+ line_split[1])
                        elif line_split[0] == "en":
                            en = float(line_split[1])
                            #print("en "+ line_split[1])
                        elif line_split[0] == "ea":
                            ea = float(line_split[1])
                            #print("ea "+ line_split[1])
                        elif line_split[0] == "cn": 
                            cn = float(line_split[1])
                            #print("cn "+ line_split[1])
                        elif line_split[0] == "l1":
                            l1 = float(line_split[1])
                        elif line_split[0] == "l2":
                            l2 = float(line_split[1])
                            if (ca+cn+en+ea) == 0:
                                accuracy = 0
                            else:
                                accuracy = (ca+cn)/(ca+cn+en+ea) 
                            if (ca + en) == 0:
                                precision = 0
                            else:
                                precision = ca /(ca + en)
                            if (ca + ea) == 0:
                            	recall = 0
                            else:
                            	recall = ca /(ca + ea)

                            tanks[index_tank][0] = accuracy
                            tanks[index_tank][1] = precision
                            tanks[index_tank][2] = recall
                            tanks[index_tank][3] = l1
                            tanks[index_tank][4] = l2
                                
                    file_el.append(tanks)
                    #append the processed file to file_list
                    files_list.append(file_el)

               
    return files_list

def filter_configurations_by_sensor(configurations, sensor):

	filtered_configurations = []
	for configuration in configurations:
		if configuration[3] == sensor:
			filtered_configurations.append(configuration)

	return filtered_configurations	

def is_score_ge(a, b):
    a_score = a[0]
    b_score = b[0]
    for i in range(0, len(a_score)):
        if b_score[i] > a_score[i]:
            return False
    return True

def find_best(configurations, sensor):
    file = open('sensor'+str(sensor)+'.txt','w') 
    for tank_id in range(3):
        file.write("\n\nTANK_"+str(tank_id))
        best_scores = []
        for config in configurations:
            evaluation = config[-1]
            config_s = config[0:11]
            curr_score = (evaluation[tank_id][0], evaluation[tank_id][1], evaluation[tank_id][2], evaluation[tank_id][3], evaluation[tank_id][4]), config_s
            
            to_be_added = True
            for score in best_scores:
               if is_score_ge(score, curr_score):
                  to_be_added = False
            if to_be_added:
               best_scores.append(curr_score)
        for i in range(len(best_scores)):
            file.write(str(best_scores[i])+"\n")
        

    file.close
    return best_scores

def plot_all_configurations_sensor(configurations, sensor):
    accuracies = [[] for tank in range(0, 3)]
    precisions = [[] for tank in range(0, 3)]
    recalls = [[] for tank in range(0, 3)]
    #l1s = [[] for tank in range(0, 3)]
    #l2s = [[] for tank in range(0, 3)]

    n_configurations_sensor = 0

    for configuration in configurations:
        if configuration[3] == sensor:
            n_configurations_sensor+=1
            evaluation_tanks = configuration[-1]
            for index_tank, evaluation_tank in enumerate(evaluation_tanks):
                accuracies[index_tank].append(evaluation_tank[0])
                precisions[index_tank].append(evaluation_tank[1])
                recalls[index_tank].append(evaluation_tank[2])
                #l1s[index_tank].append(evaluation_tank[3])
                #l2s[index_tank].append(evaluation_tank[3])


    #figure settings
    f, axes = plt.subplots(1, 3)
    f.set_figheight(5)
    f.set_figwidth(15)
    f.suptitle('All configuration evaluation sensor: '+sensors_names[sensor], fontsize = 15)

    x_axes = range (0,n_configurations_sensor)
    
    #plot evaluations
    for tank_id in range(3):
        axes[0].scatter(x_axes, accuracies[tank_id],  s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
        axes[1].scatter(x_axes, precisions[tank_id],  s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
        axes[2].scatter(x_axes, recalls[tank_id],  s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
        #axes[1][0].scatter(x_axes, l1s[tank_id],  s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])
        #axes[1][1].scatter(x_axes, l2s[tank_id],  s=10, c=tank_color[tank_id], marker=tank_marker[tank_id])


    axes[0].set_ylim(0,1)
    axes[0].set_xlabel('configuration id')
    axes[0].set_ylabel('accuracy')


    axes[1].set_ylim(0,1)
    axes[1].set_xlabel('configuration id')
    axes[1].set_ylabel('precision')
    #axes[0][1].set_ylabel(evaluation[measure_index], labelpad = 5)

    axes[2].set_ylim(0,1)
    axes[2].set_xlabel('configuration id')
    axes[2].set_ylabel('recall')

   # axes[1][0].set_xlabel('configuration id')
    #axes[1][0].set_ylabel('l1')

    #axes[1][1].set_xlabel('configuration id')
    #axes[1][1].set_ylabel('l2')

        

    #axes[-1][-1].set_visible(False)

    tank1 = mlines.Line2D([], [], color=tank_color[0], marker = tank_marker[0], markersize=10, linestyle='None', label='tank1')
    tank2 = mlines.Line2D([], [], color=tank_color[1], marker =tank_marker[1], markersize=10, linestyle='None', label='tank2')
    tank3 = mlines.Line2D([], [], color=tank_color[2], marker =tank_marker[2], markersize=10, linestyle='None', label='tank3')
    

    handles = [tank1, tank2, tank3]
    labels = [h.get_label() for h in handles] 

    f.legend(handles=handles, labels=labels)

    plt.savefig('./sensor_'+sensors_names[sensor]+'.png', dpi = 120)


def plot_by_alphabets(configurations, parameters_list, sensor):
    thresholds = parameters_list[0]
    windows = parameters_list[1]
    steps = parameters_list[2]
    alphabets = parameters_list[3]
    substring_size = parameters_list[4]
    paa_size = parameters_list[5]
    prjs_size = parameters_list[6]
    prjs_iter = parameters_list[7]
    freq_thresholds = parameters_list[8]
    anomaly_thresholds = parameters_list[9]

    line_fmt = ['r+:', 'rx:', 'g+:', 'gx:']
    legend_format = "w: {}, s: {}"
    legend = []

    f, axes = plt.subplots(3, 3)
    f.set_figheight(15)
    f.set_figwidth(22)
    f.suptitle('Sensor: '+sensors_names[sensor]+'-- accuracy, precision, recall for each tank', fontsize = 20)

    map_plot = {0 : [[[],[]],[[],[]],[[],[]]], 1: [[[],[]],[[],[]],[[],[]]], 2:[[[],[]],[[],[]],[[],[]]]}
    for threshold in thresholds:
        for window in windows:
            for step in steps:
                for alphabet in alphabets:
                    for sub_size in substring_size:
                        for pa in paa_size:
                            for prj_size in prjs_size:
                                for prj_iter in prjs_iter:
                                    for freq_threshold in freq_thresholds:
                                        for anomaly_threshold in anomaly_thresholds:
                                            for c in configurations:
                                                if c[0] == threshold and c[1] == window and c[2] == step and c[4] == alphabet \
                                                and c[5] == sub_size and c[6] == pa and c[7] == prj_size and c[8] == prj_iter\
                                                and c[9] == freq_threshold and c[10] == anomaly_threshold:
                                                    evaluation_tanks = c[-1]
                                                    for tank in range(3):
                                                        evaluation = evaluation_tanks[tank]
                                                        accuracy = evaluation[0]
                                                        precision = evaluation[1]
                                                        recall = evaluation[2]
                                                        if c[4] == 3:
                                                            (map_plot[tank])[0][0].append(accuracy)
                                                            (map_plot[tank])[1][0].append(precision)
                                                            (map_plot[tank])[2][0].append(recall)
                                                        elif c[4] == 5:
                                                            (map_plot[tank])[0][1].append(accuracy)
                                                            (map_plot[tank])[1][1].append(precision)
                                                            (map_plot[tank])[2][1].append(recall)
                legend.append(legend_format.format(window, step))

                 
    #alp1 = [[alphabets[0]]]*8   
    #alp2 = [[alphabets[1]]]*8

    for tank in range(3):

       for i in range(4):
          if tank == 0:

            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i]]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i]]
            rec = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i]]

            axes[0][0].plot(alphabets,acc, line_fmt[i])
            axes[0][1].plot(alphabets,prec, line_fmt[i])
            axes[0][2].plot(alphabets,rec, line_fmt[i])
        

          elif tank == 1:
            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i]]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i]]
            recall = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i]]


            axes[1][0].plot(alphabets,acc, line_fmt[i])
            axes[1][1].plot(alphabets,prec, line_fmt[i])
            axes[1][2].plot(alphabets,rec,line_fmt[i])

       	  
          elif tank == 2:
            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i]]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i]]
            recall = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i]]

            axes[2][0].plot(alphabets,acc, line_fmt[i])
            axes[2][1].plot(alphabets,prec,line_fmt[i])
            axes[2][2].plot(alphabets,rec,line_fmt[i])

            

            #axes[2][2].plot(alphabets, (map_plot[tank])[2][0][i]+(map_plot[tank])[2][1][i])

    for i in range(3):
        axes[i][0].set_xlabel('alphabet_size')
        axes[i][0].set_ylabel('accuracy')
        axes[i][1].set_xlabel('alphabet_size')
        axes[i][1].set_ylabel('precision')
        axes[i][2].set_xlabel('alphabet_size')
        axes[i][2].set_ylabel('recall')

        axes[i][0].set_xticks(alphabets)
        axes[i][1].set_xticks(alphabets)
        axes[i][2].set_xticks(alphabets)

    #axes[2][2].legend(legend)
    f.legend(legend)
    


    plt.savefig('./sensor_alphabet_'+sensors_names[sensor]+'.png', dpi = 120)   


def plot_by_window(configurations, parameters_list, sensor):

    thresholds = parameters_list[0]
    windows = windows_list
    steps = parameters_list[2]
    alphabets = parameters_list[3]
    substring_size = parameters_list[4]
    paa_size = parameters_list[5]
    prjs_size = parameters_list[6]
    prjs_iter = parameters_list[7]
    freq_thresholds = parameters_list[8]
    anomaly_thresholds = parameters_list[9]

    line_fmt = ['r+:', 'rx:', 'g+:', 'gx:']
    legend_format = "s: {}, a: {}"
    legend = []

    f, axes = plt.subplots(3, 3)
    f.set_figheight(15)
    f.set_figwidth(22)
    f.suptitle('Sensor: '+sensors_names[sensor]+'-- accuracy, precision, recall for each tank', fontsize = 20)

    map_plot = {0 : [[[],[]],[[],[]],[[],[]]], 1: [[[],[]],[[],[]],[[],[]]], 2:[[[],[]],[[],[]],[[],[]]]}
    #map_plot = {0 : [[[],[],[]],[[],[],[]],[[],[],[]]], 1: [[[],[],[]],[[],[],[]],[[],[],[]]], 2:[[[],[],[]],[[],[],[]],[[],[],[]]]}
    #init
    for tank in range(3):
        for measure in range(3):
            for window in range (2):
                for value in range (4):
                   (map_plot[tank])[measure][window].append(None)

    for step in steps:
    	for alphabet in alphabets:
    		legend.append(legend_format.format(step, alphabet))

    for threshold in thresholds:
        for window in windows:
            index = 0
            for step in steps:
                for alphabet in alphabets:
                    for sub_size in substring_size:
                        for pa in paa_size:
                            for prj_size in prjs_size:
                                for prj_iter in prjs_iter:
                                    for freq_threshold in freq_thresholds:
                                        for anomaly_threshold in anomaly_thresholds:
                                            for c in configurations:
                                                if c[0] == threshold and c[1] == window and c[2] == step and c[4] == alphabet \
                                                and c[5] == sub_size and c[6] == pa and c[7] == prj_size and c[8] == prj_iter\
                                                and c[9] == freq_threshold and c[10] == anomaly_threshold:
                                                    evaluation_tanks = c[-1]
                                                    for tank in range(3):
                                                        evaluation = evaluation_tanks[tank]
                                                        accuracy = evaluation[0]
                                                        precision = evaluation[1]
                                                        recall = evaluation[2]
                                                        if window == 50:
                                                            (map_plot[tank])[0][0][index] = accuracy
                                                            (map_plot[tank])[1][0][index] = precision
                                                            (map_plot[tank])[2][0][index] = recall
                                                        elif window == 75:
                                                            (map_plot[tank])[0][1][index] = accuracy
                                                            (map_plot[tank])[1][1][index] = precision
                                                            (map_plot[tank])[2][1][index] = recall
                                                    index+=1
                                                                            
                   

                 
    #alp1 = [[alphabets[0]]]*8   
    #alp2 = [[alphabets[1]]]*8

    for tank in range(3):

       for i in range(4):
          if tank == 0:

            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i]]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i]]
            rec = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i]]

            axes[0][0].plot(windows_list,acc, line_fmt[i])
            axes[0][1].plot(windows_list,prec, line_fmt[i])
            axes[0][2].plot(windows_list,rec, line_fmt[i])
        

          elif tank == 1:
            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i] ]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i] ]
            recall = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i] ]


            axes[1][0].plot(windows_list, acc, line_fmt[i])
            axes[1][1].plot(windows_list, prec, line_fmt[i])
            axes[1][2].plot(windows_list, rec,line_fmt[i])

       	  
          elif tank == 2:
            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i] ]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i] ]
            recall = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i]]

            axes[2][0].plot(windows_list,acc, line_fmt[i])
            axes[2][1].plot(windows_list,prec,line_fmt[i])
            axes[2][2].plot(windows_list,rec,line_fmt[i])

            

            #axes[2][2].plot(alphabets, (map_plot[tank])[2][0][i]+(map_plot[tank])[2][1][i])

    for i in range(3):
        axes[i][0].set_xlabel('window size')
        axes[i][0].set_ylabel('accuracy')
        axes[i][1].set_xlabel('window size')
        axes[i][1].set_ylabel('precision')
        axes[i][2].set_xlabel('window size')
        axes[i][2].set_ylabel('recall')

        axes[i][0].set_xticks(windows_list)
        axes[i][1].set_xticks(windows_list)
        axes[i][2].set_xticks(windows_list)

    #axes[2][2].legend(legend)
    f.legend(legend)
    


    plt.savefig('./sensor_windows_'+sensors_names[sensor]+'.png', dpi = 120)


def plot_by_step(configurations, parameters_list, sensor):

    thresholds = parameters_list[0]
    windows = parameters_list[1]
    steps = steps_list
    alphabets = parameters_list[3]
    substring_size = parameters_list[4]
    paa_size = parameters_list[5]
    prjs_size = parameters_list[6]
    prjs_iter = parameters_list[7]
    freq_thresholds = parameters_list[8]
    anomaly_thresholds = parameters_list[9]

    line_fmt = ['r+:', 'rx:', 'g+:', 'gx:']
    legend_format = "w: {}, a: {}"
    legend = []

    f, axes = plt.subplots(3, 3)
    f.set_figheight(15)
    f.set_figwidth(22)
    f.suptitle('Sensor: '+sensors_names[sensor]+'-- accuracy, precision, recall for each tank', fontsize = 20)

    map_plot = {0 : [[[],[]],[[],[]],[[],[]]], 1: [[[],[]],[[],[]],[[],[]]], 2:[[[],[]],[[],[]],[[],[]]]}
    #map_plot = {0 : [[[],[],[]],[[],[],[]],[[],[],[]]], 1: [[[],[],[]],[[],[],[]],[[],[],[]]], 2:[[[],[],[]],[[],[],[]],[[],[],[]]]}
    #init
    for tank in range(3):
        for measure in range(3):
            for step in range (2):
                for value in range (4):
                   (map_plot[tank])[measure][step].append(None)

    for window in windows:
    	for alphabet in alphabets:
    		legend.append(legend_format.format(window, alphabet))

    for threshold in thresholds:
        for window in windows:
            for step in steps:
                index = 0
                for alphabet in alphabets:
                    for sub_size in substring_size:
                        for pa in paa_size:
                            for prj_size in prjs_size:
                                for prj_iter in prjs_iter:
                                    for freq_threshold in freq_thresholds:
                                        for anomaly_threshold in anomaly_thresholds:
                                            for c in configurations:
                                                if c[0] == threshold and c[1] == window and c[2] == step and c[4] == alphabet \
                                                and c[5] == sub_size and c[6] == pa and c[7] == prj_size and c[8] == prj_iter\
                                                and c[9] == freq_threshold and c[10] == anomaly_threshold:
                                                    evaluation_tanks = c[-1]
                                                    for tank in range(3):
                                                        evaluation = evaluation_tanks[tank]
                                                        accuracy = evaluation[0]
                                                        precision = evaluation[1]
                                                        recall = evaluation[2]
                                                        if step == 40:
                                                            (map_plot[tank])[0][0][index] = accuracy
                                                            (map_plot[tank])[1][0][index] = precision
                                                            (map_plot[tank])[2][0][index] = recall
                                                        elif step == 70:
                                                            (map_plot[tank])[0][1][index] = accuracy
                                                            (map_plot[tank])[1][1][index] = precision
                                                            (map_plot[tank])[2][1][index] = recall
                                                    index+=1
                                                                            
                   

                 
    #alp1 = [[alphabets[0]]]*8   
    #alp2 = [[alphabets[1]]]*8

    for tank in range(3):

       for i in range(4):
          if tank == 0:

            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i]]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i]]
            rec = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i]]

            axes[0][0].plot(steps_list,acc, line_fmt[i])
            axes[0][1].plot(steps_list,prec, line_fmt[i])
            axes[0][2].plot(steps_list,rec, line_fmt[i])
        

          elif tank == 1:
            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i] ]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i] ]
            recall = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i] ]
            print(recall)

            axes[1][0].plot(steps_list, acc, line_fmt[i])
            axes[1][1].plot(steps_list, prec, line_fmt[i])
            axes[1][2].plot(steps_list, rec,line_fmt[i])

       	  
          elif tank == 2:
            acc = [(map_plot[tank])[0][0][i], (map_plot[tank])[0][1][i] ]
            prec = [(map_plot[tank])[1][0][i], (map_plot[tank])[1][1][i] ]
            recall = [(map_plot[tank])[2][0][i], (map_plot[tank])[2][1][i]]

            axes[2][0].plot(steps_list,acc, line_fmt[i])
            axes[2][1].plot(steps_list,prec,line_fmt[i])
            axes[2][2].plot(steps_list,rec,line_fmt[i])

            

            #axes[2][2].plot(alphabets, (map_plot[tank])[2][0][i]+(map_plot[tank])[2][1][i])

    for i in range(3):
        axes[i][0].set_xlabel('step size')
        axes[i][0].set_ylabel('accuracy')
        axes[i][1].set_xlabel('step size')
        axes[i][1].set_ylabel('precision')
        axes[i][2].set_xlabel('step size')
        axes[i][2].set_ylabel('recall')

        axes[i][0].set_xticks(steps_list)
        axes[i][1].set_xticks(steps_list)
        axes[i][2].set_xticks(steps_list)

    #axes[2][2].legend(legend)
    f.legend(legend)
    


    plt.savefig('./sensor_steps_'+sensors_names[sensor]+'.png', dpi = 120)  






def main(argv):
    FLAGS, _ = set_parser()

    #create directory for plots
    create_directory(FLAGS.plots_dir_fuzzy)
    create_directory(FLAGS.plots_dir_sax)

    n_tanks = 3



    

    
    configurations = load_configurations(FLAGS)
    
    #print for all sensors
    #for sensor in range(6):
       # plot_all_configurations_sensor(configurations, sensor)
    #get_plots(files_list, parameters_intitle, parameters_list, parameters_name_map, evaluation, 5)

    #get_plots(files_list, parameters_intitle, parameters_list, parameters_name_map, evaluation, 5)
    #plot_all_configurations_sensor(configurations, 5)
    #filtered_configurations_sensor = filter_configurations_by_sensor(configurations,5)
    #print(len(filtered_configurations_sensor))
    #find_best(filtered_configurations_sensor, 5)
    #exit()
    filtered_configurations_sensor = []
    parameters_list =  []

    for sensor in range(6):
        filtered_configurations_sensor = filter_configurations_by_sensor(configurations,sensor)
        if sensor == 0 or sensor == 1 or sensor == 3 or sensor == 4 or sensor == 5:
           thresholds = [0.2]
           windows_size = [50,75]
           steps_size = [40,70]
           #sensors = [0,1,2,3,4,5]
           alphabets = [3,5]
           substring_size = [5]
           paa_size = [30]
           prjs_size = [2]
           prjs_iter = [3]
           freq_thresholds = [0.7]
           anomaly_thresholds = [0.15]

           parameters_list = create_list_map(thresholds, windows_size, steps_size, alphabets, substring_size, paa_size, prjs_size, prjs_iter, freq_thresholds, anomaly_thresholds)

           #plot_by_alphabets(filtered_configurations_sensor, parameters_list, sensor)
        elif sensor == 2:
           thresholds = [0.1]
           windows_size = [50,75]
           steps_size = [40,70]
           #sensors = [0,1,2,3,4,5]
           alphabets = [3,5]
           substring_size = [7]
           paa_size = [30]
           prjs_size = [2]
           prjs_iter = [3]
           freq_thresholds = [0.7]
           anomaly_thresholds = [0.15]

           parameters_list = create_list_map(thresholds, windows_size, steps_size, alphabets, substring_size, paa_size, prjs_size, prjs_iter, freq_thresholds, anomaly_thresholds)

        #plot_by_alphabets(filtered_configurations_sensor, parameters_list, sensor)
        plot_by_window(filtered_configurations_sensor, parameters_list, sensor)
        #plot_by_step(filtered_configurations_sensor, parameters_list, sensor)




if __name__ == "__main__":
    main(sys.argv)
