#! /usr/bin/python3
import numpy as np
from saxpy.znorm import znorm
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
from saxpy.sax import ts_to_string
import string
import json
import sys
import collections
import itertools
import random
import datetime
import pickle
import os
import fnmatch
import time
sys.path.insert(0, "../../data_import/python_interface")
import DataLoader
import DataTypes


path_conf = './Configurations/'
path_result = './Results/'

#cleaning not useful files
def cleaning():
        for configuration in  os.listdir(path_conf):
                if fnmatch.fnmatch(configuration, '*.json~'):
                        os.remove(path_conf+""+configuration)

def create_directories():
        if not os.path.exists(path_conf):
                os.makedirs(path_conf)
        if not os.path.exists(path_result):
                os.makedirs(path_result)

        for i in range (6):
                path_sensor = path_result+"Sensor_"+str(i)
                path_power = path_result+"Power_"+str(i)
                if not os.path.exists(path_sensor):
                        os.makedirs(path_sensor)
                if not os.path.exists(path_power):
                        os.makedirs(path_power)

#loading configuration from file
def load_configuration(configuration):
        with open(configuration, 'r') as f:
                configuration = json.load(f)

        parameters = {}

        #loading parameters for each category
        p_dataset = configuration['parameters']['dataset']
        p_sax = configuration['parameters']['sax']
        p_smoothing = configuration['parameters']['smoothing']
        p_projection = configuration['parameters']['random_projections']

        #loading single parameters for each category
        parameters['path_to_dataset'] = p_dataset['path']
        parameters['load_size'] = p_dataset['load_size']
        parameters['tank'] =  p_dataset['tank']
        parameters['sensor_type'] = p_dataset['sensor_type']
        parameters['power_type'] = p_dataset['power_type']

        parameters['alphabet_size'] =  p_sax['alphabet_size']
        parameters['paa_size'] = p_sax['paa_size']
        parameters['window_size'] = p_sax['window_size']
        parameters['step'] = p_sax['step']
        parameters['substring_size'] = p_sax['substring_size']

        parameters['threshold_freq'] = p_smoothing['threshold']

        parameters['prj_size'] = p_projection['prj_size']
        parameters['prj_iterations'] = p_projection['prj_iterations']
        parameters['anomaly_threshold'] = p_projection['anomaly_threshold']

        return parameters



# def get_string_representation(dict_data, load_size, window_size):

#         string = ""

#         num_subsq = load_size - window_size
#         for i in range (num_subsq):
#                 for key, values in dict_data.items():
#                         if i in values:
#                                 string = string + key
#                                 break

#         return string



def smoothing(string, threshold_freq):

        #get charachters and counter (occurences)
        char_counter = collections.Counter(string)

        total_characters = len(string)

        #dictionary storing item {characters:frequency}
        dict_alphabet_frequency = {}

        #final smoothed string
        string_smoothed = ""

        #storing max charachter + its frequency
        alphabetM='';
        countM=0;

        for alphabet, count in char_counter.items():
                freq = count/total_characters
                dict_alphabet_frequency[alphabet] = freq
                if freq > countM:
                        alphabetM = alphabet
                        countM = freq

        #replacing step
        for alphabet in string:
                if ((dict_alphabet_frequency[alphabet] > threshold_freq) and (alphabet != alphabetM)):
                        string_smoothed = string_smoothed + alphabetM
                else:
                        string_smoothed = string_smoothed + alphabet

        return (string_smoothed)


def get_alphabet_letters(alphabet_size):
        alphabet=""
        for i in range(0,alphabet_size):
                alphabet = alphabet + chr(ord('a')+i)
        return alphabet;

def get_cartesian_list(alphabet, prj_size):
        if prj_size <= len(alphabet):
                p = [x for x in itertools.product(alphabet, repeat=2)]
        else:
                print("Error: projection size exceeds alphabet size")
                exit(1)
        return p

def get_random_positions(prj_size, substring_size, prj_iterations):
        retval = [[0 for i in range(0, prj_size)] for j in range(0, prj_iterations)]
        seq_gen = itertools.product(range(0,substring_size), repeat=prj_size)
        for i, item in enumerate(seq_gen):
            if i < prj_iterations:
                retval[i] = item
            else:
                j = int(random.random() * (i+1))
                if j < prj_iterations:
                    retval[j] = item
        return retval


def put_in_bucket(string_smoothed, begin_window, prj_iterations, prj_size, substring_size, k):
        #number of not overlapping substrings of size substring_size
        hash_table_substring = {}
        not_overlapping_substrings = len(string_smoothed) // substring_size

        random_positions_it = get_random_positions(prj_size, substring_size, prj_iterations)
        for random_positions in random_positions_it:
                begin_in_serie = begin_window

                for i in range (0, not_overlapping_substrings):

                        #smoothed string start
                        start = i * substring_size
                        #smoothed string end
                        end = substring_size + i*substring_size

                        substring = string_smoothed[start:end]

                        #calculate end index in original serie
                        end_in_serie = begin_in_serie + substring_size * k

                        key_bucket = ""

                        for position in random_positions:
                                key_bucket = key_bucket + substring[position]

                        #append to the hash table a tuple <substring, start_index, end_index>
                        if key_bucket in hash_table_substring:
                                (hash_table_substring[key_bucket]).append((substring, begin_in_serie, end_in_serie))
                        else:
                                hash_table_substring[key_bucket] = [(substring, begin_in_serie, end_in_serie)]

                        #new start in the original serie becomes the previous end
                        begin_in_serie = end_in_serie
        return hash_table_substring

def analyzed_bucket(hash_table_substring, total_elements, anomaly_threshold):
        bucket_with_anomalies = {}
        bucket_freq = {}
        for key, values in hash_table_substring.items():
                bucket_elements = len(values)
                bucket_freq[key] = bucket_elements/total_elements
                if bucket_elements/total_elements <= anomaly_threshold and not bucket_elements == 0 :
                        bucket_with_anomalies[key] = values
        return bucket_with_anomalies, bucket_freq

def get_unique_tuple(hash_table_substrings):
        unique_tuple = set()
        for key, values in hash_table_substrings.items():
                if values:
                        for tup in values:
                                unique_tuple.add(tup)

        return unique_tuple

def getting_score (hash_table_substrings, buckets_with_anomalies, n_buckets_anomalies):
        unique_tuple = get_unique_tuple(hash_table_substrings)
        #print(unique_tuple)
        avg_score = 0;

        if n_buckets_anomalies != 0:
                #for each tuple count how many items it appears in an bucket with anomalies
                for tup in unique_tuple:
                        is_in_n_buckets = 0
                        for key, values in buckets_with_anomalies.items():
                                for item in values:
                                        if tup[0] == item[0] and tup[1] == item[1] and tup[2] == item[2]:
                                                is_in_n_buckets  = is_in_n_buckets  + 1
                                                break
                        #calculate score
                        avg_score = avg_score + (is_in_n_buckets/n_buckets_anomalies)/len(unique_tuple)

                avg_score = round(avg_score,2)

        return avg_score

def config_gen_files(path_conf):
        for configuration in  os.listdir(path_conf):
                parameters = load_configuration(path_conf+configuration)
                yield parameters

def config_gen_combinatorial():
        name = "./window{}/step{}/sensor{}_alphabet{}_substring{}_paa{}_prjsize{}_prjiter{}_freqthr{}_anomalythr{}.dump"
        parameters = {}
        parameters ['power_type'] = -1
        sensors_list = range(0, 5)
        alphabet_size_list = [3, 5]
        substring_size_list = [5, 7]
        paa_size_list = [10, 30]

        window_size_list = [25, 50, 75]
        step_size_list = [40, 70]

        threshold_freq_list = [0.7]
        prj_size_list = [2, 3]
        prj_iterations_list = [3]
        anomaly_threshold_list = [0.15]

        for sensor_id in sensors_list:
                parameters ['sensor_type'] = sensor_id
                for alphabet_size in alphabet_size_list:
                        parameters['alphabet_size'] = alphabet_size
                        for window_size in window_size_list:
                                parameters['window_size'] = window_size
                                for paa_size in [x for x in paa_size_list if x < window_size]:
                                        parameters['paa_size'] = paa_size
                                        for step_size in step_size_list:
                                                parameters['step'] = step_size
                                                for substring_size in substring_size_list:
                                                        parameters['substring_size'] = substring_size
                                                        for threshold_freq in threshold_freq_list:
                                                                parameters['threshold_freq'] = threshold_freq
                                                                for prj_size in [x for x in prj_size_list if x < alphabet_size and \
                                                                                 substring_size / x > 2]:
                                                                        parameters['prj_size'] = prj_size
                                                                        for prj_it in prj_iterations_list:
                                                                                parameters ['prj_iterations'] = prj_it
                                                                                for anomaly_th in anomaly_threshold_list:
                                                                                        parameters['anomaly_threshold'] = anomaly_th
                                                                                        parameters['dump_file'] = \
                        name.format(window_size, step_size, sensor_id, alphabet_size, substring_size, paa_size, prj_size, prj_it, threshold_freq, anomaly_th)
                                                                                        yield parameters



def run_with_config(parameters, data, id):
        anomaly_file = parameters["dump_file"]
        # already computed.
        if os.path.isfile(anomaly_file):
                print("file: {} already exists".format(anomaly_file))
                return

        sensor_type = parameters ['sensor_type']
        power_type = parameters ['power_type']


        #SAX
        alphabet_size = parameters['alphabet_size']
        paa_size = parameters['paa_size']
        window_size = parameters['window_size']
        step = parameters['step']
        substring_size = parameters['substring_size']

        #smoothing
        threshold_freq = parameters['threshold_freq']

        #projections
        prj_size = parameters['prj_size']
        prj_iterations = parameters ['prj_iterations']
        anomaly_threshold = parameters['anomaly_threshold']

        #list containg score for each window
        anomalies_score = [None]
        #n_iterations values = {3,6} tanks or powers
        n_iterations = 0
        #file to save
        #getting first n alphabet letters
        alphabet = get_alphabet_letters(alphabet_size)

        if power_type == -1 and sensor_type != -1:
                n_iterations = 3
                anomalies_score = [[],[],[]]
        else:
                print("Error during establishing which values (tanks or powers) to load")
                exit(1)


        for i in range(n_iterations):
                print("\t{} -> tank: {}".format(id, i))
                score = []

                tank = i
                #print(data.measures[0])
                #print("Loading of %i tank %i  data from %s to %s " % (sensor_type, tank, begin_date, end_date))
                s_values = [data.measures[j][0][tank][sensor_type] for j in range(0, len(data.measures))]


                len_serie = len(s_values)

                print("\t\titer: {}/{}".format(0, len_serie / step))
                last_time = time.time()
                for index in range(0,len_serie,step):
                        if time.time() - last_time > 20:
                                print("\t\titer: {}/{}".format(index / step, len_serie / step))
                                last_time = time.time()
                        begin = index
                        end = begin + window_size

                        if end < len_serie:
                                window_values = s_values[begin:end]
                                window_znorm = znorm(window_values)
                                window_paa = paa(window_znorm, paa_size)
                                window_string = ts_to_string(window_paa, cuts_for_asize(alphabet_size))

                                #each character of the string corresponds to k values of the series
                                k = window_size//paa_size


                                #get smoothed string
                                window_smoothed = smoothing(window_string, threshold_freq)

                                #fill hash table by applying random projection
                                hash_table_substrings = \
                                        put_in_bucket(window_smoothed, begin, prj_iterations, \
                                                      prj_size, substring_size, k)

                                total = 0
                                for _, values in hash_table_substrings.items():
                                        total = total + len(values)

                                buckets_with_anomalies, bucket_freq = \
                                        analyzed_bucket(hash_table_substrings, \
                                                        total, anomaly_threshold)

                                #number of bucket with anomalies
                                n_buckets_anomalies = len(buckets_with_anomalies.keys())

                                #getting score for current window
                                avg_window_score = \
                                        getting_score(hash_table_substrings, \
                                                      buckets_with_anomalies, \
                                                      n_buckets_anomalies)

                                score.append(avg_window_score)

                        else:
                                break
                anomalies_score[i] = score
                with open(anomaly_file, 'wb') as f:
                        pickle.dump(anomalies_score, f, pickle.HIGHEST_PROTOCOL)



def main(argv):

        #create directories
        #create_directories()
        #cleaning()

        #for each configuration file
        if len(os.listdir(path_conf)) == 0 :
                print("No configuration files found: aborted")
                exit(1)



        #loading data
        loader = DataLoader.DataLoader("../../dataset")
        data = DataTypes.Data()

        loader.load_all(data, 400)
        #period from which extract anomalies
        #begin_date = datetime.datetime.fromtimestamp(data.index_to_time[0])
        #end_date = datetime.datetime.fromtimestamp(data.index_to_time[len(data.measures)-1])

        futures = []
        for id, parameters in enumerate(config_gen_combinatorial()):
                if "dump_file" in parameters:
                        anomaly_file = parameters["dump_file"]
                        # create directories
                        os.makedirs(os.path.dirname(anomaly_file), exist_ok=True)
                else:
                        anomaly_file = ""
                        create_directories()
                print("executing {}....".format(id))
                run_with_config(parameters, data, id)
                print("done")



if __name__ == "__main__":
        main(sys.argv)
