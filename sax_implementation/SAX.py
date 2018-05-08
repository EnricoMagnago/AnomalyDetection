import numpy as np
from saxpy.sax import sax_via_window
import string
import DataLoader
import DataTypes
import json
import sys
import collections
import itertools
import random
import datetime


#loading configuration from file
def load_configuration():
	with open('configuration.json', 'r') as f:
		configuration = json.load(f)

	parameters = {}

	#loading parameters for each category
	p_dataset = configuration['parameters']['dataset']
	p_sax = configuration['parameters']['sax'] 
	p_smoothing = configuration['parameters']['smoothing'] 
	p_projection = configuration['parameters']['random_projections']

	#loading single parameters for each category
	parameters['load_size'] = p_dataset['load_size']
	parameters['tank'] =  p_dataset['tank']
	parameters['sensor_type'] = p_dataset['sensor_type']
	parameters['power_type'] = p_dataset['power_type']

	parameters['alphabet_size'] =  p_sax['alphabet_size']
	parameters['paa_size'] = p_sax['paa_size']
	parameters['window_size'] = p_sax['window_size']

	parameters['threshold_freq'] = p_smoothing['threshold']

	parameters['prj_size'] = p_projection['prj_size']
	parameters['prj_iterations'] = p_projection['prj_iterations']
	parameters['anomaly_threshold'] = p_projection['anomaly_threshold']

	return parameters


	#print(configuration['parameters']['dataset'])
	#exit()

def get_string_representation(dict_data, load_size, window_size):

	string = ""

	num_subsq = load_size - window_size
	for i in range (num_subsq):
		for key, values in dict_data.items():
			if i in values:
				string = string + key
				break

	return string
	
	

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
		exit(0)
	return p
		

def get_hash_table(alphabet, prj_size):
	hash_table = {}
	cartesian_list = get_cartesian_list(alphabet, prj_size)
	
	#create keys for hash_table
	for element in cartesian_list:
		hash_table[element[0]+""+element[1]] = []
	return(hash_table)

def get_random_positions(prj_size, substring_size):

	#getting list of random positions
	random_positions = []
	
	i = 0
	while i < prj_size:
		position = random.randint(0, substring_size-1)
		if position in random_positions:
			continue #avoid to insert more than one time the same position for the current call
		else:
			random_positions.append(position)
			i = i + 1
			

	return random_positions




def put_in_bucket(hash_table_substring, string_smoothed,prj_iterations,prj_size, substring_size):
	#number of not overlapping substrings of size substring_size 
	not_overlapping_substrings = len(string_smoothed) // substring_size


	for iteation in range (prj_iterations):
		random_positions = get_random_positions(prj_size, substring_size)

		#max_bound (find substring till reach it)
		max_bound = len(string_smoothed) - substring_size

		for i in range (0, not_overlapping_substrings):

			start = i * 4
			end = substring_size + i*4

			if end <= max_bound:
				substring = string_smoothed[start:end]
				key_bucket = ""

				for position in random_positions:
					key_bucket = key_bucket + substring[position]
				(hash_table_substring[key_bucket]).append(substring)

	return hash_table_substring

def analyzed_bucket(hash_table_substring, total_elements, anomaly_threshold):
	bucket_with_anomalies = {}
	for key, values in hash_table_substring.items():
		print("Bucket %s freq %f "%(key, len(values)/total_elements))
		if len(values)/total_elements <= anomaly_threshold:
			print(key+" "+str(len(values)/total_elements))
			bucket_with_anomalies[key] = values
	return bucket_with_anomalies

def update_anomalies(anomalies, begin, end):
	for index, period in anomalies.items():
		if begin < period[1]:
			if begin > period[0]:
				if end > period[1]:
					(anomalies[index])[1] = end
					return anomalies
	index = len(anomalies.keys())
	anomalies[index] = [begin,end]
	return anomalies

def main(argv):

	#load configuration
	parameters = load_configuration()

	#load parameters

	#dataset
	load_size = parameters['load_size']
	

	#SAX
	alphabet_size = parameters['alphabet_size']
	paa_size = parameters['paa_size']
	window_size = parameters['window_size']

	#smoothing
	threshold_freq = parameters['threshold_freq']

	#projections
	prj_size = parameters['prj_size']
	prj_iterations = parameters ['prj_iterations']
	anomaly_threshold = parameters['anomaly_threshold']


	#for now put equal to paa_size
	substring_size = paa_size

	#loading data
	loader = DataLoader.DataLoader("../dataset/")
	data = DataTypes.Data()

	#loader.load_all(data,200)
	loader.load_subset(data,load_size,100)

	#period from which extract anomalies
	begin_date = datetime.datetime.fromtimestamp(data.index_to_time[0])
	end_date = datetime.datetime.fromtimestamp(data.index_to_time[load_size -1 ])

	if parameters['power_type'] == -1:
		tank = parameters['tank']
		sensor_type = parameters ['sensor_type']
		#print(data.measures[0])
		print("Loading of %i tank %i  data from %s to %s " % (sensor_type, tank, begin_date, end_date))
		s_values = [data.measures[i][0][tank][sensor_type] for i in range(0, len(data.measures))]
	else:
		power_type = parameters['power_type']
		print("Loading measures of power %i from %s to %s " % (power_type, begin_date, end_date))
		s_values = [data.measures[i][1][power_type] for i in range(0, len(data.measures))]

	




	dat = sax_via_window(s_values, window_size, paa_size, alphabet_size, nr_strategy = 'exactly', z_threshold = 0.01)

	string = get_string_representation(dat, load_size, window_size)
	
	#get smoothed string
	string_smoothed = smoothing(string, threshold_freq)

	#getting first n alphabet letters
	alphabet = get_alphabet_letters(alphabet_size)

	#creating hash table indexed by all of substrings of length k
	hash_table_substrings = get_hash_table(alphabet, prj_size)

	#fill hash table by applying random projection
	hash_table_substrings = put_in_bucket(hash_table_substrings, string_smoothed, prj_iterations, prj_size, substring_size)

	total = 0

	for key, values in hash_table_substrings.items():
		total = total + len(values)
		#print("Bucket "+key+ " Number of elements "+ str(len(values)))

	buckets_with_anomalies = analyzed_bucket(hash_table_substrings, total, anomaly_threshold)

	#print(buckets_with_anomalies)

	strings_anomalies = []
	for key, values in buckets_with_anomalies.items():
		strings_anomalies = strings_anomalies + list(set(values))
	strings_anomalies = list(set(strings_anomalies))

	#dict containing following <index_anomaly, period_anomaly(begin-end)>
	anomalies_period = {};

	for string_anomaly in strings_anomalies:
		subsequences = len(string_smoothed)//substring_size
		for index in range(len(string_smoothed)):
			start = paa_size*index
			end = start + substring_size
			if (end < len(string_smoothed)):
				string = string_smoothed[start:end]

				if string_anomaly == string:
					begin_date =datetime.datetime.fromtimestamp(data.index_to_time[index])
					end_date =datetime.datetime.fromtimestamp(data.index_to_time[index+window_size-1])
					anomalies_period = update_anomalies(anomalies_period, begin_date, end_date)
					
	for key, period in anomalies_period.items():
		print("Anomaly %i from %s to %s " % (key, period[0], period[1]))


if __name__ == "__main__":
	main(sys.argv)
