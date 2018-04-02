import numpy as np
from saxpy.sax import sax_via_window
import DataLoader
import DataTypes
import collections

#parameters for SAX
alphabet_size = 5
paa_size = 4
win_size = 20
threshold_freq = 0.06
load_size = 2000

def get_string_representation(dict_data):

	string = ""

	num_subsq = load_size - win_size
	for i in range (num_subsq):
		for key, values in dict_data.items():
			if i in values:
				string = string + key
				break

	return string
	
	

def smoothing(string):

	#get charachters and counter (occurences)
	char_counter = collections.Counter(string)
	
	total_characthers = len(string)
	
	#dictionary storing item {characters:frequency}
	dict_alphabet_frequency = {}

	#string containing final smoothed string
	string_smoothed = ""

	alphabetM='';
	countM=0;

	for alphabet, count in char_counter.items():
		freq = count/total_characthers
		dict_alphabet_frequency[alphabet] = freq
		if freq > countM:
			alphabetM = alphabet
			countM = freq

	#this is implement the replacing step in the paper
	for alphabet in string:
		if ((dict_alphabet_frequency[alphabet] > threshold_freq) and (alphabet != alphabetM)):
			string_smoothed = string_smoothed + alphabetM
		else:
			string_smoothed = string_smoothed + alphabet

	return (string_smoothed)



#loading data
loader = DataLoader.DataLoader("../dataset/")
data = DataTypes.Data()
#loader.load_all(data,200)
loader.load_subset(data,load_size,100)

#get measures for oxygen
tank_j_oxigen = [data.measures[i][0][1][0] for i in range(0, len(data.measures))]

dat = sax_via_window(tank_j_oxigen, win_size, paa_size, alphabet_size, nr_strategy = 'exactly', z_threshold = 0.01)

string = get_string_representation(dat)

string_smoothed = smoothing(string)
