import numpy as np
from saxpy.sax import sax_via_window
import string
import DataLoader
import DataTypes
import collections
import itertools
import random

#parameters for SAX
alphabet_size = 4
paa_size = 4
#for now put equal to paa_size
substring_size = 4 
#projection size
prj_size = 2
prj_iterations = 1

win_size = 20
threshold_freq = 0.7
load_size = 5000

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


def get_alphabet_letters():
	alphabet=""
	for i in range(0,alphabet_size):
		alphabet = alphabet + chr(ord('a')+i)
	return alphabet;

def get_cartesian_list(alphabet):
	if prj_size <= len(alphabet):
		p = [x for x in itertools.product(alphabet, repeat=2)]
	else:
		print("Error: projection size exceeds alphabet size")
		exit(0)
	return p
		

def get_hash_table(alphabet):
	hash_table = {}
	cartesian_list = get_cartesian_list(alphabet)
	
	#create keys for hash_table
	for element in cartesian_list:
		hash_table[element[0]+""+element[1]] = []
	return(hash_table)

def get_random_positions():

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




def put_in_bucket(hash_table_substring, string_smoothed):
	#number of not overlapping substrings of size substring_size 
	not_overlapping_substrings = len(string_smoothed) // substring_size


	for iteation in range (prj_iterations):
		random_positions = get_random_positions()

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


#loading data
loader = DataLoader.DataLoader("../dataset/")
data = DataTypes.Data()

#loader.load_all(data,200)
loader.load_subset(data,load_size,100)

#get measures for oxygen
tank_j_oxigen = [data.measures[i][0][1][0] for i in range(0, len(data.measures))]

dat = sax_via_window(tank_j_oxigen, win_size, paa_size, alphabet_size, nr_strategy = 'exactly', z_threshold = 0.01)

string = get_string_representation(dat)

#get smoothed string
string_smoothed = smoothing(string)

#getting first n alphabet letters
alphabet = get_alphabet_letters()

#creating hash table indexed by all of substrings of length k
hash_table_substring = get_hash_table(alphabet)

#fill hash table by applying random projection
hash_table_substring = put_in_bucket(hash_table_substring, string_smoothed)

total = 0
for key, values in hash_table_substring.items():
	total = total + len(values)
	print("Bucket "+key+ " Number of elements "+ str(len(values)))
