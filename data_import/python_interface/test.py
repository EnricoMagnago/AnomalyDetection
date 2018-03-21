#!/usr/bin/python3

import DataTypes
import DataLoader

loader = DataLoader.DataLoader("../../dataset/")
data = DataTypes.Data()
# loader.load_all(data) # load all data, use default log ratio
loader.load_subset(data) # load subset of data, use default size and log ratio
