#!/usr/bin/python3

import DataTypes
import DataLoader

loader = DataLoader.DataLoader("../../dataset/")
data = DataTypes.Data()
loader.load_data(data, 80)
