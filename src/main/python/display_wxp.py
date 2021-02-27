#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import logger_pb2

#read in a log from the arguments

measurement = logger_pb2.Measurement()
with open('../logs/RABrain_training.log','rb') as file:
	measurement.ParseFromString(file.read())

#get bounds of the measurements
max_coordinate = 0
for pack in measurement.packs:
	for coordinate in pack.coordinates:
		if(max_coordinate < coordinate):
			max_coordinate = coordinate
num_of_plots = max_coordinate + 1

def select_datapoint(iteration,tag):
	arr = [0 for x in range(num_of_plots)]
	for pack in measurement.packs:
		if(iteration == pack.iteration):
			if(0 < len(pack.tags) and 0 < len(pack.coordinates) and 0 < len(pack.tags) and pack.tags[0] == tag):
				arr[pack.coordinates[0]] = pack.data
				print(tag,pack.data)
	return arr

displayable_weights = select_datapoint(1,"w")
displayable_xps = select_datapoint(1,"xp")

def refresh_plots():
	for i in range(num_of_plots):
		axs[i].fill_between(displayable_weights[i],displayable_xps[i])

fig, axs = plt.subplots(1,num_of_plots)
refresh_plots()
plt.show()
