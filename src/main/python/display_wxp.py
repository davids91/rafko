#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import logger_pb2

#read in a log from the arguments

measurement = logger_pb2.Measurement()
if(1 < len(sys.argv)):
	with open(sys.argv[1],'rb') as file:
		measurement.ParseFromString(file.read())

#get bounds of the measurements
max_coordinate = 0
max_iteration = 0
for pack in measurement.packs:
	if(max_iteration < pack.iteration):
		max_iteration = pack.iteration
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
	return arr

def refresh_plots(iteration):
	displayable_weights = select_datapoint(iteration,"w")
	displayable_xps = select_datapoint(iteration,"xp")
	for i in range(num_of_plots):
		axs[i].clear()
		axs[i].fill_between(displayable_weights[i],displayable_xps[i],color="blue")
		axs[i].xaxis.tick_top()
	fig.canvas.draw() #maybe not needed

def update(val):
	refresh_plots(min(int(val), max_iteration))

fig, axs = plt.subplots(1,num_of_plots)
iter_slider = Slider(plt.axes([0.0, 0.0, 0.95, 0.05]), 'iteration', 1, max_iteration-1, valinit=1)
iter_slider.on_changed(update)
refresh_plots(1)
plt.show()
