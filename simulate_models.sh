#!/bin/bash

### NEST simulations
module load nest-2.10.0-python-2.7.9    	# ===== specific =====
cd spiking_model/simulate
python microcircuit.py stim
python microcircuit.py spon
module unload nest-2.10.0-python-2.7.9 		# ===== specific =====

### analysis of NEST simulations
source activate default						# ===== specific =====
cd ../analysis
python calculateStuff.py
python plotStuff.py

### rate and linear model analysis
cd ../../rate_linear_model
python applyer.py
