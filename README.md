########################
###### layeredOSI ###### 
########################


###### simulation code ######

The simulation and analysis are composed of three (potentially four) basic steps, which correspond to the three clocks of commands in the 'simulate_models.sh' file:

1. NEST simulations:
First, the two conditions - spontaneous and visually stimulated - need to be simulated in the spiking model by running the 'spiking_model/simulate/microcircuit.py' twice with the respective arguments. This produces the 'data.hdf5' files in the respective sub-folders, for the default simulation that is under 'data/noIStim'. These include the spike times as well as the full connectivity of the network. 
It is likely that you need to adapt the number of virtual processes (n_vp, line 76 of file 'spiking_model/simulate/sim_params.py') to the number of cores of the machine on which the simulations are run. 

2. Analysis of spiking model:
The analysis of the data is divide into two steps. The first step is to calculate firing rates, orientation selectivities and further metrics by running 'spiking_model/analysis/calculateStuff.py'. The results are stored in 'analysis.hdf5' files.
The second step is to create the plots based on the calculations, which is performed by the 'spiking_model/analysis/plotStuff.py' script. The figures are also saved to the respective 'data' sub-folders.
To run the analysis for other than the default simulation name, you need to adapt the  'data_folder' variable in the two analysis files.

3. Rate and linear models:
The simulation and analysis of these two models are run with the 'rate_linear_model/applyer.py' script. It is based on the same parameters and connectivity used by a previous NEST simulation defined by the 'experiment' and 'c_experiment' variables. It also saves the simulation results to the 'analysis.hdf5' created in the previous step. The maximum number of parallel simulations can be set in lines 23/24.

4. External current stimulation:
- Spiking simulation: In order to to simulate external current stimulations, the file 'spiking_model/simulate/sim_params.py' needs to be changed. In particular, lines 30/31 need to be exchanged by either lines 34-42 OR 45-53, where the current to the different populations are set. The relative strength and number of simulations are set in the 'np.linspace' command. The simulations can then be run by 'python microcircuit.py stim i', where 'i' is the index of the respective relative strength.
- Rate and linear model: For these models the current stimulations are performed by 'rate_linear_model/applyer_dIscan.py i', where, again,  'i' is the index of the respective relative strength. Similar than before, these simulations are based on a NEST simulation without external current. The maximal relative strength and step width be can set via the 'max_frac' and 'n_steps' variables and the  maximum number of parallel simulations by 'max_proc'.

Note that it is not necessary to run all parts of the code on the same machine. For instance, it may be convenient to run the NEST simulations on a cluster computer and, since plotting there might be difficult/impossible, run the analysis on a smaller, local machine. In order to do so, simply move the entire simulation folder including the data sub-folder to the new machine.

###### resource requirements ######

The code was successfully tested on a 12-core machine with ~50GB of memory. Note that, depending on your recourses, running all simulations and analyses can take up to several days given the provided implementation. During the original production of results, heavy use was made from additional parallelization, for instance, by running simulations for different angles and other parameters intendedly and in parallel. However, as such implementations are strongly dependent on the environment in which they are run, the version provided here was optimized for ease of adaptivity and low resource requirements.

###### software versions ######

The code was developed and run with the following versions of software packages:    
    NEST:       2.10.0
    python:     2.7.9/2.7.15
    h5py:       2.5.0/2.8.0
    matplotlib: 2.2.2
    numpy:      1.14.5
    scipy:      1.1.0

###### contact info ######

In case you are running into problems regarding the simulation and analysis code provided here, feel free to contact    benjaminmerkt@gmx.de   for questions and/or advice.
