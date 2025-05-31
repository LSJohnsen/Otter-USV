# Otter USV Repo

### Introduction to robotics: Application Programming interface for the Otter USV

### Bachelor thesis: To run the software standalone or the simulator, run the main.py file.

###  ACIT 4830 - Special Robotics & Control Subject: To run DRL models, run Otter_dl. (Related changes to the simulator are saved under Otter_simulator_DRL)
  ##### ACIT4830 Project contributions are found under Otter-USV/Otter API/Otter_dl and the changes found to the stepwise simulation in Otter_simulator_DRL


This Github Repository is built on the Pyton Vehicle Simulator of Thor I. Fossen from Autumn 2023. 


If you wish to run your own control algorithm, import the Otter_api and create an object. This object has multiple functions that can be called, 
including "update_values" which will gather all the data from the Otter and update a dictionary tied to the otter object with all values recieved
from the Otter and a few that are calculated.
