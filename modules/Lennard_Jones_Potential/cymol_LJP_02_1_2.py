'''
Title: Cymol Lennard Jones Potential 2D version
Author: Luca Zammataro, Copyright (c) 2024
Code: main function
Reference: https://towardsdatascience.com/the-lennard-jones-potential-35b2bae9446c
'''

import cymol_LJP_02_1_2
import os

def main():
  
    # Define simulation parameters
    deltaT = 0.005
    density = 0.5
    initUcell_x = 20
    initUcell_y = 20
    stepAvg = 1
    stepLimit = 10000
    temperature = 3.0

    # Run the simulation
    cymol_LJP_02_1_2.run_simulation(deltaT, density, initUcell_x, initUcell_y, stepAvg, stepLimit, temperature)

if __name__ == "__main__":
    main()
