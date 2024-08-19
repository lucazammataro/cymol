import cymol_LJP_02_1_1
import os

def main():
  
    # Define simulation parameters
    deltaT = 0.005
    density = 0.8
    initUcell_x = 20
    initUcell_y = 20
    stepAvg = 100
    stepEquil = 0
    stepLimit = 10000
    temperature = 1.0
    movie = 'n'


    # Define the working directory using os.getcwd()
    workdir = os.getcwd()

    # Run the simulation
    cymol_LJP_02_1_1.run_simulation(deltaT, density, initUcell_x, initUcell_y, stepAvg, stepEquil, stepLimit, temperature, workdir, movie)

if __name__ == "__main__":
    main()
