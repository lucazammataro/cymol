import cymol_3DLJP_02_1_3
import os

def main():
  
    # Define simulation parameters
    deltaT = 0.005
    density = 0.01
    initUcell_x = 10
    initUcell_y = 10
    initUcell_z = 10 
    stepAvg = 1
    stepLimit = 10000
    temperature = 1.0

    # Run the simulation
    cymol_3DLJP_02_1_3.run_simulation(deltaT, density, initUcell_x, initUcell_y, initUcell_z, stepAvg, stepLimit, temperature)

if __name__ == "__main__":
    main()
