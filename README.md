<p align="center">
  <img src="images/3DLJP.gif" alt="Demo">
</p>

# Cymol
A Research and Educational Project on Classical Molecular Dynamics

Developed by Luca Zammataro, Copyright (C) 2024


## Overview


Cymol is an educational and research-oriented project designed to explore classical molecular dynamics using modern programming paradigms. Inspired by Dennis C. Rapaport's seminal textbook, "The Art of Molecular Dynamics," this project implements some examples from the book. Named "Cymol" to reflect its use of Cython, this project aims to demonstrate how Cython can produce efficient, stable, and surprisingly fast code. Join us as we step-by-step uncover the impressive capabilities of Cython in molecular dynamics simulations.
Cymol is an ongoing project, still under development, with the goal of providing a powerful tool for molecular dynamics simulations.

## Project Goals

The primary objective of Cymol is not just to reimplement existing algorithms, but to enhance them using Cython, demonstrating the power of this language in achieving highly efficient, stable, and fast computational solutions. This project serves as a practical guide to understanding the intricacies of molecular dynamics simulations and the optimization capabilities of Cython.

## Why Cython?

Cython offers a unique blend of simplicity and power, providing the ease of Python with the capability to achieve performance close to C languages. Through Cymol, we aim to showcase:
- **Efficiency:** How Cython can handle computationally intensive tasks.
- **Stability:** Ensuring robust simulations that are reliable and reproducible.
- **Speed:** Achieving remarkable execution times that rival traditional C-based implementations.

## Getting Started

To get involved with Cymol or try out the simulations:
1. Clone this repository.
2. Ensure you have Python and Cython installed.
3. Follow the setup instructions in our documentation to start running your own simulations.

---

## Installation Guide for Cymol

### Prerequisites
Before starting, ensure you have either Anaconda or Python installed on your system. This project can be set up using Conda or Python's built-in venv.

### Creating a Virtual Environment

#### Using Conda (suggested)
1. Open your Terminal or Anaconda Prompt.
2. Create a new Conda environment:
  ```bash
  conda create -n cython_env
  ```

3. Activate the Environment:
  ```bash
  conda activate cython_env
  ```
4. Installing Required Libraries:
  ```bash
  conda install -c conda-forge numpy pandas pygame matplotlib pyopengl
  conda install cython
  ```

#### Using Python venv
1. Open your Terminal or Command Prompt.
2. Navigate to the project directory.
3. Create a virtual environment:
  ```bash
  python -m venv cython_env
  ```

4. Activate the environment:
  ```bash
  source cython_env/bin/activate
  ```

5. Installing Required Libraries
  ```bash
  pip install numpy pygame pandas matplotlib Cython
  ```

#### Verifying Installation
Ensure all components are installed correctly:
  ```python
  # test_installation.py
  try:
      import numpy as np
      from pygame.locals import *
      import pandas as pd
      from OpenGL.GL import *
      from OpenGL.GLU import *
      from OpenGL.GLUT import *
      import matplotlib.pyplot as plt
      print("All libraries are installed correctly.")
  except ImportError as e:
      print("An error occurred:", e)
  ```


#### Run this test using:
  ```bash
  python test_installation.py
  ```

Adjust parameters within the `main.py` as needed to customize the simulation conditions.

---

## Practical Examples with Cymol: 2D and 3D Simulation of the Lennard-Jones Potential

<p align="center">
  <img src="images/LJP.gif" alt="Demo">
</p>

### Introduction

This guide provides detailed steps on how to compile and run the 2D and 3D simulations of the Lennard-Jones Potential using Cymol. The project consists of a Cython file (`.pyx`), a setup script, and a Python script that includes the main function to execute the simulation.

### Requirements

Before you begin, ensure you have Python installed along with the following packages:
- Cython
- NumPy
- A C compiler (like GCC for Linux/Mac or MSVC for Windows)

### Compilation Steps

1. **Compile the Cython Code**:
   
Compile and run two and three-dimensional simulations of the Lennard-Jones Potential (source codes available in the `modules` directory), a fundamental model used to describe interactions between particles in physics and chemistry. This simulations provide a dynamic and detailed visualization of how molecules interact through attractive and repulsive forces at varying distances.
Watch the video to explore how the Lennard-Jones Potential manifests in a 3D environment, with particles attracting and repelling each other, illustrating the dynamics that drive molecular behavior.

[Watch the 2D Lennard-Jones Simulation on YouTube](https://youtu.be/mai3VEOZH0c)

[Watch the 3D Lennard-Jones Simulation on YouTube](https://youtu.be/Y6BNPL-ZChw?si=aWSa4FRqT2bbJ3EF)

The 3D Lennard-Jones Potential simulation, for example, consists of three files: a cymol_3DLJP_02_1_3.pyx file containing the Cython code (module), a cymol_3DLJP_02_1_3.py file with the module calls and the main function, and a setup file in Python named setup_cymol_3DLJP_02_1_3.py."
   
   - Download the necessary files (.pyx, .py, and setup.py), provided inthe `modules` directory.
   - Navigate to the directory containing your files.
   - Run the setup script to compile the `.pyx` file into a C extension. Use the following command:
     ```bash
     python setup_cymol_3DLJP_02_1_3.py build_ext --inplace
     ```
   - This command will generate a `.so` file (on Linux/Mac) or a `.pyd` file (on Windows) in the same directory, which is the compiled module that can be imported in Python.

3. **Verify Compilation**:
   - Ensure that the compilation has produced the necessary binary file without errors.

### Running the Simulation

1. **Execute the Simulation**:
   - Open the Python script containing the main function.
   - Modify the parameters in the run_simulation function call if necessary (e.g., adjusting the number of particles, time steps, etc.). The `.py` file appears as follows:
     
  
  ```python
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
  ```
  
  
  1. **`deltaT = 0.005`**:
     - **Description**: This parameter represents the time step used in the simulation. It controls the increment of time in each iteration of the simulation loop. A smaller `deltaT` provides more accurate results but requires more computational steps.
  
  2. **`density = 0.01`**:
     - **Description**: This parameter defines the particle density in the simulation space. It is the ratio of the number of particles to the volume of the simulation box. A lower density implies fewer particles in a given volume, while a higher density indicates more crowded conditions.
  
  3. **`initUcell_x = 10`, `initUcell_y = 10`, `initUcell_z = 10`**:
     - **Description**: These parameters specify the initial number of unit cells (lattice points) along the x, y, and z dimensions of the simulation box. This determines the initial arrangement of particles in a 3D grid structure.
  
  4. **`stepAvg = 1`**:
     - **Description**: This parameter indicates how frequently (in terms of simulation steps) the simulation averages or records certain quantities like energy or temperature. A value of `1` means that these values are averaged or recorded every single step.
  
  5. **`stepLimit = 10000`**:
     - **Description**: This parameter sets the maximum number of simulation steps to be executed. It defines the total duration of the simulation, with more steps allowing for a longer simulation time.
  
  6. **`temperature = 1.0`**:
     - **Description**: This parameter defines the initial temperature of the system, influencing the initial velocities of the particles. A higher temperature corresponds to higher kinetic energy and faster-moving particles.
  
  These parameters collectively control the dynamics, scale, and duration of the Lennard-Jones Potential simulation, allowing you to customize the conditions under which the particles interact.

   - Execute the script by running:
     ```bash
     python cymol_3DLJP_02_1_3.py
     ```
   - This script will import the compiled module and start the simulation using the parameters specified.

### Expected Results

- Upon execution, the script will simulate the interactions of particles under the Lennard-Jones potential in either 2D or 3D, depending on the configuration.
- Outputs such as particle trajectories, energy calculations, and potentially visualization (if implemented) will be displayed or saved according to the script's functionality.

### Troubleshooting

- If you encounter errors during compilation, ensure that your environment is set up correctly with all necessary dependencies installed.
- For runtime errors, check the parameter values and ensure that the `.pyx` module is correctly compiled and accessible.

---

### Detailed Description of Functions in 2D and 3D Lennard-Jones Potential Simulations: Underlying Theoretical Foundations

The simulations of the Lennard-Jones Potential, both in 2D and 3D, are built upon the principles of classical mechanics and use Newton's equations of motion to model the interactions between particles. Below is a detailed explanation of the primary functions used in these simulations:

1. **Initialization Functions (`init_particles`, `init_simulation`)**:
   - These functions are responsible for setting up the initial conditions of the simulation. 
   - `init_particles` initializes the position, velocity, and mass of each particle in the system. The particles are usually placed randomly within a predefined simulation box, with initial velocities sampled from a Maxwell-Boltzmann distribution to mimic thermal equilibrium.
   - `init_simulation` sets up global simulation parameters, including the number of particles, time step (`dt`), and the boundaries of the simulation box.

2. **Force Calculation Function (`compute_forces`)**:
   - This core function computes the forces acting on each particle due to interactions with all other particles in the system. 
   - It uses the Lennard-Jones potential, defined as:
     ![LJ Potential](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}V(r)%3D4\epsilon%20\left[\left(\frac{\sigma}{r}\right)^{12}%20-%20\left(\frac{\sigma}{r}\right)^6\right])
     where \(r\) is the distance between two particles, \(\epsilon\) represents the depth of the potential well, and \(\sigma\) is the finite distance at which the inter-particle potential is zero.
   - The force between two particles is derived from the potential as:
     ![Force Equation](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}F(r)%3D-\nabla%20V(r))
   - In both 2D and 3D simulations, the function loops over all pairs of particles, calculates the distance \(r\), and then computes the corresponding force components, updating each particle's force vector accordingly.

3. **Leapfrog Integration Step Algorithm (`leapfrog_step`)**:
   - The Leapfrog algorithm is an alternative to the Velocity-Verlet method and is widely used in molecular dynamics simulations due to its simplicity and stability.
   - This algorithm updates velocities and positions in a staggered manner, effectively "leapfrogging" over each other:
     ![Leapfrog Step 1](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}\mathbf{v}\left(t%20+%20\frac{\Delta%20t}{2}\right)%20=%20\mathbf{v}\left(t%20-%20\frac{\Delta%20t}{2}\right)%20+%20\mathbf{a}(t)%20\Delta%20t)
     ![Leapfrog Step 2](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}\mathbf{r}(t%20+%20\Delta%20t)%20=%20\mathbf{r}(t)%20+%20\mathbf{v}\left(t%20+%20\frac{\Delta%20t}{2}\right)%20\Delta%20t)
   - Here, the velocity is updated at half-integer time steps, while the position is updated at integer time steps. This method is particularly useful for ensuring that energy is conserved over long simulation periods.
   - The `leapfrog_step` function updates the particle positions based on the current velocities and then updates the velocities using the newly calculated forces.

4. **Integration Function (`integrate_motion`)**:
   - Depending on the implementation, this function could utilize the Leapfrog integration step or the Velocity-Verlet method to update the positions and velocities of the particles.
   - Both methods are rooted in Newton's second law, where the acceleration of each particle is calculated from the forces acting on it:
     ![Newton's Second Law](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}\mathbf{a}%20=%20\frac{\mathbf{F}}{m})
   - In the case of the Velocity-Verlet algorithm, the equations of motion are integrated as follows:
     ![Velocity-Verlet 1](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}\mathbf{r}(t%20+%20\Delta%20t)%20=%20\mathbf{r}(t)%20+%20\mathbf{v}(t)%20\Delta%20t%20+%20\frac{1}{2}%20\mathbf{a}(t)%20\Delta%20t^2)
     ![Velocity-Verlet 2](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}\mathbf{v}(t%20+%20\Delta%20t)%20=%20\mathbf{v}(t)%20+%20\frac{1}{2}%20[\mathbf{a}(t)%20+%20\mathbf{a}(t%20+%20\Delta%20t)]%20\Delta%20t)
   - The `integrate_motion` function updates the positions and velocities of all particles for each time step in the simulation.

5. **Boundary Condition Function (`apply_boundary_conditions`)**:
   - This function ensures that particles remain within the bounds of the simulation box.
   - Common boundary conditions include periodic boundary conditions (PBC), where a particle exiting one side of the box re-enters from the opposite side, effectively simulating an infinite system.
   - For 3D simulations, this function checks and corrects the positions and velocities in all three spatial dimensions, while in 2D, it only needs to consider two dimensions.

6. **Energy Calculation Function (`compute_energy`)**:
   - This function calculates the total energy of the system, which is the sum of kinetic and potential energies.
   - The kinetic energy is computed as:
     ![Kinetic Energy](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}E_k%20=%20\frac{1}{2}%20\sum_{i=1}^{N}%20m_i%20v_i^2)
   - The potential energy is the sum of all pairwise Lennard-Jones potentials:
     ![Potential Energy](https://latex.codecogs.com/png.latex?\dpi{110}\bg{transparent}\color{white}E_p%20=%20\sum_{i<j}%20V(r_{ij}))
   - This function helps in monitoring the conservation of energy during the simulation, an essential check for the accuracy of the simulation.

7. **Visualization and Rendering Functions (`render_particles`, `update_display`)**:
   - These functions handle the graphical representation of the particles in the simulation.
   - In the 2D simulation, particles are typically displayed as circles, while in 3D, they might be rendered as spheres.
   - The `update_display` function ensures that the particle positions and colors are updated in real-time, providing a visual representation of the simulation's progress.

8. **Simulation Loop (`run_simulation`)**:
   - This is the main loop that orchestrates the simulation, calling the other functions in sequence: initializing particles, computing forces, integrating motion, applying boundary conditions, and updating the display.
   - It iterates over the defined number of time steps, continuously updating the state of the system according to Newton's laws of motion.

### Key Points
- The simulations utilize Newtonian mechanics, with forces derived from the Lennard-Jones potential.
- Both 2D and 3D versions of the simulation use the same physical principles, differing only in the dimensionality of space.
- The Leapfrog integration step algorithm provides an efficient and stable method for updating particle positions and velocities, especially useful for long-term simulations.
- The Velocity-Verlet algorithm is another integration method used for stable and accurate updates of particle positions and velocities.
- Boundary conditions are applied to simulate an infinite system, and energy calculations ensure the conservation of energy throughout the simulation.

These functions, when combined, create a comprehensive simulation of particle interactions under the Lennard-Jones potential, allowing for detailed studies of molecular dynamics in both two and three dimensions.



## Contribution

Contributions are welcome! Whether you're looking to fix bugs, enhance the functionality, or propose new features, please feel free to fork this repository, make changes, and submit pull requests.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the foundational principles set by Dennis C. Rapaport's book, which has been instrumental in guiding this project's development.

Join us in advancing the field of molecular dynamics through innovative programming and collaborative development!


---

## References

1. Rapaport DC. The Art of Molecular Dynamics Simulation. 2nd ed. Cambridge University Press; 2004.

2. Smith KW. Cython: A Guide for Python Programmers 1st Edition, O'Really ISBN-13: 978-1491901557

3. Zammataro L. The Lennard-Jones potential Why the art of molecular dynamics is so fascinating, and why I got so emotionally overwhelmed, (https://towardsdatascience.com/the-lennard-jones-potential-35b2bae9446c)
