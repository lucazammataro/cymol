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

<p align="center">
  <img src="images/LJP.gif" alt="Demo">
</p>

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


## Contribution

Contributions are welcome! Whether you're looking to fix bugs, enhance the functionality, or propose new features, please feel free to fork this repository, make changes, and submit pull requests.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the foundational principles set by Dennis C. Rapaport's book, which has been instrumental in guiding this project's development.

## References

1. Rapaport DC. The Art of Molecular Dynamics Simulation. 2nd ed. Cambridge University Press; 2004.

2. Smith KW. Cython: A Guide for Python Programmers 1st Edition, O'Really ISBN-13: 978-1491901557

3. Zammataro L. The Lennard-Jones potential Why the art of molecular dynamics is so fascinating, and why I got so emotionally overwhelmed, (https://towardsdatascience.com/the-lennard-jones-potential-35b2bae9446c)


Join us in advancing the field of molecular dynamics through innovative programming and collaborative development!

---

## Examples: 2D and 3D Simulation of the Lennard-Jones Potential

Compile and run two and three-dimensional simulations of the Lennard-Jones Potential (source codes available in the "modules" directory), a fundamental model used to describe interactions between particles in physics and chemistry. This simulations provide a dynamic and detailed visualization of how molecules interact through attractive and repulsive forces at varying distances.
Watch the video to explore how the Lennard-Jones Potential manifests in a 3D environment, with particles attracting and repelling each other, illustrating the dynamics that drive molecular behavior.

### Detailed Description of Functions in 2D and 3D Lennard-Jones Potential Simulations

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


[Watch the 2D Lennard-Jones Simulation on YouTube](https://youtu.be/mai3VEOZH0c)

[Watch the 3D Lennard-Jones Simulation on YouTube](https://youtu.be/Y6BNPL-ZChw?si=aWSa4FRqT2bbJ3EF)

