'''
Module: cymol_LJP_02_1_2
Title: Cymol Lennard Jones Potential 2D version
Author: Luca Zammataro, Copyright (c) 2024
Reference: https://towardsdatascience.com/the-lennard-jones-potential-35b2bae9446c
'''

# cython: language_level=3

cimport numpy as cnp
import numpy as np
from libc.math cimport sqrt, cos, sin, pow, pi
from libc.stdlib cimport malloc, free
import time
import sys
import pygame
from pygame.locals import *
import pandas as pd
import matplotlib.pyplot as plt


def init_pygame(int width, int height):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    if screen is None:
        raise ValueError("Failed to create a display surface.")
    return screen


def display_particles_pygame(screen, positions, nMol, kinEnergy, totEnergy, pressure, timeNow, sigma):
    if screen is None:
        raise ValueError("Display surface not initialized.")

    background_color = pygame.Color(0, 0, 0)  # Nero
    screen.fill(background_color)
    #particle_color_default = pygame.Color(51, 153, 255)  # Bianco
    particle_color_default = pygame.Color(5, 186, 227)  # Bianco    
    particle_color_1 = pygame.Color(194, 35, 23)  # Rosso per la prima particella
    particle_color_2 = pygame.Color(13, 166, 39)  # Verde per la seconda particella    

    # Pulisci lo schermo
    screen.fill(background_color)

    # Disegna le particelle
    scale = 30
    width, height = screen.get_size()
    for i in range(nMol):
        x = (positions[i][0] + 0.5) * scale + width / 2
        y = (positions[i][1] + 0.5) * scale + height / 2
        
        # Scegli il colore della particella
        if i == 0:
            color = particle_color_1  # Colora la prima particella di rosso
        elif i == 1:
            color = particle_color_2  # Colora la seconda particella di verde
        else:
            color = particle_color_default  # Colora le altre particelle con il colore di default

        pygame.draw.circle(screen, color, (int(x), int(y)), int(sigma * scale / 2))
        

    # Visualizza le proprietà
    font = pygame.font.Font(None, 36)
    text = font.render(f'Time: {timeNow:.2f} s, KE: {kinEnergy:.2f}, TE: {totEnergy:.2f}, Pressure: {pressure:.2f}', True, (255, 255, 255))
    screen.blit(text, (10, 10))

    # Aggiorna il display
    pygame.display.flip()

    # Gestisci eventi
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()


# Global variable and constant declarations
cdef int IMUL = 314159269  # Multiplicative part of the random number generator
cdef int IADD = 453806245  # Additive part of the random number generator
cdef int MASK = 2147483647  # Mask to ensure result stays within integer limit
cdef double SCALE = 0.4656612873e-9  # Scale to convert integer to double
cdef int randSeedP = 17  # Seed for the random number generator

# Mol structure to represent a molecule with position, velocity, and acceleration
cdef struct Mol:
    double r[2]  # Position vector
    double rv[2]  # Velocity vector
    double ra[2]  # Acceleration vector

# Prop class to hold thermodynamic properties
cdef class Prop:
    cdef public double val, sum1, sum2  # Properties and their summations
    
    def __init__(self, double val=0.0, double sum1=0.0, double sum2=0.0):
        self.val = val
        self.sum1 = sum1
        self.sum2 = sum2

# Inline function to calculate the square of a number
cdef inline double Sqr(double x):
    return x * x

# Inline function to calculate the cube of a number
cdef inline double Cube(double x):
    return x * x * x

# Function to generate a random double value
cdef double RandR():
    global randSeedP
    randSeedP = (randSeedP * IMUL + IADD) & MASK
    return randSeedP * SCALE

# Function to assign random velocities to molecules
cdef void VRand(double* p):
    cdef double s = 2.0 * pi * RandR()
    p[0] = cos(s)
    p[1] = sin(s)

# Function to apply periodic boundary conditions to vectors
cdef void VWrapAll(double* v, double* region):
    if v[0] >= 0.5 * region[0]:
        v[0] -= region[0]
    elif v[0] < -0.5 * region[0]:
        v[0] += region[0]
        
    if v[1] >= 0.5 * region[1]:
        v[1] -= region[1]
    elif v[1] < -0.5 * region[1]:
        v[1] += region[1]

# Function to apply periodic boundary conditions to all molecules
cdef void ApplyBoundaryCond(int nMol, Mol* mol, double* region):
    for n in range(nMol):
        VWrapAll(mol[n].r, region)

# Function to initialize molecule coordinates
cdef void InitCoords(int[2] initUcell, int nMol, Mol* mol, double* region):
    cdef double gap[2]
    cdef int nx, ny, n = 0
    gap[0] = region[0] / initUcell[0]
    gap[1] = region[1] / initUcell[1]
    for ny in range(initUcell[1]):
        for nx in range(initUcell[0]):
            mol[n].r[0] = (nx + 0.5) * gap[0] - 0.5 * region[0]
            mol[n].r[1] = (ny + 0.5) * gap[1] - 0.5 * region[1]
            n += 1

# Function to initialize molecule velocities
cdef void InitVels(int nMol, Mol* mol, double* vSum, double velMag):
    cdef int n
    vSum[0] = 0.0
    vSum[1] = 0.0
    for n in range(nMol):
        VRand(mol[n].rv)
        mol[n].rv[0] *= velMag
        mol[n].rv[1] *= velMag
        vSum[0] += mol[n].rv[0]
        vSum[1] += mol[n].rv[1]

    for n in range(nMol):
        mol[n].rv[0] -= vSum[0] / nMol
        mol[n].rv[1] -= vSum[1] / nMol

# Function to initialize molecule accelerations
cdef void InitAccels(int nMol, Mol* mol):
    for n in range(nMol):
        mol[n].ra[0] = 0.0
        mol[n].ra[1] = 0.0

# Function to set simulation parameters
cdef void SetParams(double density, int[2] initUcell, double sigma, double temperature, int nMol,
                    double* rCut, double* region, double* velMag):
    rCut[0] = pow(2.0, 1.0/6.0 * sigma) # The limiting separation
    region[0] = 1.0 / sqrt(density) * initUcell[0]
    region[1] = 1.0 / sqrt(density) * initUcell[1]
    velMag[0] = sqrt(2.0 * (1.0 - 1.0 / nMol) * temperature)

# Function to configure the job for the simulation
cdef void SetupJob(int nMol, Mol* mol, int[2] initUcell, double* region, 
                   double* vSum, double velMag, int* stepCount):
    stepCount[0] = 0
    InitCoords(initUcell, nMol, mol, region)
    InitVels(nMol, mol, vSum, velMag)
    InitAccels(nMol, mol)

# Function to compute forces between molecules
cdef void ComputeForces(int nMol, Mol* mol, double sigma, double epsilon, double rCut, 
                        double* region, double* uSum, double* virSum):
    cdef double rrCut = Sqr(rCut)  # Square of the cutoff distance
    cdef int i, j
    cdef double dr[2]
    cdef double rr, rInv, sigmaOverR, sigmaOverR6, sigmaOverR12, fcVal, uSumValue

    uSum[0] = 0.0
    virSum[0] = 0.0
    for i in range(nMol):
        mol[i].ra[0] = 0.0  # Initialize accelerations to zero
        mol[i].ra[1] = 0.0
    for i in range(nMol):
        for j in range(i + 1, nMol):
            dr[0] = mol[i].r[0] - mol[j].r[0]  # Calculate the x distance
            dr[1] = mol[i].r[1] - mol[j].r[1]  # Calculate the y distance
            VWrapAll(dr, region)  # Apply periodic boundary conditions
            rr = dr[0] * dr[0] + dr[1] * dr[1]  # Calculate the square of the distance

            if rr < rrCut:
                # FULL LJP
                rInv = 1.0 / sqrt(rr)  # Calculate 1/r
                sigmaOverR = sigma * rInv  # Calculate (sigma/r)
                sigmaOverR6 = pow(sigmaOverR, 6)  # Calculate (sigma/r)^6
                sigmaOverR12 = sigmaOverR6 * sigmaOverR6  # Calculate (sigma/r)^12

                # Calculate the force using the full Lennard-Jones potential
                fcVal = 48.0 * epsilon * (sigmaOverR12 - 0.5 * sigmaOverR6) * rInv

                mol[i].ra[0] += fcVal * dr[0]  # Update acceleration for i
                mol[i].ra[1] += fcVal * dr[1]
                mol[j].ra[0] -= fcVal * dr[0]  # Update acceleration for j
                mol[j].ra[1] -= fcVal * dr[1]

                # Calculate the contribution to uSum (full LJP)
                uSumValue = 4 * epsilon * (sigmaOverR12 - sigmaOverR6)

                uSum[0] += uSumValue
                virSum[0] += fcVal * rr  # Update the virial

# Function to perform the Leapfrog integration step
cdef void LeapfrogStep(int nMol, Mol* mol, double deltaT, int part):
    cdef int n
    if part == 1:
        for n in range(nMol):
            mol[n].rv[0] += 0.5 * deltaT * mol[n].ra[0]
            mol[n].rv[1] += 0.5 * deltaT * mol[n].ra[1]

            mol[n].r[0] += deltaT * mol[n].rv[0]
            mol[n].r[1] += deltaT * mol[n].rv[1]

    else:
        for n in range(nMol):
            mol[n].rv[0] += 0.5 * deltaT * mol[n].ra[0]
            mol[n].rv[1] += 0.5 * deltaT * mol[n].ra[1]

# Function to evaluate properties
cdef void EvalProps(int nMol, Mol* mol, double* vSum, Prop kinEnergy, 
                    Prop totEnergy, Prop pressure, double uSum, double virSum, double density, int stepCount):
    cdef double vvSum = 0.0
    cdef int n
    vSum[0] = 0.0
    vSum[1] = 0.0
    for n in range(nMol):
        vSum[0] += mol[n].rv[0]
        vSum[1] += mol[n].rv[1]
        vvSum += mol[n].rv[0]**2 + mol[n].rv[1]**2

    kinEnergy.val = 0.5 * vvSum / nMol
    totEnergy.val = kinEnergy.val + uSum / nMol
    pressure.val = density * (vvSum + virSum) / (nMol * 2)

# Function to accumulate properties
cdef void AccumProps(Prop totEnergy, Prop kinEnergy, Prop pressure, int icode, int stepAvg):
    if icode == 0:
        PropZero(totEnergy)
        PropZero(kinEnergy)
        PropZero(pressure)
    elif icode == 1:
        PropAccum(totEnergy)
        PropAccum(kinEnergy)
        PropAccum(pressure)
    elif icode == 2:
        PropAvg(totEnergy, stepAvg)
        PropAvg(kinEnergy, stepAvg)
        PropAvg(pressure, stepAvg)

# Function to zero out properties
cdef Prop PropZero(Prop v):
    v.sum1 = v.sum2 = 0.0
    return v

# Function to accumulate properties
cdef Prop PropAccum(Prop v):
    v.sum1 += v.val
    v.sum2 += Sqr(v.val)
    return v

# Function to average properties
cdef Prop PropAvg(Prop v, double n):
    v.sum1 /= n
    v.sum2 = sqrt(max(v.sum2 / n - Sqr(v.sum1), 0.0))
    return v

# Function to print a summary of the simulation step
cdef tuple print_summary(int stepCount, double timeNow, double* vSum, int nMol,
                         Prop totEnergy, Prop kinEnergy, Prop pressure):
    if totEnergy.sum1 != 0.0 or totEnergy.sum2 != 0.0 or \
       kinEnergy.sum1 != 0.0 or kinEnergy.sum2 != 0.0 or \
       pressure.sum1 != 0.0 or pressure.sum2 != 0.0:
        print(f"{stepCount} {timeNow:.4f} {vSum[0] / nMol:.4f} {totEnergy.sum1:.4f} {totEnergy.sum2:.4f} "
              f"{kinEnergy.sum1:.4f} {kinEnergy.sum2:.4f} {pressure.sum1:.4f} {pressure.sum2:.4f}")
        return (stepCount, timeNow, vSum[0] / nMol, totEnergy.sum1, totEnergy.sum2,
                kinEnergy.sum1, kinEnergy.sum2, pressure.sum1, pressure.sum2)
    else:
        return None



cdef GraphOutput(df_systemParams):
    
    # Global variable for df_systemParams  # Adding the global variable
    
    ax = df_systemParams.plot(x="timestep", y='$\Sigma v$', kind="line", linestyle='-', color='blue')
    df_systemParams.plot(x="timestep", y='E', kind="line", ax=ax, linestyle='-', color="green")
    df_systemParams.plot(x="timestep", y='$\sigma E$', kind="line", ax=ax, linestyle='-', color="red")
    df_systemParams.plot(x="timestep", y='Ek', kind="line", ax=ax, linestyle='-', color="orange")
    df_systemParams.plot(x="timestep", y='$\sigma Ek$', kind="line", ax=ax, linestyle='-', color="purple")
    df_systemParams.plot(x="timestep", y='P_1', kind="line", ax=ax, linestyle='-', color="brown")
    df_systemParams.plot(x="timestep", y='P_2', kind="line", ax=ax, linestyle='-', color="cyan")

    # Save the graph to a file
    plt.savefig("graph_output.png", dpi=300)

    # Display the graph
    plt.show()



cdef void run_simulation_c(int nMol, Mol* mol, double[:] region, double sigma, double epsilon,
                           double rCut, double deltaT, double[:] vSum, Prop kinEnergy,
                           Prop totEnergy, Prop pressure, double density, int stepAvg, int* stepCount, 
                           list systemParams, object screen, double stepLimit):

    cdef int iStep
    cdef double timeNow, uSum, virSum

    for iStep in range(int(stepLimit)):
        uSum = 0.0
        virSum = 0.0
        stepCount[0] += 1
        timeNow = stepCount[0] * deltaT

        LeapfrogStep(nMol, mol, deltaT, 1)
        ApplyBoundaryCond(nMol, mol, &region[0])
        ComputeForces(nMol, mol, sigma, epsilon, rCut, &region[0], &uSum, &virSum)
        LeapfrogStep(nMol, mol, deltaT, 2)
        EvalProps(nMol, mol, &vSum[0], kinEnergy, totEnergy, pressure, uSum, virSum, density, stepCount[0])
        AccumProps(totEnergy, kinEnergy, pressure, 1, stepAvg)

        # Visualizzare il progresso ad ogni intervallo stepAvg
        if stepCount[0] % stepAvg == 0:
            positions = [(mol[i].r[0], mol[i].r[1]) for i in range(nMol)]
            display_particles_pygame(screen, positions, nMol, kinEnergy.val, totEnergy.val, pressure.val, timeNow, sigma)

            # Collect summary data
            AccumProps(totEnergy, kinEnergy, pressure, 2, stepAvg)
            summary = print_summary(stepCount[0], timeNow, &vSum[0], nMol, totEnergy, kinEnergy, pressure)
            if summary is not None:
                systemParams.append(summary)
            AccumProps(totEnergy, kinEnergy, pressure, 0, stepAvg)

cpdef void run_simulation(double deltaT, double density, int initUcell_x, int initUcell_y, 
                          int stepAvg, double stepLimit, double temperature):

    cdef int nMol, stepCount
    cdef Mol* mol
    cdef cnp.ndarray region, vSum
    cdef double[:] region_view, vSum_view
    cdef Prop kinEnergy, totEnergy, pressure
    cdef double sigma = 1.0, epsilon = 1.0, rCut, velMag
    cdef int initUcell[2]
    cdef list systemParams = []

    # Initialize unit cell dimensions
    initUcell[0] = initUcell_x
    initUcell[1] = initUcell_y
    nMol = initUcell[0] * initUcell[1]

    # Allocate memory for molecules
    mol = <Mol*>malloc(nMol * sizeof(Mol))
    if not mol:
        raise MemoryError("Unable to allocate memory for molecules.")
    
    # Initialize arrays for regions and velocity sums
    region = np.zeros(2, dtype=np.float64)
    vSum = np.zeros(2, dtype=np.float64)
    region_view = region
    vSum_view = vSum

    # Initialize property objects
    kinEnergy = Prop()
    totEnergy = Prop()
    pressure = Prop()

    try:
        # Set simulation parameters and initialize the system
        SetParams(density, initUcell, sigma, temperature, nMol, &rCut, &region_view[0], &velMag)
        SetupJob(nMol, mol, initUcell, &region_view[0], &vSum_view[0], velMag, &stepCount)

        # Initialize Pygame window
        screen = init_pygame(1100, 900)

        # Run the simulation in Cython
        run_simulation_c(nMol, mol, region_view, sigma, epsilon, rCut, deltaT, vSum_view, kinEnergy,
                         totEnergy, pressure, density, int(stepAvg), &stepCount, systemParams, screen, stepLimit)

        # Get the positions for Pygame visualization
        positions = np.array([(mol[i].r[0], mol[i].r[1]) for i in range(nMol)])

        # Display the final state
        display_particles_pygame(screen, positions, nMol, kinEnergy.val, totEnergy.val, pressure.val, stepCount * deltaT, sigma)
                
    finally:
        # Free memory allocated for molecules
        free(mol)
        columns = ['timestep', 'timeNow', '$\Sigma v$', 'E', '$\sigma E$', 'Ek', '$\sigma Ek$', 'P_1', 'P_2']
        df_systemParams = pd.DataFrame([s for s in systemParams if s is not None], columns=columns)
        print(df_systemParams)
        GraphOutput(df_systemParams)
