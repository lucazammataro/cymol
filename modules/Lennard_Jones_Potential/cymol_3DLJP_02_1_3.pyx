# Module: cymol_3DLJP_02_1_3
# Title: Cymol Lennard Jones Potential 3D version
# Author: Luca Zammataro, Copyright (c) 2024
# Reference: https://towardsdatascience.com/the-lennard-jones-potential-35b2bae9446c

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
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import matplotlib.pyplot as plt

def init_pygame(int width, int height):
    pygame.init()
    screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
    if screen is None:
        raise ValueError("Failed to create a display surface.")
    return screen

def init_opengl(int width, int height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 500.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


'''
def display_particles_opengl(positions, nMol, sigma):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, -100, 0, 0, 0, 0, 1, 0)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    for i in range(nMol):
        glPushMatrix()
        glTranslatef(positions[i][0], positions[i][1], positions[i][2])
        glutSolidSphere(sigma, 20, 20)  # Draw a solid sphere for each particle
        glPopMatrix()


    pygame.display.flip()

'''

def display_particles_opengl(positions, nMol, sigma):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, -100, 0, 0, 0, 0, 1, 0)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    for i in range(nMol):
        glPushMatrix()
        glTranslatef(positions[i][0], positions[i][1], positions[i][2])
        
        # Define the material color
        if i == 0:
            color = [1.0, 0.0, 0.0, 1.0]  # Red
        elif i == 1:
            color = [0.0, 0.7, 1.0, 1.0]  # Green
        else:
            color = [1.0, 1.0, 1.0, 1.0]  # White

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color)
        glutSolidSphere(sigma, 20, 20)  # Draw a solid sphere for each particle
        glPopMatrix()

    pygame.display.flip()



# Global variable and constant declarations
cdef int IMUL = 314159269  # Multiplicative part of the random number generator
cdef int IADD = 453806245  # Additive part of the random number generator
cdef int MASK = 2147483647  # Mask to ensure result stays within integer limit
cdef double SCALE = 0.4656612873e-9  # Scale to convert integer to double
cdef int randSeedP = 17  # Seed for the random number generator

# Mol structure to represent a molecule with position, velocity, and acceleration
cdef struct Mol:
    double r[3]  # Position vector in 3D
    double rv[3]  # Velocity vector in 3D
    double ra[3]  # Acceleration vector in 3D

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
    cdef double t = 2.0 * pi * RandR()
    cdef double u = RandR()

    p[0] = cos(s) * sqrt(1 - u * u)
    p[1] = sin(s) * sqrt(1 - u * u)
    p[2] = cos(t) * u

# Function to apply periodic boundary conditions to vectors
cdef void VWrapAll(double* v, double* region):
    for i in range(3):  # Loop through x, y, z
        if v[i] >= 0.5 * region[i]:
            v[i] -= region[i]
        elif v[i] < -0.5 * region[i]:
            v[i] += region[i]

# Function to apply periodic boundary conditions to all molecules
cdef void ApplyBoundaryCond(int nMol, Mol* mol, double* region):
    for n in range(nMol):
        VWrapAll(mol[n].r, region)

# Function to initialize molecule coordinates
cdef void InitCoords(int[3] initUcell, int nMol, Mol* mol, double* region):
    cdef double gap[3]
    cdef int nx, ny, nz, n = 0
    gap[0] = region[0] / initUcell[0]
    gap[1] = region[1] / initUcell[1]
    gap[2] = region[2] / initUcell[2]
    for nz in range(initUcell[2]):
        for ny in range(initUcell[1]):
            for nx in range(initUcell[0]):
                mol[n].r[0] = (nx + 0.5) * gap[0] - 0.5 * region[0]
                mol[n].r[1] = (ny + 0.5) * gap[1] - 0.5 * region[1]
                mol[n].r[2] = (nz + 0.5) * gap[2] - 0.5 * region[2]
                n += 1

# Function to initialize molecule velocities
cdef void InitVels(int nMol, Mol* mol, double* vSum, double velMag):
    cdef int n, k

    for k in range(3):
        vSum[k] = 0.0

    for n in range(nMol):
        VRand(mol[n].rv)
        for k in range(3):
            mol[n].rv[k] *= velMag
            vSum[k] += mol[n].rv[k]

    for n in range(nMol):
        for k in range(3):
            mol[n].rv[k] -= vSum[k] / nMol

# Function to initialize molecule accelerations
cdef void InitAccels(int nMol, Mol* mol):
    for n in range(nMol):
        for k in range(3):
            mol[n].ra[k] = 0.0

# Function to set simulation parameters
cdef void SetParams(double density, int[3] initUcell, double sigma, double temperature, int nMol,
                    double* rCut, double* region, double* velMag):
    rCut[0] = pow(2.0, 1.0/6.0 * sigma) # The limiting separation
    region[0] = 1.0 / pow(density, 1.0/3.0) * initUcell[0]
    region[1] = 1.0 / pow(density, 1.0/3.0) * initUcell[1]
    region[2] = 1.0 / pow(density, 1.0/3.0) * initUcell[2]
    velMag[0] = sqrt(2.0 * (1.0 - 1.0 / nMol) * temperature)

# Function to configure the job for the simulation
cdef void SetupJob(int nMol, Mol* mol, int[3] initUcell, double* region, 
                   double* vSum, double velMag, int* stepCount):
    stepCount[0] = 0
    InitCoords(initUcell, nMol, mol, region)
    InitVels(nMol, mol, vSum, velMag)
    InitAccels(nMol, mol)

# Function to compute forces between molecules
cdef void ComputeForces(int nMol, Mol* mol, double sigma, double epsilon, double rCut, 
                        double* region, double* uSum, double* virSum):
    cdef double rrCut = Sqr(rCut)  # Square of the cutoff distance
    cdef int i, j, k
    cdef double dr[3]
    cdef double rr, rInv, sigmaOverR, sigmaOverR6, sigmaOverR12, fcVal, uSumValue

    uSum[0] = 0.0
    virSum[0] = 0.0
    for i in range(nMol):
        for k in range(3):
            mol[i].ra[k] = 0.0  # Initialize accelerations to zero
    for i in range(nMol):
        for j in range(i + 1, nMol):
            for k in range(3):
                dr[k] = mol[i].r[k] - mol[j].r[k]  # Calculate the distance in 3D
            VWrapAll(dr, region)  # Apply periodic boundary conditions
            rr = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]  # Calculate the square of the distance

            if rr < rrCut:
                # FULL LJP
                rInv = 1.0 / sqrt(rr)  # Calculate 1/r
                sigmaOverR = sigma * rInv  # Calculate (sigma/r)
                sigmaOverR6 = pow(sigmaOverR, 6)  # Calculate (sigma/r)^6
                sigmaOverR12 = sigmaOverR6 * sigmaOverR6  # Calculate (sigma/r)^12

                # Calculate the force using the full Lennard-Jones potential
                fcVal = 48.0 * epsilon * (sigmaOverR12 - 0.5 * sigmaOverR6) * rInv

                for k in range(3):
                    mol[i].ra[k] += fcVal * dr[k]  # Update acceleration for i
                    mol[j].ra[k] -= fcVal * dr[k]  # Update acceleration for j

                # Calculate the contribution to uSum (full LJP)
                uSumValue = 4 * epsilon * (sigmaOverR12 - sigmaOverR6)
                uSum[0] += uSumValue
                virSum[0] += fcVal * rr  # Update the virial

# Function to perform the Leapfrog integration step
cdef void LeapfrogStep(int nMol, Mol* mol, double deltaT, int part):
    cdef int n, k
    if part == 1:
        for n in range(nMol):
            for k in range(3):
                mol[n].rv[k] += 0.5 * deltaT * mol[n].ra[k]
                mol[n].r[k] += deltaT * mol[n].rv[k]
    else:
        for n in range(nMol):
            for k in range(3):
                mol[n].rv[k] += 0.5 * deltaT * mol[n].ra[k]

# Function to evaluate properties
cdef void EvalProps(int nMol, Mol* mol, double* vSum, Prop kinEnergy, 
                    Prop totEnergy, Prop pressure, double uSum, double virSum, double density, int stepCount):
    cdef double vvSum = 0.0
    cdef int n, k
    for k in range(3):
        vSum[k] = 0.0
    for n in range(nMol):
        for k in range(3):
            vSum[k] += mol[n].rv[k]
            vvSum += mol[n].rv[k]**2

    kinEnergy.val = 0.5 * vvSum / nMol
    totEnergy.val = kinEnergy.val + uSum / nMol
    pressure.val = density * (vvSum + virSum) / (nMol * 3)

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
            positions = [(mol[i].r[0], mol[i].r[1], mol[i].r[2]) for i in range(nMol)]
            display_particles_opengl(positions, nMol, sigma)

            # Collect summary data
            AccumProps(totEnergy, kinEnergy, pressure, 2, stepAvg)
            summary = print_summary(stepCount[0], timeNow, &vSum[0], nMol, totEnergy, kinEnergy, pressure)
            if summary is not None:
                systemParams.append(summary)
            AccumProps(totEnergy, kinEnergy, pressure, 0, stepAvg)

cpdef void run_simulation(double deltaT, double density, int initUcell_x, int initUcell_y, int initUcell_z, 
                          int stepAvg, double stepLimit, double temperature):

    cdef int nMol, stepCount
    cdef Mol* mol
    cdef cnp.ndarray region, vSum
    cdef double[:] region_view, vSum_view
    cdef Prop kinEnergy, totEnergy, pressure
    cdef double sigma = 1.0, epsilon = 1.0, rCut, velMag
    cdef int initUcell[3]
    cdef list systemParams = []

    # Initialize unit cell dimensions
    initUcell[0] = initUcell_x
    initUcell[1] = initUcell_y
    initUcell[2] = initUcell_z
    nMol = initUcell[0] * initUcell[1] * initUcell[2]

    # Allocate memory for molecules
    mol = <Mol*>malloc(nMol * sizeof(Mol))
    if not mol:
        raise MemoryError("Unable to allocate memory for molecules.")
    
    # Initialize arrays for regions and velocity sums
    region = np.zeros(3, dtype=np.float64)
    vSum = np.zeros(3, dtype=np.float64)
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

        # Initialize Pygame window and OpenGL context
        screen = init_pygame(1100, 900)
        init_opengl(1100, 900)

        # Run the simulation in Cython
        run_simulation_c(nMol, mol, region_view, sigma, epsilon, rCut, deltaT, vSum_view, kinEnergy,
                         totEnergy, pressure, density, int(stepAvg), &stepCount, systemParams, screen, stepLimit)

        # Get the positions for OpenGL visualization
        positions = np.array([(mol[i].r[0], mol[i].r[1], mol[i].r[2]) for i in range(nMol)])

        # Display the final state
        display_particles_opengl(positions, nMol, sigma)
                
    finally:
        # Free memory allocated for molecules
        free(mol)
        columns = ['timestep', 'timeNow', '$\Sigma v$', 'E', '$\sigma E$', 'Ek', '$\sigma Ek$', 'P_1', 'P_2']
        df_systemParams = pd.DataFrame([s for s in systemParams if s is not None], columns=columns)
        print(df_systemParams)
        GraphOutput(df_systemParams)
