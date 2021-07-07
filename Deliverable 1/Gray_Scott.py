# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:43:26 2021

@author: Cillian
"""

import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl


def discrete_laplacian(M):
    """Get the discrete Laplacian of matrix M"""
    L = -4*M
    L += np.roll(M, (0,-1), (0,1)) # right neighbor
    L += np.roll(M, (0,+1), (0,1)) # left neighbor
    L += np.roll(M, (-1,0), (0,1)) # top neighbor
    L += np.roll(M, (+1,0), (0,1)) # bottom neighbor
    
    return L


def gray_scott_update(A, B, DA, DB, f, k, delta_t):
    """
    Updates a concentration configuration according to a Gray-Scott model
    with diffusion coefficients DA and DB, as well as feed rate f and
    kill rate k.
    """
    
    # Let's get the discrete Laplacians first
    LA = discrete_laplacian(A)
    LB = discrete_laplacian(B)
    
    # Now apply the update formula
    diff_A = (DA*LA - A*B**2 + f*(1-A)) * delta_t
    diff_B = (DB*LB + A*B**2 - (k+f)*B) * delta_t
    
    A += diff_A
    B += diff_B
    
    return A, B


def get_initial_configuration(N, random_influence=0.2):
    """
    Initialize a concentration configuration. N is the side length
    of the (N x N)-sized grid.
    `random_influence` describes how much noise is added.
    """
    
    # We start with a configuration where on every grid cell 
    # there's a lot of chemical A, so the concentration is high
    A = (1-random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))
    
    # Let's assume there's only a bit of B everywhere
    B = random_influence * np.random.random((N,N))
    
    # Now let's add a disturbance in the center
    N2 = N//2
    radius = r = int(N/10.0)
    
    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25
    
    return A, B


# Only have one output
def Y(var=[], N = 200, Time=  False):
    """
    Gray_Scott Model, outputs concentration at one grid point
    
    :param N: (int), Size of the NxN grid
    :param Time: (bool), Indicates if concentration at every time-step (True) or just the final concentration (False) should be returned

    :return: (float), concentration at one grid point
    """
    A, B = get_initial_configuration(N)
    
    # simulation steps
    N_simulation_steps = 100    
    
    DA, DB, f, k = var
    delta_t = 1
    
    Eval_pts = [(50, 50)]
    
    if Time:
        print("Collecting all time data points...")
        concentration = []
        
        for t in range(N_simulation_steps):
            A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
            concentration.append(A[Eval_pts[0]])
            
        return concentration
        
    else:
        for t in range(N_simulation_steps):
            A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
    
        return A[Eval_pts[0]]


def Y_T(var=[], N = 200):
    """
    Gray_Scott Model, outputs concentration at one grid point for all time points
    
    :param N: (int), Size of the NxN grid

    :return: (float), concentration at one grid point
    """
    A, B = get_initial_configuration(N)
    
    # simulation steps
    N_simulation_steps = 100    
    
    DA, DB, f, k = var
    delta_t = 1
    
    Eval_pts = [(50, 50)]
    
    for t in range(N_simulation_steps):
        A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
    
    return A[Eval_pts[0]]

# All Outputs - at every 25 steps (so 4 outputs)
def Y_t(var=[], N = 200):
    """
    Gray_Scott Model, outputs concentration at one grid point every 25 time points
    
    :param N: (int), Size of the NxN grid

    :return: (float), concentration at one grid point
    """
    A, B = get_initial_configuration(N)
    
    # simulation steps
    N_simulation_steps = 100    
    
    DA, DB, f, k = var
    delta_t = 1
    
    Eval_pts = [(50, 50)]
    
    concentration = []
    
    for t in range(N_simulation_steps):
        A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
        
        if t%25==0 or t==N_simulation_steps:
            concentration.append(A[Eval_pts[0]])
        
    return concentration

def Y_MO(var=[], N = 200):
    """
    Gray_Scott Model, outputs concentration at multiple (4) grid points
    
    :param N: (int), Size of the NxN grid

    :return: (float), concentration at four grid points
    """
    A, B = get_initial_configuration(N)
    
    # simulation steps
    N_simulation_steps = 100    
    
    DA, DB, f, k = var
    delta_t = 1
    
    Eval_pts = [(50, 50),(40, 40),(30, 30),(20, 30) ]
    
    for t in range(N_simulation_steps):
        A, B = gray_scott_update(A, B, DA, DB, f, k, delta_t)
    
    
    return A[Eval_pts[0]],A[Eval_pts[1]],A[Eval_pts[2]],A[Eval_pts[3]]