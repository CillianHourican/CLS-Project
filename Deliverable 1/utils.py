# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:50:22 2021

@author: Cillian
"""


import numpy as np
from pyDOE import lhs

def get_initial_pts(parameter_samples, parameter_ranges, criteria='center' ):
    """Get initial Latin Hypercube sample points and scale


    Args:
        parameter_samples (str): Number of samples required
         
        parameter_ranges (list of lists): Each inner list contains the range of 
         a parameter [upper bound, lower bound]
         
        criteria (str, optional): Latin Hypercube sampling method
    """
    
    # Number of parameters
    n = len(parameter_ranges)
    
    # Get LHS Samples
    lhs_pts = lhs(n, samples=parameter_samples, criterion=criteria)
    
    # Scale parameters to required ranges
    for i,j in enumerate(parameter_ranges):
        lhs_pts[:,i] = lhs_pts[:,i]*(j[0] - j[1]) + j[1]
        
    return( lhs_pts )

def get_grid_points(M):
    """Discretise input domain into a grid
    TOOD: Make more generic. Currently only works for specific Grey-Scott bounds

    Args:
        
        M (int): Number of equally spaced points to use for each input parameter
     
    Returns:
        np array with shape (M, num_parameters )
         
    """

    Grid_pts = np.empty((M**4,4))
    i = 0
    for DA in np.linspace(0.002,0.01,M) :
        for DB in np.linspace(0.0001,0.001,M):
            for k in np.linspace(0.01,0.1,M):
                for f in np.linspace(0.1,0.2,M):
                    Grid_pts[i,:] = np.array([DA, DB, k ,f])
                    i += 1
    return(Grid_pts  )


def obtain_samples_GS(M, p_range = [[0.01,0.002], [0.001,0.0001],[0.1,0.01],[0.2,0.1]]): 
    """Sample four input parameters from uniform distributions


    Args:
        M (int): Number of samples required
         
        p_range (list of lists): Each inner list contains the range of 
         a parameter [upper bound, lower bound]
                 
    Returns:
        np array with shape (M, num_parameters )
        
    """
    
    DA =  np.random.uniform(p_range[0][1],p_range[0][0],M) 
    DB =  np.random.uniform(p_range[1][1],p_range[1][0],M) 
    k =  np.random.uniform(p_range[2][1],p_range[2][0],M) 
    f = np.random.uniform(p_range[3][1],p_range[3][0],M) 
    
    return np.array([DA, DB, k,f])
