# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 20:32:40 2021

@author: Cillian
"""
import GPy
import AL_strategies
from utils import *
import numpy as np
import Gray_Scott

def GP_model(Y,AL_method,parameter_ranges,M, max_eval = 15, lhs_samples=20, lhs_stratrgy='center'):
    """Train Gaussian Process surrigate for a given function

    Parameters
    ----------
    Y : function
        The function we wish to approxinate
    AL_method : str
        Active learning stratrgy to use. See AL_strategies for available options (ALC,ALM)
    parameter_ranges: list of lists
    M : int 
        Number of equally spaced points to use for each input parameter
    max_eval: int
        Maximu number of function evaluations allowed, after LHS sampling
    lhs_samples: int
        Number of initial LHS samples to use
    lhs_stratrgy: str
        Sampling strategy. OPtions incluse 'center', 'maximum'
    
    
    Returns
    -------
    GPy regression model
        a trained gaussian process model
    """ 

    # Number of parameters
    n = len(parameter_ranges)
    
    # Define kernel
    #kg = GPy.kern.RBF(input_dim=n, variance=1., lengthscale=1.) # vector will be better
    kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
    kb = GPy.kern.Bias(input_dim=n)
    k = kg + kb
    
    # Initial scaled LHS samples
    Inputs = get_initial_pts(lhs_samples, parameter_ranges, criteria=lhs_stratrgy)
    print("Inputs Shape:",Inputs.shape )
    
    # Evaluate LHS points
    Y_vals =  np.apply_along_axis(Y, 0, Inputs.T ).reshape(lhs_samples,1)

    
    # Define set of unevaluated points
    uneval_pts = get_grid_points(M)
    
    # Train regression function
    model = GPy.models.GPRegression(Inputs,Y_vals,kernel = k)

    # optimise parameters
    #model.optimize()
    model.optimize_restarts(num_restarts = 10, verbose=False)

    # Use the model to make predictions for unevaluated points
    post_means, post_vars = model.predict(uneval_pts)
    
    # Apply active learning strategy to find new points
    new_point, new_indx = AL_method(Inputs, uneval_pts, model,k )
    
    for pt in range(len(new_indx)):
        # Remove the new point from the set of unevaluated points
        uneval_pts = np.delete(uneval_pts,new_indx[pt],axis = 0)
    
    it = 0
    while it < max_eval:
        
        for pt in range(len(new_indx)):
            
            # Evaluate new point
            New_eval = Y(new_point[pt])
        
            # Add new point to inputs and outputs
            Inputs = np.vstack((Inputs,new_point[pt]))
            Y_vals = np.vstack((Y_vals,New_eval))
    
        # Train regression model
        model = GPy.models.GPRegression(Inputs,Y_vals,k)
        model.constrain_positive('.*') 

        # optimize the model
        model.optimize()

        # Use the model to make predictions for unevaluated points
        post_means, post_vars = model.predict(uneval_pts)

        # Apply active learning strategy to find new points
        new_point, new_indx = AL_method(Inputs, uneval_pts, model,k )#AL_method(post_means, post_vars, uneval_pts )
    
        for pt in range(len(new_indx)):
            # Remove the new point from the set of unevaluated points
            uneval_pts = np.delete(uneval_pts,new_indx[pt],axis = 0)
    
        # Print something so you have soething to stare at while the code runs..
        print("IT;",it)
        it += 1
    
        if it % 50 ==0: 
            print("variances range:", min(post_vars), max(post_vars))
            
    
    return(model, Inputs)
            


def GP_model_t(Y,AL_method,parameter_ranges,M,T = 100, max_eval = 30, lhs_samples=20, lhs_stratrgy='center'):
    """Train Gaussian Process surrigate for a given function

    Parameters
    ----------
    Y : function
        The function we wish to approxinate
    AL_method : str
        Active learning stratrgy to use. See AL_strategies for available options (ALC,ALM)
    parameter_ranges: list of lists
    M : int 
        Number of equally spaced points to use for each input parameter
    max_eval: int
        Maximu number of function evaluations allowed, after LHS sampling
    lhs_samples: int
        Number of initial LHS samples to use
    lhs_stratrgy: str
        Sampling strategy. OPtions incluse 'center', 'maximum'
    
    
    Returns
    -------
    GPy regression model
        a trained gaussian process model
    """ 

    # Number of parameters
    n = len(parameter_ranges)
    
    # Define kernel
    kg = GPy.kern.RBF(input_dim=n, variance=1., lengthscale=1.) # vector will be better
    kb = GPy.kern.Bias(input_dim=n)
    k = kg + kb
    
    # Initial scaled LHS samples
    Inputs = get_initial_pts(lhs_samples, parameter_ranges, criteria=lhs_stratrgy)
    
    # Evaluate LHS points
    Y_vals =  np.apply_along_axis(Y, 0, Inputs.T ).T #reshape(lhs_samples,1) # T or 1?
    print(Y_vals.shape)
    
    # Define set of unevaluated points
    uneval_pts = get_grid_points(M)
    
    # Train regression function
    model = GPy.models.GPRegression(Inputs,Y_vals,kernel = k)

    # optimise parameters
    model.optimize()

    # Use the model to make predictions for unevaluated points
    post_means, post_vars = model.predict(uneval_pts)
    
    # Apply active learning strategy to find new points
    new_point, new_indx = AL_method(Inputs, uneval_pts, model,k )
    
    # Remove the new point from the set of unevaluated points
    uneval_pts = np.delete(uneval_pts,new_indx,axis = 0)
    
    it = 0
    while it < max_eval:
        
        # Evaluate new point
        New_eval = Y(new_point)
    
        # Add new point to inputs and outputs
        Inputs = np.vstack((Inputs,new_point))
        Y_vals = np.vstack((Y_vals,New_eval))
    
        # Train regression model
        model = GPy.models.GPRegression(Inputs,Y_vals,k)
        model.constrain_positive('.*') 

        # optimize the model
        model.optimize()

        # Use the model to make predictions for unevaluated points
        post_means, post_vars = model.predict(uneval_pts)

        # Apply active learning strategy to find new points
        new_point, new_indx = AL_method(Inputs, uneval_pts, model,k )#AL_method(post_means, post_vars, uneval_pts )
    
        # Remove the new point from the set of unevaluated points
        uneval_pts = np.delete(uneval_pts,new_indx,axis = 0)
    
        # Print something so you have soething to stare at while the code runs..
        print("IT;",it)
        it += 1
    
        if it % 50 ==0: 
            print("variances range:", min(post_vars), max(post_vars))
            
    
    return(model, Inputs)
# Testing the function
#p = [[0.01,0.002], [0.001,0.0001],[0.1,0.01],[0.2,0.1]]
#modz = GP_model(Gray_Scott.Y,AL_strategies.ALC,parameter_ranges = p, M = 5, max_eval = 30, lhs_samples=20, lhs_stratrgy='center')

#from Sobol_indices import get_sobol_indices__via_saltelli
#get_sobol_indices__via_saltelli(modz,"GP", 6)
