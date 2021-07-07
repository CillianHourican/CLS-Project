# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 14:53:00 2021

@author: Cillian
"""


import GPy
import AL_strategies
from utils import *
import numpy as np

# Inputs2 = Inputs.to_numpy()
# data2 = data[:,-1,1]

# parameter_ranges = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]

# import pickle
# import numpy as np
# data = np.load('ISR3D_data\LumenData.npy')

# data.shape  # instances, time steps, slices
# # (128, 721, 526)

# Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb"))


def GP_model(retrieved_Inputs,Outputs,AL_method,parameter_ranges,max_eval = 30):
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
    kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
    #GPy.kern.RBF(input_dim=n, variance=1., lengthscale=1.) # vector will be better
    kb = GPy.kern.Bias(input_dim=n)
    k = kg + kb
    
    # Initial scaled LHS samples
    #Inputs = get_initial_pts(lhs_samples, parameter_ranges, criteria=lhs_stratrgy)
    
    # Randomly pick 20 starting points
    indx = np.random.randint(retrieved_Inputs.shape[0], size = 20)
    Inputs = retrieved_Inputs[indx,:]
    
    # Evaluate LHS points
    Y_vals =  Outputs[indx][:,None]#.squeeze()
    Outputs = np.delete(Outputs,indx,axis = 0)
    
    # Define set of unevaluated points
    uneval_pts = np.delete(retrieved_Inputs,indx,axis = 0)
    
    # Train regression function
    model = GPy.models.GPRegression(Inputs,Y_vals,kernel = k)

    # optimise parameters
    #model.optimize()
    model.optimize_restarts(num_restarts = 10, verbose=False)

    # Use the model to make predictions for unevaluated points
    post_means, post_vars = model.predict(uneval_pts)
    
    # Apply active learning strategy to find new points
    new_point, new_indx = AL_method(Inputs, uneval_pts, model,k )
    
    # # Remove the new point from the set of unevaluated points
    # uneval_pts = np.delete(uneval_pts,new_indx,axis = 0)
    
    for pt in range(len(new_indx)):
        # Remove the new point from the set of unevaluated points
        uneval_pts = np.delete(uneval_pts,new_indx[pt],axis = 0)
    
    
    it = 0
    while it < int(max_eval/len(new_indx)):
        
        for pt in range(len(new_indx)):
                        
            # Evaluate new point
            New_eval = Outputs[new_indx].squeeze()[pt]
            Outputs = np.delete(Outputs,new_indx[pt],axis = 0)
        
            # Add new point to inputs and outputs
            Inputs = np.vstack((Inputs,new_point[pt]))
            Y_vals = np.vstack((Y_vals,New_eval))
        
        # # Evaluate new point
        # New_eval = Outputs[new_indx].squeeze()
        # Outputs = np.delete(Outputs,new_indx,axis = 0)
    
        # # Add new point to inputs and outputs
        # Inputs = np.vstack((Inputs,new_point))
        # Y_vals = np.vstack((Y_vals,New_eval))
    
        # Train regression model
        model = GPy.models.GPRegression(Inputs,Y_vals,k)
        model.constrain_positive('.*') 

        # optimize the model
        model.optimize()

        # Use the model to make predictions for unevaluated points
        post_means, post_vars = model.predict(uneval_pts)

        # Apply active learning strategy to find new points
        new_point, new_indx = AL_method(Inputs, uneval_pts, model,k )#AL_method(post_means, post_vars, uneval_pts )
    
        # # Remove the new point from the set of unevaluated points
        # uneval_pts = np.delete(uneval_pts,new_indx,axis = 0)
        
        for pt in range(len(new_indx)):
            # Remove the new point from the set of unevaluated points
            uneval_pts = np.delete(uneval_pts,new_indx[pt],axis = 0)
    
        # Print something so you have soething to stare at while the code runs..
        print("IT;",it)
        it += 1
    
        if it % 50 ==0: 
            print("variances range:", min(post_vars), max(post_vars))
            
    
    return(model, Inputs)
            