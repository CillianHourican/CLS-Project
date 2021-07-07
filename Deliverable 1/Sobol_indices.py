# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 21:27:30 2021

@author: Cillian
"""

import copy
import numpy as np
import utils
import pickle
import Gray_Scott
import multiprocess
import GPy


###################################################
# Calculate Sobol Indices using Saltelli Algorithm 
###################################################
    
def get_sobol_indices__via_saltelli(Y,model,parameter_ranges,name = "GP_ALM", M=10000000, q_names = ['DA', 'DB', 'f', 'k']):
    """This function uses the Saltelli Algorithm to compute first and total order Sobol indices



    :param Y: model currently being trained, either a GPy object or the Gray_Scott model
    :param model: (Str) indicator function for a GPy object (GP) or the Gray_Scott (GS) model
    :param parameter_ranges: (list), The upper and lower bounds for each input parameter
        
    :param q_names: (list), Parameter Names
    :param M: (int), Number of function evaluations. M(p+2) evaluations will be performed, where p is the number of parameters

    :return: (Dict), first and total order Sobol indices as two separate pickle files
    """
    
    
    # Obtain Samples
    A = utils.obtain_samples_GS(M,parameter_ranges)
    B = utils.obtain_samples_GS(M,parameter_ranges)
    
    print("Constructed A and B...")
    
    #Construct Ci matrices using A and B
    C_matrices = {}
    for i in range(len(q_names)):
        Ci = copy.copy(B)
        Ci[i,:] = A[i,:]
        C_matrices[q_names[i]] = copy.copy(Ci)
    
    if model=="GS":
        #Evaluate Y function M(p+2) evaluations
        YA = np.apply_along_axis(Y, 0, A)
        YB = np.apply_along_axis(Y, 0, B)
        YC = {q_name: np.apply_along_axis(Y, 0, Ci) for q_name, Ci in C_matrices.items()}
        print("GS")
        
    elif model == "GP":
        YA, _ = Y.predict(A.T)
        YB, _ = Y.predict(B.T)
        YA = YA.squeeze() # Remove the extra dimension
        YB = YB.squeeze()
        
        YC = {q_name: Y.predict(Ci.T)[0] for q_name, Ci in C_matrices.items()}
        
        
    else:
        assert 1==1, "Valid model not chosen. Options include GS, GP "
        
    print("Constructed Ci matrices using A and B")
        
    
    #Estimate mean f02
    f02 = np.mean(YA)*np.mean(YB)
    
    #Calculate Sobol indices
    S = {}
    for q_name, YCi in YC.items():
        S[q_name] = (np.dot(YA, YCi)/M - f02)/((np.dot(YA, YA)/M)-f02)
        
    print("Constructed Si...")
            
    for q_name, Si in S.items():
        print("Si ", q_name, Si)
        #print(f'Si {q_name}: {Si:.5f}')
    
    print('SUM Si', sum(list(S.values())))
        
    #Calculate total Sobol indices
    ST = {}
    for q_name, YCi in YC.items():
        ST[q_name] = 1 - (np.dot(YB, YCi)/M - f02)/((np.dot(YA, YA)/M)-f02)
    
    #Print total Sobol indices
    for q_name, STi in ST.items():
        print("ST ", q_name, STi)
    
    print('SUM STi', sum(list(ST.values())))
        
            
    # Save values
    with open("GP_analysis\S1"+ str(name)+ ".dump", "wb") as f:
        pickle.dump(S, f)  
        
    with open("GP_analysis\ST"+ str(name) +".dump" , "wb") as f:
        pickle.dump(ST, f) 
        


def get_sobol(Y,model,name,parameter_ranges,Q_names = ['DA', 'DB', 'f', 'k'], sema = None):
    """Helper function to allow multi-threading/processing. Calls Sobol function, computes sobol indices and then releases node for other jobs
    
    """
    
    try:
        get_sobol_indices__via_saltelli(Y,model,parameter_ranges,name, M=10000000, q_names = Q_names)
    
        if sema:
            sema.release()
        
    except:
        if sema:
            sema.release()
 