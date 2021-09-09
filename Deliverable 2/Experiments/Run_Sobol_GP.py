# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:17:46 2021

@author: Cillian
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:10:52 2021

@author: Cillian
"""
import argparse
import pickle
import numpy as np
import multiprocess
#from Sobol_indices import get_sobol_indices__via_saltelli,get_sobol


import copy
#import numpy as np
import utils
#import pickle
#import Gray_Scott
#import multiprocess
import GPy


###################################################
# Calculate Sobol Indices using Saltelli Algorithm 
###################################################
def obtain_samples_GS(M, p_range): 
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
    ff = np.random.uniform(p_range[4][1],p_range[4][0],M) 
    
    return np.array([DA, DB, k,f,ff])

def get_sobol_indices__via_saltelli(Y,model,parameter_ranges,name = "GP_ALM", M=10000000, q_names = ['DA', 'DB', 'f', 'k'], sema = None):
    """This function uses the Saltelli Algorithm to compute first and total order Sobol indices



    :param Y: model currently being trained, either a GPy object or the Gray_Scott model
    :param model: (Str) indicator function for a GPy object (GP) or the Gray_Scott (GS) model
    :param parameter_ranges: (list), The upper and lower bounds for each input parameter
        
    :param q_names: (list), Parameter Names
    :param M: (int), Number of function evaluations. M(p+2) evaluations will be performed, where p is the number of parameters

    :return: (Dict), first and total order Sobol indices as two separate pickle files
    """
    
    
    # Obtain Samples
    #A = utils.obtain_samples_GS(M,parameter_ranges)
    #B = utils.obtain_samples_GS(M,parameter_ranges)
    A = obtain_samples_GS(M,parameter_ranges)
    B = obtain_samples_GS(M,parameter_ranges)    
    print(A.shape)
    print("Constructed A and B...")
    
    #Construct Ci matrices using A and B
    C_matrices = {}
    for i in range(len(q_names)):
        Ci = copy.copy(B)
        Ci[i,:] = A[i,:]
        C_matrices[q_names[i]] = copy.copy(Ci)
        print("Computed a C matrix")
    
    if model=="GS":
        #Evaluate Y function M(p+2) evaluations
        YA = np.apply_along_axis(Y, 0, A)
        YB = np.apply_along_axis(Y, 0, B)
        YC = {q_name: np.apply_along_axis(Y, 0, Ci) for q_name, Ci in C_matrices.items()}
        print("GS")
        
    elif model == "GP":
        print("We're a GP")
        YA, _ = Y.predict(A.T)
        print("Made a prediation")
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
        
    print("name=", name)
    # Save values
    with open("Data2_SobolIndices_v3\S1"+ str(name)+ ".dump", "wb") as f:
        pickle.dump(S, f)  
        
    with open("Data2_SobolIndices_v3\ST"+ str(name) +".dump" , "wb") as f:
        pickle.dump(ST, f) 
        
    if sema:
        sema.release()
        


def get_sobol(Y,model,name,parameter_ranges,Q_names = ['DA', 'DB', 'f', 'k'], sema = None):
    """Helper function to allow multi-threading/processing. Calls Sobol function, computes sobol indices and then releases node for other jobs
    
    """
    
    try:
        S, ST = get_sobol_indices__via_saltelli(Y,model,parameter_ranges,name, M=10000000, q_names = Q_names)
        
        with open("Data2_SobolIndices_v3\S1"+ str(name)+ ".dump", "wb") as f:
            pickle.dump(S, f)  
            
        with open("Data2_SobolIndices_v3\ST"+ str(name) +".dump" , "wb") as f:
            pickle.dump(ST, f) 
    
        if sema:
            sema.release()
        
    except:
        if sema:
            sema.release()

     
def get_sobol_for_all_ISR_imlemented_models(n_sims = 100, Implemented_strategies = ["ALM", "ALC", "IMSE", "MMSE"]):
    """Function for computing Sobol indices for ISR3D. GPy files are read in and Sobol Indices are computed, using multiprocessing.
    TODO: Changes are needed to make this function more generic
    
    
    :param n_sims: (int) The number of simulations ran for the implemented active learning strategy
    :param Implemented_strategies: (list) List of implemented active learning strategies
    
    :return: (pickle file), first and total order Sobol indices are saved as dictionaries for each simulation and each implemented active learning method.
    """
    jobs = []
    
    # Set parameter ranges
    #ps = [[0.01,0.002], [0.001,0.0001],[0.1,0.01],[0.2,0.1]]
    #Y = Gray_Scott.Y_T
    model = "GP"
    parameter_ranges = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    q_names = ['Endo', 'balloon', 'max_strain', 'prob_fenestration']
    
    sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    Implemented_strategies = ["ALC","ALM", "SLRGP", "IMSE", "MMSE"]

    for AL in Implemented_strategies:

        for i in range(n_sims):
            
            # Change to location and naming convention of your GP files
            s = "_All_Volume_"
            Y = pickle.load(open("GP_data_ISR3D\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(i) +".dump" , "rb"))

            name = "ISR_T_volume_" + str(AL)+ "_GP_rep_"+str(i)
            
            sema.acquire()
            
            p = multiprocess.Process(target=get_sobol, args=(Y,model,name,parameter_ranges,q_names, sema))
            
            jobs.append(p)
            p.start()
    
import GPy
def get_sobol_for_SLRGP_various_times(n_sims = 30, K = 5):
    #s = "_All_Volume_"
    jobs = []
    
    # Set parameter ranges
    #ps = [[0.01,0.002], [0.001,0.0001],[0.1,0.01],[0.2,0.1]]
    #Y = Gray_Scott.Y_T
    model = "GP"
    parameter_ranges = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.02]]
    q_names = ['Endo', 'balloon', 'max_strain', 'prob_fenestration']
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(1))
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    Outputs = np.load('ISR3D_data\LumenData.npy')
    print(Outputs.shape)

    # Get the Lumen area at final time step
    #Volumes_d10 = Outputs[:,241,:].sum(axis = 1)*0.03125
    #Volumes_d20 = Outputs[:,481,:].sum(axis = 1)*0.03125
    #Volumes_d30 = Outputs[:,-1,:].sum(axis = 1)*0.03125
    
    #Volumes_d01 = Outputs[:,25,:].sum(axis = 1)*0.03125
    #Volumes_d05 = Outputs[:,121,:].mean(axis = 1)
    Volumes_d10 = Outputs[:,241,:].mean(axis = 1)
    #Volumes_d15 = Outputs[:,361,:].mean(axis = 1)
    Volumes_d20 = Outputs[:,481,:].mean(axis = 1)
    #Volumes_d25 = Outputs[:,601,:].mean(axis = 1)
    Volumes_d30 = Outputs[:,-1,:].mean(axis = 1)
    
    for ii in range(n_sims):
        print("Starting iteration ", ii)
        for kk in range(K):
            GP_inputs = pickle.load(open("NoBias_GP_data_ISR3D_perf5\GP_SLRGP_NoiseBounds_slice__All_Volume__Inputs_rep_"+str(ii)+"_k_"+str(kk)+".dump", "rb"))
            #Outputs_d05 = np.zeros((50,1))
            Outputs_d10 = np.zeros((50,1))
            #Outputs_d15 = np.zeros((50,1))
            Outputs_d20 = np.zeros((50,1))
            #Outputs_d25 = np.zeros((50,1))
            Outputs_d30 = np.zeros((50,1))
            
            # Outputs should have been saved!
            Inputs_indx = []
            for i in range(50):
                Inputs_indx.append( np.where(Inputs==GP_inputs[i])[0][0]  )
                
            for _,j in enumerate(Inputs_indx):
                #Outputs_d05[_] = Volumes_d05[j]
                Outputs_d10[_] = Volumes_d10[j]
                #Outputs_d15[_] = Volumes_d15[j]
                Outputs_d20[_] = Volumes_d20[j]
                #Outputs_d25[_] = Volumes_d25[j]
                Outputs_d30[_] = Volumes_d30[j]
                
            
            # Number of parameters
            n = 4
            
            # # Number of output dimensions
            # P = 6
            
            # kern_list = [GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True) for _ in range(P)]
                
            # GP_models = [GPy.models.GPRegression(GP_inputs,Outputs_d10,kernel = kern_list[_] ) for _ in range(P) ]    
            
            # for GP in GP_models:
            #     GP['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
            #     GP.optimize()
            #     GP.optimize_restarts(num_restarts = 10, verbose=False)
        
            # Define kernel
            kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
            #kb = GPy.kern.Bias(input_dim=n)
            k = kg# + kb  
            # Train regression function
            GP_model10 = GPy.models.GPRegression(GP_inputs,Outputs_d10,kernel = k)
        
            # optimise parameters
            GP_model10['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
            GP_model10.optimize()
            GP_model10.optimize_restarts(num_restarts = 10, verbose=False)
            
            # Use the model to make predictions for unevaluated points
            #GP_means_10, _ = GP_model10.predict(Inputs)
            
            # # Define kernel
            kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
            #kb = GPy.kern.Bias(input_dim=n)
            k = kg# + kb
            
            # # Train regression function
            GP_model20 = GPy.models.GPRegression(GP_inputs,Outputs_d20,kernel = k)
            
            # optimise parameters
            GP_model20['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
            GP_model20.optimize()
            GP_model20.optimize_restarts(num_restarts = 10, verbose=False)
        
            #Use the model to make predictions for unevaluated points
            GP_means_20, _ = GP_model20.predict(Inputs)
            
            # Define kernel
            kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)
            #kb = GPy.kern.Bias(input_dim=n)
            k = kg 
            
            # Train regression function
            GP_model30 = GPy.models.GPRegression(GP_inputs,Outputs_d30,kernel = k)
            
            # optimise parameters
            GP_model30['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
            GP_model30.optimize()
            GP_model30.optimize_restarts(num_restarts = 10, verbose=False)
            
            # Use the model to make predictions for unevaluated points
            #GP_means_30, _ = GP_model30.predict(Inputs)
                
            # for _,k in enumerate(kern_list):
            #     name = "Quick_init20_d"+str(5 + 5*_)+"_rep"+str(ii)+"_k_"+str(kk)
            #     sema.acquire()
            #     p = multiprocess.Process(target=get_sobol_indices__via_saltelli, args=(GP_models[_],model,parameter_ranges,name,10000000,q_names, sema))
            #     jobs.append(p)
            #     p.start()
            
            name = "SLRGP_NoiseBounds_init20_d10_rep"+str(ii)+"_k_"+str(kk)
            sema.acquire()
            p = multiprocess.Process(target=get_sobol_indices__via_saltelli, args=(GP_model10,model,parameter_ranges,name,10000000,q_names, sema))
            jobs.append(p)
            p.start()
            
            name = "SLRGP_NoiseBounds_init20_d20_rep"+str(ii)+"_k_"+str(kk)
            sema.acquire()
            p = multiprocess.Process(target=get_sobol_indices__via_saltelli, args=(GP_model20,model,parameter_ranges,name,10000000,q_names, sema))
            jobs.append(p)
            p.start()            
            
            name = "SLRGP_NoiseBounds_init20_d30_rep"+str(ii)+"_k_"+str(kk)
            sema.acquire()
            p = multiprocess.Process(target=get_sobol_indices__via_saltelli, args=(GP_model30,model,parameter_ranges,name,10000000,q_names, sema))
            jobs.append(p)
            p.start()


 
def get_sobol_one_models(n_sims = 1, Implemented_strategies = ["ALM", "ALC", "IMSE", "MMSE"]):
    """Function for computing Sobol indices for ISR3D. GPy files are read in and Sobol Indices are computed, using multiprocessing.
    TODO: Changes are needed to make this function more generic
    
    
    :param n_sims: (int) The number of simulations ran for the implemented active learning strategy
    :param Implemented_strategies: (list) List of implemented active learning strategies
    
    :return: (pickle file), first and total order Sobol indices are saved as dictionaries for each simulation and each implemented active learning method.
    """
    jobs = []
    
    # Set parameter ranges
    #ps = [[0.01,0.002], [0.001,0.0001],[0.1,0.01],[0.2,0.1]]
    #Y = Gray_Scott.Y_T
    model = "GP"
    parameter_ranges = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.02]]
    q_names = ['Endo', 'balloon', 'max_strain', 'prob_fenestration']
    
    sema = multiprocess.Semaphore(1)
    Implemented_strategies = ["SLRGP_InitiallyFixNoise"]
    ii = 0

    for AL in Implemented_strategies:

        for kk in range(5):
            
            # Change to location and naming convention of your GP files
            #s = "_All_Volume_"
            Y = pickle.load(open("NoBias_GP_data_ISR3D_perf5\GP_SLRGP_InitiallyFixNoise_slice__All_Volume__GP_rep_"+str(ii)+"_k_"+str(kk)+".dump", "rb"))
            name = "ISR_" + str(AL)+ "_GP_rep_"+str(kk)
            
            sema.acquire()
            
            p = multiprocess.Process(target=get_sobol, args=(Y,model,name,parameter_ranges,q_names, sema))
            
            jobs.append(p)
            p.start()        
    
def get_sobol_for_Multiple_GPs():
    """Function for computing Sobol indices for ISR3D. GPy files are read in and Sobol Indices are computed, using multiprocessing.
    TODO: Changes are needed to make this function more generic
    
    
    :param n_sims: (int) The number of simulations ran for the implemented active learning strategy
    :param Implemented_strategies: (list) List of implemented active learning strategies
    
    :return: (pickle file), first and total order Sobol indices are saved as dictionaries for each simulation and each implemented active learning method.
    """
    jobs = []
    
    # Set parameter ranges
    #ps = [[0.01,0.002], [0.001,0.0001],[0.1,0.01],[0.2,0.1]]
    #Y = Gray_Scott.Y_T
    model = "GP"
    AL = None
    parameter_ranges = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    q_names = ['Endo', 'balloon', 'max_strain', 'prob_fenestration']
    
    sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
   # Implemented_strategies = ["ALC","ALM", "SLRGP", "IMSE", "MMSE"]

    for i in range(128):
        
        # Change to location and naming convention of your GP files
        s = "_All_Volume_"
        Y = pickle.load(open("ISR3D_LOOCV\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(i) +".dump" , "rb"))

        name = "ISR_T_volume_" + str(AL)+ "_GP_rep_"+str(i)
        
        sema.acquire()
        
        p = multiprocess.Process(target=get_sobol, args=(Y,model,name,parameter_ranges,q_names, sema))
        
        jobs.append(p)
        p.start() 
        
def get_sobol_for_MOGP():
    jobs = []
    model = "GP"
    
    parameter_ranges = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    q_names = ['Endo', 'balloon', 'max_strain', 'prob_fenestration']
    sema = multiprocess.Semaphore(1)
    #Data = ["Max","Mean","SD"]
    for i in range(30):
        
        for kk in range(5):
            
            # Change to location and naming convention of your GP files
            Y = pickle.load(open("ISR3D_MOGP_D30_LumenArea\GP_layers3_mean_option_PCA_it_"+str(i)+"_k_"+str(kk)+".dump", "rb"))
            
            for _, target in enumerate(["Max","Mean","SD"]):
            
                name = "MOGP_"+str(target) +"_"+ str(i)+ "_rep_"+str(kk)
                
                sema.acquire()
                
                p = multiprocess.Process(target=get_sobol, args=(Y[_],model,name,parameter_ranges,q_names, sema))
                
                jobs.append(p)
                p.start()   

def get_sobol_for_NewData_MOGP():
    jobs = []
    model = "GP"
            
    parameter_ranges = [[0.399,0.133], [20,10],[0.1,0],[0.17,0.32],[(3*18)/np.pi,10000]]
    q_names = ['flow_velocity', 'endo_endpoint', 'fenestration_probability', 'deployment_depth','curvature_radius']
    sema = multiprocess.Semaphore(1)
    #Data = ["Max","Mean","SD"]
    for i in range(int(iteration)-1,int(iteration)):
        
            
        # Change to location and naming convention of your GP files
        Y = pickle.load(open("Data2_SobolIndices_v3\GP_AllNewData.dump", "rb"))
        
        for _, target in enumerate(["Max","Mean","SD"]):
        
            name = "MOGP_"+str(target) +"_"+ str(i)
            
            sema.acquire()
            
            p = multiprocess.Process(target=get_sobol, args=(Y[_],model,name,parameter_ranges,q_names, sema))
            
            jobs.append(p)
            p.start()     
           
# if __name__ == '__main__':
# #     #get_sobol_for_all_ISR_imlemented_models()
# #     get_sobol_for_SLRGP_various_times()
# #     #get_sobol_one_models()
#     get_sobol_for_NewData_MOGP()
            

if __name__ == '__main__':
    
    #Setup the argument parser class
    parser = argparse.ArgumentParser(prog='Experiments program',
                                      description='''\
            Performs K-fold validation

              ''')
    #We use the optional switch -- otherwise it is mandatory
    parser.add_argument('iteration', action='store', help='Run id', default=1)
    #Run the argument parser
    args = parser.parse_args()
    #Extract our value or default
    iteration = args.iteration

    get_sobol_for_NewData_MOGP(iteration)