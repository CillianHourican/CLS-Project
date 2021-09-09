# -*- coding: utf-8 -*-
"""
Created on Sat May 29 23:32:58 2021

@author: Cillian
"""
import time
start = time.time()
import sys
import argparse

#from scipy import linalg
import pickle
import numpy as np
#import plotly.graph_objects as go
import GPy
import AL_strategies
#from AL_strategies import calculate_laplacian_in_parts, calculate_laplacian
from scipy.stats.mstats import gmean
from numpy import linalg as LA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error


def Build_local_surrogates(n,P, Inputs,Volumes, combine_option,test_inputs, test_outputs, max_eval):
       
    #if init_size == None:
    #    init_size = 20
    #print("Total Inputs given",Inputs.shape )
    # Randomly pick 20 starting points
    indx = np.random.choice(Inputs.shape[0], size = 5, replace=False) #np.random.randint(Inputs.shape[0], size = 20)
    GP_Inputs = Inputs[indx,:]
    #print("GP_Inputs", GP_Inputs.shape)
    
    # Define set of unevaluated points
    uneval_pts = np.delete(Inputs,indx,axis = 0)
    #print("uneval_pts", uneval_pts.shape)
    
    kern_list = [GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True) for _ in range(P)]
    
    RMSE_metrics = [ [] for  _ in range(P) ]
    MAX_metrics = [ [] for  _ in range(P) ]
    r2_metrics = [ [] for  _ in range(P) ]
    MAPE_metrics = [ [] for  _ in range(P) ]
    
  
    # Unevaluated Outputs
    GP_Outputs = [ np.delete(Volumes[_],indx,axis = 0) for _ in range(P)]
    
    # Retrived Outputs
    GP_Yvals = [ Volumes[_][indx] for _ in range(P)]
    print("Initial indices", indx, "GP Inputs shape",GP_Inputs.shape, "GP_Yvals[0] shape", GP_Yvals[0][:,None].shape )
    
    GP_models = [GPy.models.GPRegression(GP_Inputs,GP_Yvals[_][:,None], kernel = kern_list[_] ) for _ in range(P) ]    
 
    for GP in GP_models:
        GP['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
        #print(kern_list[0])
        #GP['bias.variance'].constrain_bounded(1e-10,1e-9)
        GP.optimize()
        GP.optimize_restarts(num_restarts = 10, verbose=False)
    
    for it in range(1,max_eval):
        print("GP it", it)
        #print(kern_list[0])
        
        Z = GP_Inputs
        U = uneval_pts
        #X = Inputs
        
        ave_reductions = [AL_strategies.SLRGP_ave_reductions2(Z,U,GP_models[_],kern_list[_]) for _ in range(P)]
    
        
        # Combine Laplacians for each layer -> Many options here
        if combine_option == 'sum':
            LX = np.sum(ave_reductions, axis = 0)
            indx = np.argmax(LX)
            #new_point = U[indx]
            
        elif combine_option == 'multiply':
            LX = np.prod(ave_reductions, axis = 0)
            indx = np.argmax(LX)
            #new_point = U[indx]
            
        elif combine_option == 'max':
            LX = np.max(ave_reductions, axis = 0)
            indx = np.argmax(LX)
            #new_point = U[indx]
            
        elif combine_option == 'geometric':
            LX = gmean( np.abs(ave_reductions) )
            indx = np.argmin(LX)
            #new_point = U[indx]
            
        elif combine_option == 'harmonic':
            LX = np.power( sum( np.power(ave_reductions, -1) ), -1)*P
            indx = np.argmax(LX)
            #new_point = U[indx]

        elif combine_option == 'alternating_selection':
            LX = ave_reductions[it%P]
            indx = np.argmax(LX)
            #new_point = U[indx]
            
        elif combine_option == 'testing':
            LX = ave_reductions[1]
            indx = np.argmax(LX)
            #new_point = U[indx]
            
        elif combine_option == 'ResponseDimension0':
            LX = ave_reductions[0]
            indx = np.argmax(LX)
            
        elif combine_option == 'ResponseDimension1':
            LX = ave_reductions[1]
            indx = np.argmax(LX)
            
        elif combine_option == 'ResponseDimension2':
            LX = ave_reductions[2]
            indx = np.argmax(LX)
           
            
        else:
            raise Exception('This is not a valid mean option. Chose from maximum,arithmetic,geometric,harmonic or minimum') 
        
        
        new_point = U[indx]
        
        # Evaluate New Point
        #New_eval = Outputs[indx]
        #print("New_eval",New_eval)
        
        # Add new_pt to set of evaluated points
        GP_Inputs = np.vstack((GP_Inputs,new_point))

        #GP_Yvals = [ Volumes[_][indx] for _ in range(P)]
        New_eval = [ GP_Outputs[_][indx] for _ in range(P)]
        
        for _ in range(P):
            GP_Yvals[_] = np.hstack((GP_Yvals[_],New_eval[_]))
        
        #print("GP_Yvals",GP_Yvals)
        
        # Add Y(new_pt) to set of Y vals
        #Y_vals = np.vstack((Y_vals,New_eval))
 

        # Remove new_pt from set of uneval pts
        uneval_pts = np.delete(uneval_pts,indx,axis = 0)
        #print("uneval_pts",uneval_pts.shape)
    
        
        # Remove Y(new_pt) from uneval outputs
        GP_Outputs = [ np.delete(GP_Outputs[_],indx,axis = 0) for _ in range(P)]
        #print("After another ")
        #print("New indices", indx, "GP Inputs shape",GP_Inputs.shape, "GP_Yvals[0] shape", GP_Yvals[0][:,None].shape )
    
        
        GP_models = [GPy.models.GPRegression(GP_Inputs,GP_Yvals[_][:,None], kernel = kern_list[_] ) for _ in range(P) ]    
        #GP_means = []
        for _,GP in enumerate(GP_models):
            GP['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
            #print(kern_list[0])
            #GP['bias.variance'].constrain_bounded(1e-10,1e-9)
            GP.optimize()
            GP.optimize_restarts(num_restarts = 10, verbose=False)
            GP_means, GP_vars = GP.predict(test_inputs)
        
            RMSE_metrics[_].append( mean_squared_error(GP_means, test_outputs[_], squared=False) ) 
          
            # Get Max Error
            MAX_metrics[_].append(max_error(GP_means, test_outputs[_]) )
          
            # R2 Score
            r2_metrics[_].append(r2_score(test_outputs[_],GP_means) )
            
            MAPE_metrics[_].append(mean_absolute_percentage_error(test_outputs[_],GP_means))
        
        #print(kern_list[0])
        
    return(GP_models, GP_Inputs, RMSE_metrics, MAX_metrics, r2_metrics,MAPE_metrics)

def MO_SLRGP_with_PCA(n,P, Inputs,Volumes,combine_option,test_inputs, test_outputs, max_eval):
    indx = np.random.choice(Inputs.shape[0], size = 5, replace=False) #np.random.randint(Inputs.shape[0], size = 20)
    GP_Inputs = Inputs[indx,:]
    
    RMSE_metrics = [ [] for  _ in range(P) ]
    MAX_metrics = [ [] for  _ in range(P) ]
    r2_metrics = [ [] for  _ in range(P) ]
    MAPE_metrics = [ [] for  _ in range(P) ]
    
    # Define set of unevaluated points
    uneval_pts = np.delete(Inputs,indx,axis = 0)
    #print("uneval_pts", uneval_pts.shape)
    
    kern_list = [GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)  for _ in range(P)]
    
    # Unevaluated Outputs
    GP_Outputs = [ np.delete(Volumes[_],indx,axis = 0) for _ in range(P)]
    
    # Retrived Outputs
    GP_Yvals = [ Volumes[_][indx] for _ in range(P)]
    
    GP_models = [GPy.models.GPRegression(GP_Inputs,GP_Yvals[_][:,None], kernel = kern_list[_] ) for _ in range(P) ]  
    
    for GP in GP_models:
        GP['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
        GP.optimize()
        GP.optimize_restarts(num_restarts = 10, verbose=False)
           
        
    for it in range(max_eval):
        print("GP it", it)
        Z = GP_Inputs
        U = uneval_pts
        
        #ave_reductions = [AL_strategies.SLRGP_ave_reductions(Z,U,GP_models[_],kern_list[_]) for _ in range(P)]
        variances = [AL_strategies.SLRGP_z_variances(Z,U,GP_models[_],kern_list[_]) for _ in range(P)]
        
        
        all_current_vars = np.vstack(( variances[_][0] for _ in range(P) )).T
        all_potential_vars = np.vstack(( variances[_][1] for _ in range(P) )).T
        
        #current_vars, potential_vars = [AL_strategies.SLRGP_z_variances(Z,U,GP_models[_],kern_list[_]) for _ in range(P)]
        
        #all_current_vars = np.vstack(( current_vars[0],current_vars[1],current_vars[2]))
        #all_potential_vars = np.vstack(( potential_vars[0],potential_vars[1],potential_vars[2]))
        
        
        #pca = PCA(n_components=1)
        #pca.fit(all_current_vars)
        
        pca = PCA(n_components=0.999, svd_solver='full')
        pca.fit_transform(all_current_vars)
        
        # Eigenvalues
        print("PCA Singular values", pca.singular_values_)
        
        # Eigenvectors
        #pca.components_
        
        # Project onto PC space
        pc_current_vars = pca.transform(all_current_vars) 
        pc_potential_vars = pca.transform(all_potential_vars) 
        pc_ave_reductions = np.zeros((pc_current_vars.shape[0],1))#pc_current_vars - pc_potential_vars
        ## Q. Is this the Euclidean (l2) norm? LA.norm(a, 2)
        for i in range(pc_current_vars.shape[0]):
            pc_ave_reductions[i] = (LA.norm(np.array([pc_current_vars[i],pc_potential_vars[i]]),2))
        
        indx = np.argmax(pc_ave_reductions)
        new_point = U[indx]
        
        # Add new_pt to set of evaluated points
        GP_Inputs = np.vstack((GP_Inputs,new_point))
    
        #GP_Yvals = [ Volumes[_][indx] for _ in range(P)]
        New_eval = [ GP_Outputs[_][indx] for _ in range(P)]
        
        for _ in range(P):
            GP_Yvals[_] = np.hstack((GP_Yvals[_],New_eval[_]))
        #print("GP_Yvals",GP_Yvals)
        
        # Add Y(new_pt) to set of Y vals
        #Y_vals = np.vstack((Y_vals,New_eval))
     
    
        # Remove new_pt from set of uneval pts
        uneval_pts = np.delete(uneval_pts,indx,axis = 0)
        #print("uneval_pts",uneval_pts.shape)
    
        
        # Remove Y(new_pt) from uneval outputs
        GP_Outputs = [ np.delete(GP_Outputs[_],indx,axis = 0) for _ in range(P)]
        
        GP_models = [GPy.models.GPRegression(GP_Inputs,GP_Yvals[_][:,None], kernel = kern_list[_] ) for _ in range(P) ]    
     
       # for GP in GP_models:
       #     GP.optimize()
       #     GP.optimize_restarts(num_restarts = 10, verbose=False)
    
        #GP_means = []
        for _,GP in enumerate(GP_models):
            GP['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
            #print(kern_list[0])
            #GP['bias.variance'].constrain_bounded(1e-10,1e-9)
            GP.optimize()
            GP.optimize_restarts(num_restarts = 10, verbose=False)
            GP_means, GP_vars = GP.predict(test_inputs)
        
            RMSE_metrics[_].append( mean_squared_error(GP_means, test_outputs[_], squared=False) ) 
          
            # Get Max Error
            MAX_metrics[_].append(max_error(GP_means, test_outputs[_]) )
          
            # R2 Score
            r2_metrics[_].append(r2_score(test_outputs[_],GP_means) )
            
            MAPE_metrics[_].append(mean_absolute_percentage_error(test_outputs[_],GP_means))
        
        #print(kern_list[0])
        
    return(GP_models, GP_Inputs, RMSE_metrics, MAX_metrics, r2_metrics,MAPE_metrics)

# Train the model
import multiprocess
from sklearn.metrics import mean_squared_error,max_error,r2_score



def train_validate_model(save_location,test_inputs,test_outputs,run_id,k, n,P, Inputs,Volumes, combine_option,GP_setup, max_eval, sema = None):
    try:    
        trained_model, trained_inputs, RMSE_metrics, MAX_metrics, r2_metrics, MAPE_metrics = GP_setup(n,P, Inputs,Volumes, combine_option,test_inputs,test_outputs, max_eval)
        
        predicted_means,predicted_vars = [], [] 
        RMSE, MAX,r2,MAPE = [],[],[],[]
        
        for _,GP in enumerate(trained_model):
            #GP.optimize()
            #GP.optimize_restarts(num_restarts = 10, verbose=False)
            mean, var = GP.predict(test_inputs)
            predicted_means.append([mean])
            predicted_vars.append([var])
            
            # Get RMSE of Validation data
            RMSE.append(mean_squared_error(mean, test_outputs[_], squared=False) )
              
            # Get Max Error
            MAX.append( max_error(mean, test_outputs[_]) )
              
            # R2 Score
            r2.append(r2_score(test_outputs[_],mean) )
            
            MAPE.append(mean_absolute_percentage_error(test_outputs[_],mean))
        
        print("I'm fully trained! Now time to be tested!!")
            
        with open(str(save_location)+"GP_layers" + str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(trained_model, f) 
          
        with open(str(save_location)+"Inputs_layers" + str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(trained_inputs, f) 
    
        with open(str(save_location)+"RMSE_layers" + str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(RMSE, f) 
          
        with open(str(save_location)+"MAX_layers" + str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(MAX, f) 
        with open(str(save_location)+"MAPE_layers" + str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(MAPE, f)           
        with open(str(save_location)+"R2_layers" + str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(r2, f) 
        with open(str(save_location)+"GP_"+ str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(test_inputs, f)      
          
        with open(str(save_location)+"OutputData_"+ str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(test_outputs, f)    
          
        with open(str(save_location)+"TimeRMSE_"+ str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(RMSE_metrics, f)  
        with open(str(save_location)+"TimeMaxError_"+ str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(MAX_metrics, f) 
        with open(str(save_location)+"TimeR2_"+ str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(r2_metrics, f)      
        with open(str(save_location)+"TimeMAPE_"+ str(P)+ "_mean_option_"+str(combine_option)+"_it_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
          pickle.dump(MAPE_metrics, f)   
        if sema:
            sema.release()
            
    except:
        if sema:
            sema.release()        
        

def run_MOGP_CrossValidation(n_sims = 1):
    
    jobs = []
    start = time.time()
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    n = len(ps)
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(6))
    #sema = None
    
    # Create range 1 to 128
    rn = range(1,128)
    
    # Read in Data Set
    Inputs = np.load('ISR3D_data\input_list_numberseq.npy') 
    
    # Save Location
    save_location = "ISR3D_MOGP_D30_LumenArea2\MOGP2"
    
    Max_D30 = np.load('ISR3D_data\Max_PercentLost_LumenArea.npy') 
    Mean_D30 = np.load('ISR3D_data\Mean_PercentLost_LumenArea.npy') 
    SD_D30 = np.load('ISR3D_data\SD_PercentLost_LumenArea.npy') 

    # Max_D20 = np.load('ISR3D_data\Max_PercentLost_LumenArea480.npy') 
    # Mean_D20 = np.load('ISR3D_data\Mean_PercentLost_LumenArea480.npy') 
    # SD_D20 = np.load('ISR3D_data\SD_PercentLost_LumenArea480.npy') 

    # Max_D10 = np.load('ISR3D_data\Max_PercentLost_LumenArea240.npy') 
    # Mean_D10 = np.load('ISR3D_data\Mean_PercentLost_LumenArea240.npy') 
    # SD_D10 = np.load('ISR3D_data\SD_PercentLost_LumenArea240.npy')     

    
    # Volumes = [Volumes_d01,Volumes_d02,Volumes_d03,Volumes_d04,Volumes_d05,Volumes_d06,Volumes_d07,Volumes_d08,Volumes_d09,Volumes_d10,Volumes_d11,Volumes_d12,Volumes_d13,Volumes_d14,Volumes_d15,Volumes_d16,Volumes_d17,Volumes_d18,Volumes_d19,Volumes_d20,Volumes_d21,Volumes_d22,Volumes_d23,Volumes_d24,Volumes_d25,Volumes_d26,Volumes_d27,Volumes_d28,Volumes_d29,Volumes_d30]

    #Volumes = [Volumes_d10,Volumes_d20,Volumes_d30]
    #Data = [Max_D30,Mean_D30,SD_D30,Max_D20,Mean_D20,SD_D20,Max_D10 ,Mean_D10,SD_D10]
    Data = [Max_D30,Mean_D30,SD_D30]
    
    #Volumes = [Volumes_d02,Volumes_d04,Volumes_d06,Volumes_d08,Volumes_d10,Volumes_d12,Volumes_d14,Volumes_d16,Volumes_d18,Volumes_d20,Volumes_d22,Volumes_d24,Volumes_d26,Volumes_d28,Volumes_d30]

    
    for i in range(n_sims):
        #Volumes = [Volumes_d10,Volumes_d20,Volumes_d30] 
        max_eval = 45
        
        P = len(Data)
        kf10 = KFold(n_splits=5, shuffle=True)
    
    # for i in range(1,n_sims):
        print("Starting iteration " +str(i))
        
        k = 0
        
        for train_index, test_index in kf10.split(rn):
            
            train_inputs = Inputs[train_index]
            train_outputs = [ Data[_][train_index] for _ in range(P) ]
            
            test_inputs = Inputs[test_index]
            test_outputs = [ Data[_][test_index] for _ in range(P) ]
            
            # combine_option = 'sum'
            # #print(combine_option)
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()
            #train_validate_model(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema )


            # combine_option = 'multiply'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()

            # combine_option = 'max'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()

            # combine_option = 'geometric'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()                       

            # combine_option = 'harmonic'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start() 

            # combine_option = 'alternating_selection'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start() 
            
            # combine_option = 'ResponseDimension0'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()   
            
            # combine_option = 'ResponseDimension1'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()  
            
            combine_option = 'PCA'
            sema.acquire()
            p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option,MO_SLRGP_with_PCA, max_eval, sema ))
            jobs.append(p)
            p.start()
            
            # combine_option = 'ResponseDimension2'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start() 
                
            k += 1
            
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
    end = time.time()
    print("Time taken = ", end - start)
        
def run_MO_SLRGP_with_PCA_CrossValidation(n_sims = 30):
    
    jobs = []
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    n = len(ps)
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(8))
    
    # Create range 1 to 128
    rn = range(1,128)
    
    # Read in Data Set
    #Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    # Outputs = np.load('ISR3D_data\LumenData.npy')
    
    # #Volumes_d05 = Outputs[:,121,:].sum(axis = 1)*0.03125
    # Volumes_d10 = Outputs[:,241,:].sum(axis = 1)*0.03125
    # #Volumes_d15 = Outputs[:,361,:].sum(axis = 1)*0.03125
    # Volumes_d20 = Outputs[:,481,:].sum(axis = 1)*0.03125
    # #Volumes_d25 = Outputs[:,601,:].sum(axis = 1)*0.03125
    # Volumes_d30 = Outputs[:,-1,:].sum(axis = 1)*0.03125
    
    # # Volumes_d01 = Outputs[:,25,:].sum(axis = 1)*0.03125
    # Volumes_d02 = Outputs[:,49,:].mean(axis = 1)
    # # Volumes_d03 = Outputs[:,73,:].sum(axis = 1)*0.03125
    # Volumes_d04 = Outputs[:,97,:].mean(axis = 1)
    # # Volumes_d05 = Outputs[:,121,:].sum(axis = 1)*0.03125
    # Volumes_d06 = Outputs[:,145,:].mean(axis = 1)
    # # Volumes_d07 = Outputs[:,169,:].sum(axis = 1)*0.03125
    # Volumes_d08 = Outputs[:,193,:].mean(axis = 1)
    # # Volumes_d09 = Outputs[:,217,:].sum(axis = 1)*0.03125
    # Volumes_d10 = Outputs[:,241,:].mean(axis = 1)
    # # Volumes_d11 = Outputs[:,265,:].sum(axis = 1)*0.03125
    # Volumes_d12 = Outputs[:,289,:].mean(axis = 1)
    # # Volumes_d13 = Outputs[:,313,:].sum(axis = 1)*0.03125
    # Volumes_d14 = Outputs[:,337,:].mean(axis = 1)
    # # Volumes_d15 = Outputs[:,361,:].sum(axis = 1)*0.03125
    # Volumes_d16 = Outputs[:,385,:].mean(axis = 1)
    # # Volumes_d17 = Outputs[:,409,:].sum(axis = 1)*0.03125
    # Volumes_d18 = Outputs[:,433,:].mean(axis = 1)
    # # Volumes_d19 = Outputs[:,457,:].sum(axis = 1)*0.03125
    # Volumes_d20 = Outputs[:,481,:].mean(axis = 1)
    # # Volumes_d21 = Outputs[:,505,:].sum(axis = 1)*0.03125
    # Volumes_d22 = Outputs[:,529,:].mean(axis = 1)
    # # Volumes_d23 = Outputs[:,553,:].sum(axis = 1)*0.03125
    # Volumes_d24 = Outputs[:,577,:].mean(axis = 1)
    # # Volumes_d25 = Outputs[:,601,:].sum(axis = 1)*0.03125
    # Volumes_d26 = Outputs[:,601,:].mean(axis = 1)
    # # Volumes_d27 = Outputs[:,625,:].sum(axis = 1)*0.03125
    # Volumes_d28 = Outputs[:,649,:].mean(axis = 1)
    # # Volumes_d29 = Outputs[:,673,:].sum(axis = 1)*0.03125
    # Volumes_d30 = Outputs[:,-1,:].mean(axis = 1)
    
    # Read in Data Set
    Inputs = np.load('ISR3D_data\input_list_numberseq.npy') 
    
    # Save Location
    save_location = "ISR3D_MOGP_D30_LumenArea\MOGP2"
    
    Max_D30 = np.load('ISR3D_data\Max_PercentLost_LumenArea.npy') 
    Mean_D30 = np.load('ISR3D_data\Mean_PercentLost_LumenArea.npy') 
    SD_D30 = np.load('ISR3D_data\SD_PercentLost_LumenArea.npy') 

    # Max_D20 = np.load('ISR3D_data\Max_PercentLost_LumenArea480.npy') 
    # Mean_D20 = np.load('ISR3D_data\Mean_PercentLost_LumenArea480.npy') 
    # SD_D20 = np.load('ISR3D_data\SD_PercentLost_LumenArea480.npy') 

    # Max_D10 = np.load('ISR3D_data\Max_PercentLost_LumenArea240.npy') 
    # Mean_D10 = np.load('ISR3D_data\Mean_PercentLost_LumenArea240.npy') 
    # SD_D10 = np.load('ISR3D_data\SD_PercentLost_LumenArea240.npy')     

    
    # Volumes = [Volumes_d01,Volumes_d02,Volumes_d03,Volumes_d04,Volumes_d05,Volumes_d06,Volumes_d07,Volumes_d08,Volumes_d09,Volumes_d10,Volumes_d11,Volumes_d12,Volumes_d13,Volumes_d14,Volumes_d15,Volumes_d16,Volumes_d17,Volumes_d18,Volumes_d19,Volumes_d20,Volumes_d21,Volumes_d22,Volumes_d23,Volumes_d24,Volumes_d25,Volumes_d26,Volumes_d27,Volumes_d28,Volumes_d29,Volumes_d30]

    #Volumes = [Volumes_d10,Volumes_d20,Volumes_d30]
    #Data = [Max_D30,Mean_D30,SD_D30,Max_D20,Mean_D20,SD_D20,Max_D10 ,Mean_D10,SD_D10]
    Data = [Max_D30,Mean_D30,SD_D30]
    
    # #Volumes = [Volumes_d01,Volumes_d02,Volumes_d03,Volumes_d04,Volumes_d05,Volumes_d06,Volumes_d07,Volumes_d08,Volumes_d09,Volumes_d10,Volumes_d11,Volumes_d12,Volumes_d13,Volumes_d14,Volumes_d15,Volumes_d16,Volumes_d17,Volumes_d18,Volumes_d19,Volumes_d20,Volumes_d21,Volumes_d22,Volumes_d23,Volumes_d24,Volumes_d25,Volumes_d26,Volumes_d27,Volumes_d28,Volumes_d29,Volumes_d30]
    # Volumes = [Volumes_d02,Volumes_d04,Volumes_d06,Volumes_d08,Volumes_d10,Volumes_d12,Volumes_d14,Volumes_d16,Volumes_d18,Volumes_d20,Volumes_d22,Volumes_d24,Volumes_d26,Volumes_d28,Volumes_d30]

    
    
    for i in range(21,n_sims):
        #Volumes = [Volumes_d10,Volumes_d20,Volumes_d30] 
        max_eval = 45
        
        
        
        P = len(Data)
        kf10 = KFold(n_splits=5, shuffle=True)
    
    # for i in range(1,n_sims):
        print("Starting iteration " +str(i))
        
        k = 0
        
        for train_index, test_index in kf10.split(rn):
            
            train_inputs = Inputs[train_index]
            train_outputs = [ Data[_][train_index] for _ in range(P) ]
            
            test_inputs = Inputs[test_index]
            test_outputs = [ Data[_][test_index] for _ in range(P) ]
            
            combine_option = 'PCA999'
            sema.acquire()
            p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option,MO_SLRGP_with_PCA, max_eval, sema ))
            jobs.append(p)
            p.start()
            
            k += 1


    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
    end = time.time()
    print("Time taken = ", end - start)
        
#if __name__ == '__main__':
#    run_MOGP_CrossValidation()
    
    #run_MO_SLRGP_with_PCA_CrossValidation()
    

def CrossValidation_SLURM_JOBS(iteration):
    #n_sims = iteration*10
    # if int(iteration) == 1:
    #     n_min = 0
    #     n_max = 7
    # if int(iteration) == 2:
    #     n_min = 7
    #     n_max = 14
    # if int(iteration) == 3:
    #     n_min = 14
    #     n_max = 20
    #i = int(iteration) + 1
    jobs = []
    start = time.time()
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    n = len(ps)
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(4))
    #sema = None
    
    # Create range 1 to 128
    rn = range(1,128)
    
    # Read in Data Set
    Inputs = np.load('ISR3D_data\input_list_numberseq.npy') 
    
    # Save Location
    save_location = "ISR3D_MOGP_D30_Extras\MOGP_"
    
    Max_D30 = np.load('ISR3D_data\Max_PercentLost_LumenArea.npy') 
    Mean_D30 = np.load('ISR3D_data\Mean_PercentLost_LumenArea.npy') 
    SD_D30 = np.load('ISR3D_data\SD_PercentLost_LumenArea.npy') 
    Data = [Max_D30,Mean_D30,SD_D30]
    
    max_eval = 45
    
    P = len(Data)
    kf10 = KFold(n_splits=5, shuffle=True)

    #print("Starting iteration " +str(i))
    k = 0
    for i in range(int(iteration)-1,int(iteration)):
        k = 4
        #for i in range(int(iteration)-1, int(iteration)):
        for train_index, test_index in kf10.split(rn):
            
            train_inputs = Inputs[train_index]
            train_outputs = [ Data[_][train_index] for _ in range(P) ]
            
            test_inputs = Inputs[test_index]
            test_outputs = [ Data[_][test_index] for _ in range(P) ]
            
            # combine_option = 'sum'
            # #print(combine_option)
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()
            #train_validate_model(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema )
    
    
            combine_option = 'multiply'
            sema.acquire()
            p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            jobs.append(p)
            p.start()
    
            combine_option = 'max'
            sema.acquire()
            p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            jobs.append(p)
            p.start()
    
            combine_option = 'geometric'
            sema.acquire()
            p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            jobs.append(p)
            p.start()                       
    
            combine_option = 'harmonic'
            sema.acquire()
            p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            jobs.append(p)
            p.start() 
    
            combine_option = 'alternating_selection'
            sema.acquire()
            p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            jobs.append(p)
            p.start() 
            
            # combine_option = 'ResponseDimension0'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()   
            
            # combine_option = 'ResponseDimension1'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start()  
            
            # combine_option = 'PCA999'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option,MO_SLRGP_with_PCA, max_eval, sema ))
            # jobs.append(p)
            # p.start()
            
            # combine_option = 'ResponseDimension2'
            # sema.acquire()
            # p = multiprocess.Process(target=train_validate_model, args=(save_location,test_inputs,test_outputs,i,k,n,P, train_inputs,train_outputs, combine_option, Build_local_surrogates, max_eval, sema ))
            # jobs.append(p)
            # p.start() 
                
            #k += 1
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
    end = time.time()
    print("Time taken = ", end - start)        

if __name__ == '__main__':
    
    #Setup the argument parser class
    parser = argparse.ArgumentParser(prog='Experiments program',
                                     description='''\
            Performs K-fold validation

             ''')
    #We use the optional switch -- otherwise it is mandatory
    parser.add_argument('iteration', action='store', help='Run id', default=19)
    #Run the argument parser
    args = parser.parse_args()
    #Extract our value or default
    iteration = args.iteration

    CrossValidation_SLURM_JOBS(iteration)