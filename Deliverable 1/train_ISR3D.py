# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:07:17 2021

@author: Cillian
"""
import GPy
import AL_strategies
from utils import *
import numpy as np
#import Gray_Scott
#from Sobol_indices import get_sobol_indices__via_saltelli
import GP_model_ISR3D as GP_model
import pickle
#import time
from sklearn.metrics import mean_absolute_error, mean_squared_error,max_error,r2_score

import multiprocess

def train_model(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps, run_id,s, sema = None):
    try:
        pickle.load(open("GP_data_ISR3D\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id) +".dump", "rb"))
        if sema:
            sema.release()
    
    except:
        try:
            GP, GP_Inputs = GP_model.GP_model(retrieved_Inputs,retrieved_Outputs,AL_strategy,parameter_ranges=ps)
            with open("GP_data_ISR3D\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id) +".dump" , "wb") as f:
              pickle.dump(GP, f) 
              
            with open("GP_data_ISR3D\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id) +".dump" , "wb") as f:
              pickle.dump(GP_Inputs, f) 
                    
            if sema:
                sema.release()
        except:
            trained_flag = 0
            attempt_nr = 0
            while trained_flag == 0:
                try:
                    GP, GP_Inputs = GP_model.GP_model(retrieved_Inputs,retrieved_Outputs,AL_strategy,parameter_ranges=ps)
                    with open("GP_data_ISR3D\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id) +".dump" , "wb") as f:
                      pickle.dump(GP, f) 
                      
                    with open("GP_data_ISR3D\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id) +".dump" , "wb") as f:
                      pickle.dump(GP_Inputs, f) 
                
                    trained_flag = 1
                    if sema:
                        sema.release()
                    
                except:
                    attempt_nr += 1
                    print("Still not training.... having an error somewhere. On attempt number"+str(attempt_nr))
                    
def train_validate_model(test_Inputs,test_Outputs,k,retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps, run_id,s, sema = None):
    try:
        pickle.load(open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump", "rb"))
        if sema:
            sema.release()
    
    except:
        try:
            GP, GP_Inputs = GP_model.GP_model(retrieved_Inputs,retrieved_Outputs,AL_strategy,parameter_ranges=ps)
            # with open("GP_data_ISR3D_val\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
            #   pickle.dump(GP, f) 
              
            # with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
            #   pickle.dump(GP_Inputs, f) 
            
            print("I'm fully trained! Now time to be tested!!")
            # Predict validation data
            GP_means, GP_vars = GP.predict(test_Inputs)
              
            # Get RMSE of Validation data
            RMSE = mean_squared_error(GP_means, test_Outputs, squared=False)
              
            # Get Max Error
            MAX = max_error(GP_means, test_Outputs)
              
            # R2 Score
            r2 = r2_score(test_Outputs,GP_means)
            
            scores = [RMSE,MAX,r2]
            
            with open("GP_data_ISR3D_val\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(GP, f) 
              
            with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(GP_Inputs, f) 
        
            with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Val_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
              pickle.dump(scores, f) 
            print("I'm fully trained and tested!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
            if sema:
                sema.release()
        except:
            trained_flag = 0
            attempt_nr = 0
            while trained_flag == 0:
                try:
                    GP, GP_Inputs = GP_model.GP_model(retrieved_Inputs,retrieved_Outputs,AL_strategy,parameter_ranges=ps)
                    # with open("GP_data_ISR3D_val\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                    #   pickle.dump(GP, f) 
                      
                    # with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                    #   pickle.dump(GP_Inputs, f) 
                      
                    print("I'm fully trained! Now time to be tested!!")
                
                    # Predict validation data
                    GP_means, GP_vars = GP.predict(test_Inputs)
                      
                    # Get RMSE of Validation data
                    RMSE = mean_squared_error(GP_means, test_Outputs, squared=False)
                      
                    # Get Max Error
                    MAX = max_error(GP_means, test_Outputs)
                      
                    # R2 Score
                    r2 = r2_score(test_Outputs,GP_means)
                    
                    scores = [RMSE,MAX,r2]
                    
                    with open("GP_data_ISR3D_val\GP_" + str(AL)+ "_slice_"+str(s)+"_GP_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(GP, f) 
                      
                    with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Inputs_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(GP_Inputs, f) 
                
                    with open("GP_data_ISR3D_val\GP_"+str(AL)+ "_slice_"+str(s)+"_Val_rep_"+str(run_id)+"_k_" +str(k) +".dump" , "wb") as f:
                      pickle.dump(scores, f) 
              
                    trained_flag = 1
                    if sema:
                        sema.release()
                    
                except:
                    attempt_nr += 1
                    print("Still not training.... having an error somewhere. On attempt number"+str(attempt_nr))
  
    #except:
    #    if sema:
    #        sema.release()
        
def run_sims_for_ISR3D_Retrieved_slice_Data(n_sims = 30):
    
    jobs = []
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb"))
    data = np.load('ISR3D_data\LumenData.npy')
    
    retrieved_Inputs = Inputs.to_numpy()
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(5))
    
    # For different slices
    for s in range(10,128,10):
        retrieved_Outputs = data[:,-1,s]
    

    
        
        
        # AL_strategy = AL_strategies.ALM
        # AL = "ALM"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        # AL_strategy = AL_strategies.ALC
        # AL = "ALC"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        # AL_strategy = AL_strategies.SLRGP
        # AL = "SLRGP"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        AL_strategy = AL_strategies.SLRGP_Batch
        AL = "SLRGP_Batch_5"
        for i in range(n_sims):
            
            sema.acquire()
            
            p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
            jobs.append(p)
            p.start()
        
            
        # AL_strategy = AL_strategies.IMSE
        # AL = "IMSE"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        # AL_strategy = AL_strategies.MMSE
        # AL = "MMSE"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
            
        # AL_strategy = AL_strategies.SLRV
        # AL = "SLRV"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        

def run_sims_for_ISR3D_Retrieved_Volume_Data(n_sims = 100):
    
    jobs = []
    
    # For naming conventions
    s = "_All_Volume_"
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb"))
    data = np.load('ISR3D_data\LumenData.npy')
    
    retrieved_Inputs = Inputs.to_numpy()
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    
    # Convert to Areas
    retrieved_Outputs = data[:,-1,:].sum(axis = 1)*0.03125
    
    # For different slices
    #for s in range(10,128,10):
    #    retrieved_Outputs = data[:,-1,s]
    

    
        
        
    # AL_strategy = AL_strategies.ALM
    # AL = "ALM"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
        
    # AL_strategy = AL_strategies.ALC
    # AL = "ALC"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
            
    AL_strategy = AL_strategies.SLRGP
    AL = "SLRGP"
    for i in range(13,n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
        jobs.append(p)
        p.start()
        
            
    # AL_strategy = AL_strategies.IMSE
    # AL = "IMSE"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
        
    # AL_strategy = AL_strategies.MMSE
    # AL = "MMSE"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
            
        # AL_strategy = AL_strategies.SLRV
        # AL = "SLRV"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
   
from sklearn.model_selection import cross_validate
scoring = ["neg_root_mean_squared_error", "r2", "max_error"]

def run_sims_for_Volume_with_CrossValidation(n_sims = 100):
    
    jobs = []
    
    # For naming conventions
    s = "_All_Volume_"
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb"))
    data = np.load('ISR3D_data\LumenData.npy')
    
    retrieved_Inputs = Inputs.to_numpy()
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    
    # Convert to Areas
    retrieved_Outputs = data[:,-1,:].sum(axis = 1)*0.03125
    
    # For different slices
    #for s in range(10,128,10):
    #    retrieved_Outputs = data[:,-1,s]
    

    
        
        
    # AL_strategy = AL_strategies.ALM
    # AL = "ALM"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
        
    # AL_strategy = AL_strategies.ALC
    # AL = "ALC"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
            
    AL_strategy = AL_strategies.SLRGP
    AL = "SLRGP"
    for i in range(13,n_sims):
        
        sema.acquire()
        
        p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
        jobs.append(p)
        p.start()
        
            
    # AL_strategy = AL_strategies.IMSE
    # AL = "IMSE"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
        
    # AL_strategy = AL_strategies.MMSE
    # AL = "MMSE"
    # for i in range(n_sims):
        
    #     sema.acquire()
        
    #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
        
    #     jobs.append(p)
    #     p.start()
            
        # AL_strategy = AL_strategies.SLRV
        # AL = "SLRV"
        # for i in range(n_sims):
            
        #     sema.acquire()
            
        #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
            
        #     jobs.append(p)
        #     p.start()
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        
from sklearn.model_selection import KFold
def run_sims_for_Volume_with_CrossValidation(n_sims = 50):
    
    jobs = []
    
    # For naming conventions
    s = "_All_Volume_"
    
    Inputs = pickle.load(open("ISR3D_data\InputList.pkl", "rb")).to_numpy()
    data = np.load('ISR3D_data\LumenData.npy')
    
    #retrieved_Inputs = Inputs.to_numpy()
    
    # Set parameter ranges
    ps = [[20,10], [1.5,0.5],[1.8,1.2],[0.1,0.0]]
    
    #sema = multiprocess.Semaphore(int(multiprocess.cpu_count()-2))
    sema = multiprocess.Semaphore(int(3))
    
    # Convert to Areas
    retrieved_Outputs = data[:,-1,:].sum(axis = 1)*0.03125
    
    # For different slices
    #for s in range(10,128,10):
    #    retrieved_Outputs = data[:,-1,s]
    
    # Create range 1 to 128
    rn = range(1,128)
    
    for i in range(30,n_sims):
        print("Starting iteration " +str(i))
        
        kf10 = KFold(n_splits=5, shuffle=True)
        k = 0
        for train_index, test_index in kf10.split(rn):
            
            retrieved_Inputs = Inputs[train_index]
            Outputs = retrieved_Outputs[train_index]
            
            test_Inputs = Inputs[train_index]
            test_Outputs = retrieved_Outputs[train_index]
            
            # AL_strategy = AL_strategies.ALM
            # AL = "ALM"
            # #for i in range(n_sims):
                
            # sema.acquire()
            
            # p = multiprocess.Process(target=train_validate_model, args=(test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
                
            # AL_strategy = AL_strategies.ALC
            # AL = "ALC"
            # #for i in range(n_sims):
                
            # sema.acquire()
            
            # p = multiprocess.Process(target=train_validate_model, args=(test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
                    
            # AL_strategy = AL_strategies.SLRGP
            # AL = "SLRGP"
            # #for i in range(n_sims):
                
            # sema.acquire()
            
            # p = multiprocess.Process(target=train_validate_model, args=(test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
                
                    
            # AL_strategy = AL_strategies.IMSE
            # AL = "IMSE"
            # #for i in range(n_sims):
                
            # sema.acquire()
            
            # p = multiprocess.Process(target=train_validate_model, args=(test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
                
            # AL_strategy = AL_strategies.MMSE
            # AL = "MMSE"
            # #for i in range(n_sims):
                
            # sema.acquire()
            
            # p = multiprocess.Process(target=train_validate_model, args=(test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
            
            # jobs.append(p)
            # p.start()
            
            AL_strategy = AL_strategies.SLRGP_Batch
            AL = "SLRGP_Batch_2"
            #for i in range(n_sims):
                
            sema.acquire()
            
            p = multiprocess.Process(target=train_validate_model, args=(test_Inputs,test_Outputs,k,retrieved_Inputs,Outputs,AL_strategy,AL,ps,i,s, sema))
              
            jobs.append(p)
            p.start()
        
                
            k += 1
                
            # AL_strategy = AL_strategies.SLRV
            # AL = "SLRV"
            # for i in range(n_sims):
                
            #     sema.acquire()
                
            #     p = multiprocess.Process(target=train_model, args=(retrieved_Inputs,retrieved_Outputs,AL_strategy,AL,ps,i,s, sema))
                
            #     jobs.append(p)
            #     p.start()
        
    print("Number of jobs: " + str(len(jobs)))    
    count = 0
    for p in jobs:
        count += 1
        p.join()
        print("jobs done...", count)
        

        
if __name__ == '__main__':
    #run_sims_for_ISR3D_Retrieved_slice_Data()
    #run_sims_for_ISR3D_Retrieved_Volume_Data()
    run_sims_for_Volume_with_CrossValidation()