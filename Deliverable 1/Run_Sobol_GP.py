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
import pickle
import multiprocess
from Sobol_indices import get_sobol_indices__via_saltelli,get_sobol

     
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
    
            
if __name__ == '__main__':
    get_sobol_for_all_ISR_imlemented_models()
