# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 11:10:52 2021

@author: Cillian
"""
import Gray_Scott
import multiprocess
from Sobol_indices import get_sobol_indices__via_saltelli,get_sobol

    
def get_sobol_for_Gray_Scott(n_sims = 20):
    
    jobs = []
    
    model = "GS"
    
    # Set parameter ranges
    p_range = [[0.01,0.002], [0.001,0.0001],[0.1,0.01],[0.2,0.1]]
    Q_names = ['DA', 'DB', 'f', 'k']
    
    Y = Gray_Scott.Y
    
    sema = multiprocess.Semaphore(int(4)) #Semaphore(int(multiprocess.cpu_count()-2))
    

    for i in range(n_sims):#n_sims
        
        # Set save name
        name = "Grey_Scott_rep_"+str(i)
        
        sema.acquire()

        p = multiprocess.Process(target=get_sobol, args=(Y,model,name,p_range,Q_names, sema))
        
        jobs.append(p)
        p.start()
            
if __name__ == '__main__':
    get_sobol_for_Gray_Scott()