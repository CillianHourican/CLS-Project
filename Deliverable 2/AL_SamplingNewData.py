# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:08:02 2021

@author: Cillian
"""


import time
start = time.time()

import pickle
import numpy as np
import GPy
import AL_strategies
from numpy import linalg as LA
from sklearn.decomposition import PCA

def MOGP_PCA(Z,U,GP_models,kern_list,GP_Yvals):

    for GP in GP_models:
       GP['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
       GP.optimize()
       GP.optimize_restarts(num_restarts = 10, verbose=False)  
    
    variances = [AL_strategies.SLRGP_z_variances(Z,U,GP_models[_],kern_list[_]) for _ in range(P)]
    
    
    all_current_vars = np.vstack(( variances[_][0] for _ in range(P) )).T
    all_potential_vars = np.vstack(( variances[_][1] for _ in range(P) )).T
    
    pca = PCA(n_components=0.999, svd_solver='full')
    pca.fit_transform(all_current_vars)
    
    # Eigenvalues
    print("PCA Singular values", pca.singular_values_)
    
    # Project onto PC space
    pc_current_vars = pca.transform(all_current_vars) 
    pc_potential_vars = pca.transform(all_potential_vars) 
    pc_ave_reductions = np.zeros((pc_current_vars.shape[0],1))
    
    for i in range(pc_current_vars.shape[0]):
        pc_ave_reductions[i] = (LA.norm(np.array([pc_current_vars[i],pc_potential_vars[i]]),2))
    
    indx = np.argmax(pc_ave_reductions)

    new_point = U[indx]

    return(new_point,indx,pc_ave_reductions)


def SLRGP_PCA_Batch(Z,U, GP_models,k, n_pts = 5,Y=None):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    chosen_pts, chosen_indices = [],[]
    kern_list = [GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)  for _ in range(P)]
    var_reduction = []
        
    for pt in range(n_pts):
        pt, index,red_var = MOGP_PCA(Z,U, GP_models,k,Y)
        Z = np.vstack((Z,pt))

        print("...Choose one point...")
        var_reduction.append(red_var)
            
        for _ in range(P):
            Y[_] = np.vstack((Y[_],GP_models[_].predict(np.array(pt)[:,None].T)[0]))
         
        # Remove new_pt from set of uneval pts
        U = np.delete(U, index,axis = 0)
                            
        chosen_pts.append(pt)
        chosen_indices.append(index)
        
        GP_models = [GPy.models.GPRegression(Z,Y[_], kernel = kern_list[_],normalizer=True  ) for _ in range(P) ]  
    
        for GP in GP_models:
           GP['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
           GP.optimize()
           GP.optimize_restarts(num_restarts = 10, verbose=False)  
        
    return chosen_pts, chosen_indices,var_reduction

Max_data=np.load('ISR3D_Data_Collected/Max_PercentLost_d30.npy') 
Mean_data = np.load('ISR3D_Data_Collected/Mean_PercentLost_d30.npy') 
SD_data = np.load('ISR3D_Data_Collected/SD_PercentLost_d30.npy') 

Max_data=np.load('ISR3D_Data_Collected/Max_PercentLost_new.npy') 
Mean_data = np.load('ISR3D_Data_Collected/Mean_PercentLost_new.npy') 
SD_data = np.load('ISR3D_Data_Collected/SD_PercentLost_new.npy') 

Data = [SD_data,Max_data,Mean_data]
evaluated_indices = [101,11,128,135,152,164,17,182,184,198,210,224,225,241,51,60,64,78,93,95]


#------------New Batches! ---------------------------#

b1Max_data=np.load('ISR3D_Data_Collected/New_Batch1/Max_PercentLost_d30.npy') 
b1Mean_data = np.load('ISR3D_Data_Collected/New_Batch1/Mean_PercentLost_d30.npy') 
b1SD_data = np.load('ISR3D_Data_Collected/New_Batch1/SD_PercentLost_d30.npy') 

Max_data = np.hstack((Max_data,b1Max_data))
Mean_data = np.hstack((Mean_data,b1Mean_data))
SD_data = np.hstack((SD_data,b1SD_data))

Data = [SD_data,Max_data,Mean_data]
evaluated_indices = [101,11,128,135,152,164,17,182,184,198,210,224,225,241,51,60,64,78,93,95,255,142,186,122,113]

b2Max_data=np.load('ISR3D_Data_Collected/New_Batch2/Max_PercentLost_d30.npy') 
b2Mean_data = np.load('ISR3D_Data_Collected/New_Batch2/Mean_PercentLost_d30.npy') 
b2SD_data = np.load('ISR3D_Data_Collected/New_Batch2/SD_PercentLost_d30.npy') 

Max_data = np.hstack((Max_data,b2Max_data))
Mean_data = np.hstack((Mean_data,b2Mean_data))
SD_data = np.hstack((SD_data,b2SD_data))

evaluated_indices = [101,11,128,135,152,164,17,182,184,198,210,224,225,241,51,60,64,78,93,95,255,142,186,122,113,191,253,252,251,254]
Data = [SD_data,Max_data,Mean_data]

b3Max_data=np.load('ISR3D_Data_Collected/New_Batch3/Max_PercentLost_d30.npy') 
b3Mean_data = np.load('ISR3D_Data_Collected/New_Batch3/Mean_PercentLost_d30.npy') 
b3SD_data = np.load('ISR3D_Data_Collected/New_Batch3/SD_PercentLost_d30.npy') 

Max_data = np.hstack((Max_data,b3Max_data))
Mean_data = np.hstack((Mean_data,b3Mean_data))
SD_data = np.hstack((SD_data,b3SD_data))

evaluated_indices = [101,11,128,135,152,164,17,182,184,198,210,224,225,241,51,60,64,78,93,95,255,142,186,122,113,191,253,252,251,254,62,81,250,3,29]
Data = [SD_data,Max_data,Mean_data]

b4Max_data=np.load('ISR3D_Data_Collected/New_Batch4/Max_PercentLost_d30.npy') 
b4Mean_data = np.load('ISR3D_Data_Collected/New_Batch4/Mean_PercentLost_d30.npy') 
b4SD_data = np.load('ISR3D_Data_Collected/New_Batch4/SD_PercentLost_d30.npy') 

Max_data = np.hstack((Max_data,b4Max_data))
Mean_data = np.hstack((Mean_data,b4Mean_data))
SD_data = np.hstack((SD_data,b4SD_data))

evaluated_indices = [101,11,128,135,152,164,17,182,184,198,210,224,225,241,51,60,64,78,93,95,255,142,186,122,113,191,253,252,251,254,62,81,250,3,29,30,223,63,127,126]
Data = [SD_data,Max_data,Mean_data]

b5Max_data=np.load('ISR3D_Data_Collected/New_Batch5/Max_PercentLost_d30.npy') 
b5Mean_data = np.load('ISR3D_Data_Collected/New_Batch5/Mean_PercentLost_d30.npy') 
b5SD_data = np.load('ISR3D_Data_Collected/New_Batch5/SD_PercentLost_d30.npy') 

Max_data = np.hstack((Max_data,b5Max_data))
Mean_data = np.hstack((Mean_data,b5Mean_data))
SD_data = np.hstack((SD_data,b5SD_data))

evaluated_indices = [101,11,128,135,152,164,17,182,184,198,210,224,225,241,51,60,64,78,93,95,255,142,186,122,113,191,253,252,251,254,62,81,250,3,29,30,223,63,127,126,190,31,158,94,222]
Data = [SD_data,Max_data,Mean_data]

#------------ Setup GP -----------------------------------#
All_inputs = np.load("ISR3D_Data_Collected/input_list_numberseq.npy")
n = All_inputs.shape[1]
P = 3 # Number of Output dimensions
Batch_size = 5

Z = All_inputs[evaluated_indices,:]
U = np.delete(All_inputs,evaluated_indices,axis = 0)

kern_list = [GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)  for _ in range(P)]
GP_Yvals = [ Data[_][:,None] for _ in range(P)]
GP_models = [GPy.models.GPRegression(Z,GP_Yvals[_], kernel = kern_list[_],normalizer=True ) for _ in range(P) ] 

#------------ To Select New Data -------------------------------#
chosen_pts, chosen_indices, var_reduction = SLRGP_PCA_Batch(Z,U, GP_models,k=kern_list, n_pts = 5,Y=GP_Yvals)
print(chosen_indices)

# Then transform indices back to original indices!
for pt in chosen_pts:
    print( np.where(pt==All_inputs))
    
#------------ Optimise and Save GP -----------------------------#

for GP in GP_models:
   GP['Gaussian_noise.variance'].unconstrain()#constrain_bounded(1e-1,1e2)
   GP.optimize()
   GP.optimize_restarts(num_restarts = 10, verbose=False)  
   
with open("GP_NoNoiseBounds.dump", "wb") as f:
    pickle.dump(GP_models, f)  

    


#------------- Plots for Reduction in Predictive Variance ----------#
StdVsMean = np.zeros((P,2))
Var_sum = np.zeros((P,2))
Var_max =np.zeros((P,2))
Var_std = np.zeros((P,2))
#-------------------------------------------------------------------#
prediative_variances = []
StdVsMean2 = np.zeros((P,6))
Var_sum2 = np.zeros((P,6))
Var_max2 =np.zeros((P,6))
Var_std2 = np.zeros((P,6))
n_batches = 1
n_batch = 5
for _ in range(P):
    #prediative_variances[_] = np.vstack((prediative_variances[_],GP_models[_].predict(np.array(All_inputs)[:,None].T)[1]))
    a,b = GP_models[_].predict(np.array(All_inputs))
    StdVsMean2[_,n_batch] = b.std()/a.mean()
    Var_sum2[_,n_batch] = b.sum()
    Var_max2[_,n_batch] = b.max()
    Var_std2[_,n_batch] = b.std()
    #Data = [SD_data,Max_data,Mean_data]

for i in range(5):
    StdVsMean2[:,i] = StdVsMean[:,i]
    Var_sum2[:,i] = Var_sum[:,i]
    Var_max2[:,i] = Var_max[:,i]
    Var_std2[:,i] = Var_std[:,i]

StdVsMean = StdVsMean2
Var_sum = Var_sum2
Var_max = Var_max2
Var_std = Var_std2

np.save("StdVsMean.npy", StdVsMean)
np.save("Var_sum.npy", Var_sum)
np.save("Var_max.npy", Var_max)
np.save("Var_std.npy", Var_std)

# np.hstack((Max_data,b5Max_data))
    
import matplotlib.pyplot as plt
plt.plot(Var_sum[2,:])
plt.title("Mean GP: Sum of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("Sum of predictive variance")
plt.savefig("Plots\Batch Perfformance\Mean_sum_varALL")
plt.show()

plt.plot(Var_max[2,:])
plt.title("Mean GP: Max of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("Max of predictive variance")
plt.savefig("Plots\Batch Perfformance\Mean_max_varALL")
plt.show()

plt.plot(Var_std[2,:])
plt.title("Mean GP: Standard Deviation of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("SD of predictive variance")
plt.savefig("Plots\Batch Perfformance\Mean_SD_varALL")
plt.show()

plt.plot(np.abs(StdVsMean[2,:]))
plt.title("Mean GP: Standard Deviation vs mean of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("SD(GP var) / Mean(GP_means)")
plt.savefig("Plots\Batch Perfformance\Mean_ratio_varALL")
plt.show()

#- MAX --#
plt.plot(Var_sum[1,:])
plt.title("Max GP: Sum of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("Sum of predictive variance")
plt.savefig("Plots\Batch Perfformance\Max_sum_varALL")
plt.show()

plt.plot(Var_max[1,:])
plt.title("Max GP: Max of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("Max of predictive variance")
plt.savefig("Plots\Batch Perfformance\Max_max_varALL")
plt.show()

plt.plot(Var_std[1,:])
plt.title("Max GP: Standard Deviation of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("SD of predictive variance")
plt.savefig("Plots\Batch Perfformance\Max_SD_varALL")
plt.show()

plt.plot(np.abs(StdVsMean[1,:]))
plt.title("Max GP: Standard Deviation vs mean of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("SD(GP var) / Mean(GP_means)")
plt.savefig("Plots\Batch Perfformance\Max_ratio_varALL")
plt.show()

#- SD --#
plt.plot(Var_sum[0,:])
plt.title("SD GP: Sum of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("Sum of predictive variance")
plt.savefig("Plots\Batch Perfformance\SD_sum_varALL")
plt.show()

plt.plot(Var_max[0,:])
plt.title("SD GP: Max of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("Max of predictive variance")
plt.savefig("Plots\Batch Perfformance\SD_max_varALL")
plt.show()

plt.plot(Var_std[0,:])
plt.title("SD GP: Standard Deviation of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("SD of predictive variance")
plt.savefig("Plots\Batch Perfformance\SD_SD_varALL")
plt.show()

plt.plot(np.abs(StdVsMean[0,:]))
plt.title("SD GP: Standard Deviation vs mean of predictive variance")
plt.xlabel("Batch Number")
plt.ylabel("SD(GP var) / Mean(GP_means)")
plt.savefig("Plots\Batch Perfformance\SD_ratio_varALL")
plt.show()