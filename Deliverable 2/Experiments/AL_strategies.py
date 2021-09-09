# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:52:21 2021

@author: Cillian
"""
import numpy as np
from scipy import linalg
#from tqdm import tqdm
#from sklearn.neighbors import NearestNeighbors
import GPy

from itertools import combinations
def SLRGP_ave_reductions2(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses the next evaluation point based on the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    

    q = U.shape[0] # No. Unevaluated pts
    
    # # All points -> Evaluated and Unevaluated points
    X = np.vstack((Z,U))
    
    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
        
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += T1 -  T2
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
            
    return np.array(ave_reductions).squeeze()

def SLRGP_Batch_Clustering2(Z,U, model,k, n_pts = 5,Y_vals=None,n_fac = 2):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    """   
    n_pts = 5
    n_fac = 2
    chosen_pts, chosen_indices = [],[]
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k) 
    
    ind = np.argpartition(ave_reductions, -n_pts*n_fac)[-n_pts*n_fac:]
    clusters = KMeans(n_clusters = n_pts)
    clusters.fit(U[ind])
    # clusters.labels_
    
    for cluster in range(n_pts):
        for i in np.where(clusters.labels_ == cluster):
            #print("cluster min = ",np.argmax(ave_reductions[ind][i]))
            print("Index = ", i[np.argmax(ave_reductions[ind][i])])
            chosen_indices.append(i[np.argmax(ave_reductions[ind][i])])
            chosen_pts.append(U[i[np.argmax(ave_reductions[ind][i])]])
            
    return chosen_pts, chosen_indices
	
def SLRGP_Batch_Clustering3(Z,U, model,k, n_pts = 5,Y_vals=None,n_fac = 3):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    """   
    n_pts = 5
	#n_fac = 3
    chosen_pts, chosen_indices = [],[]
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k) 
    
    ind = np.argpartition(ave_reductions, -n_pts*n_fac)[-n_pts*n_fac:]
    clusters = KMeans(n_clusters = n_pts)
    clusters.fit(U[ind])
    # clusters.labels_
    
    for cluster in range(n_pts):
        for i in np.where(clusters.labels_ == cluster):
            #print("cluster min = ",np.argmax(ave_reductions[ind][i]))
            print("Index = ", i[np.argmax(ave_reductions[ind][i])])
            chosen_indices.append(i[np.argmax(ave_reductions[ind][i])])
            chosen_pts.append(U[i[np.argmax(ave_reductions[ind][i])]])
            
    return chosen_pts, chosen_indices
	
def SLRGP_Batch_Clustering4(Z,U, model,k, n_pts = 5,Y_vals=None,n_fac = 4):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    """   
    n_pts = 5
	#n_fac = 4
    chosen_pts, chosen_indices = [],[]
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k) 
    
    ind = np.argpartition(ave_reductions, -n_pts*n_fac)[-n_pts*n_fac:]
    clusters = KMeans(n_clusters = n_pts)
    clusters.fit(U[ind])
    # clusters.labels_
    
    for cluster in range(n_pts):
        for i in np.where(clusters.labels_ == cluster):
            #print("cluster min = ",np.argmax(ave_reductions[ind][i]))
            print("Index = ", i[np.argmax(ave_reductions[ind][i])])
            chosen_indices.append(i[np.argmax(ave_reductions[ind][i])])
            chosen_pts.append(U[i[np.argmax(ave_reductions[ind][i])]])
            
    return chosen_pts, chosen_indices
	
def SLRGP_Batch_Clustering5(Z,U, model,k, n_pts = 5,Y_vals=None,n_fac = 5):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    """   
    n_pts = 5
	#n_fac = 5
    chosen_pts, chosen_indices = [],[]
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k) 
    
    ind = np.argpartition(ave_reductions, -n_pts*n_fac)[-n_pts*n_fac:]
    clusters = KMeans(n_clusters = n_pts)
    clusters.fit(U[ind])
    # clusters.labels_
    
    for cluster in range(n_pts):
        for i in np.where(clusters.labels_ == cluster):
            #print("cluster min = ",np.argmax(ave_reductions[ind][i]))
            print("Index = ", i[np.argmax(ave_reductions[ind][i])])
            chosen_indices.append(i[np.argmax(ave_reductions[ind][i])])
            chosen_pts.append(U[i[np.argmax(ave_reductions[ind][i])]])
            
    return chosen_pts, chosen_indices
	
def SLRGP_Batch_Clustering6(Z,U, model,k, n_pts = 5,Y_vals=None,n_fac = 6):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    """   
    n_pts = 5
	#n_fac = 6
    chosen_pts, chosen_indices = [],[]
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k) 
    
    ind = np.argpartition(ave_reductions, -n_pts*n_fac)[-n_pts*n_fac:]
    clusters = KMeans(n_clusters = n_pts)
    clusters.fit(U[ind])
    # clusters.labels_
    
    for cluster in range(n_pts):
        for i in np.where(clusters.labels_ == cluster):
            #print("cluster min = ",np.argmax(ave_reductions[ind][i]))
            print("Index = ", i[np.argmax(ave_reductions[ind][i])])
            chosen_indices.append(i[np.argmax(ave_reductions[ind][i])])
            chosen_pts.append(U[i[np.argmax(ave_reductions[ind][i])]])
            
    return chosen_pts, chosen_indices

def SLRGP_Batch_Group1_2(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 2):
    uneval_pts = U
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    partitioned = np.argpartition(ave_reductions, -n_pts*fac)
    best_indices = partitioned[-n_pts*fac:][::-1]
    

    SX = []
    comb = combinations(range(n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        # "i" maps to ranking in ave_reduction, then best_indices maps this to uneval_Pts
        indices = best_indices[list(i)]
        pts = uneval_pts[indices] #   
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
        SX.append(S_x)
    comb = combinations(range(n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = uneval_pts[indices]
            #print(eval_pts)
            
    chosen_pts = eval_pts
    chosen_indices = indices.T
    
    return chosen_pts, chosen_indices
	
def SLRGP_Batch_Group1_3(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 3):
    uneval_pts = U
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    

    SX = []
    comb = combinations(range(n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        # "i" maps to ranking in ave_reduction, then best_indices maps this to uneval_Pts
        indices = best_indices[list(i)]
        pts = uneval_pts[indices] #   
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
        SX.append(S_x)
    comb = combinations(range(n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = uneval_pts[indices]
            #print(eval_pts)
            
    chosen_pts = eval_pts
    chosen_indices = indices.T
    return chosen_pts, chosen_indices
	
def SLRGP_Batch_Group1_4(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 4):
    uneval_pts = U
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    

    SX = []
    comb = combinations(range(n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        # "i" maps to ranking in ave_reduction, then best_indices maps this to uneval_Pts
        indices = best_indices[list(i)]
        pts = uneval_pts[indices] #   
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
        SX.append(S_x)
    comb = combinations(range(n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = uneval_pts[indices]
            #print(eval_pts)
            
    chosen_pts = eval_pts
    chosen_indices = indices.T
    return chosen_pts, chosen_indices
	
def SLRGP_Batch_Group1_5(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 5):
    uneval_pts = U
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    

    SX = []
    comb = combinations(range(n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        # "i" maps to ranking in ave_reduction, then best_indices maps this to uneval_Pts
        indices = best_indices[list(i)]
        pts = uneval_pts[indices] #   
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
        SX.append(S_x)
    comb = combinations(range(n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = uneval_pts[indices]
            #print(eval_pts)
            
    chosen_pts = eval_pts
    chosen_indices = indices.T
    return chosen_pts, chosen_indices
	
def SLRGP_Batch_Group1_6(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 6):
    uneval_pts = U
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    

    SX = []
    comb = combinations(range(n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        # "i" maps to ranking in ave_reduction, then best_indices maps this to uneval_Pts
        indices = best_indices[list(i)]
        pts = uneval_pts[indices] #   
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
        SX.append(S_x)
    comb = combinations(range(n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = uneval_pts[indices]
            #print(eval_pts)
            
    chosen_pts = eval_pts
    chosen_indices = indices.T
    return chosen_pts, chosen_indices


def SLRGP_Batch_Group0_2(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 2):
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    
    n_pts -= 1
    SX = []
    comb = combinations(range(1,n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        indices = best_indices[list(i)]
        pts = U[indices] # 
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
            S_x += np.linalg.norm(pts[j] - U[best_indices[0]])
        SX.append(S_x)
    comb = combinations(range(1,n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = U[indices]
            print(eval_pts)
            
    chosen_pts = np.vstack((U[np.argmax(ave_reductions)][:,None].T,eval_pts))
    chosen_indices = np.hstack((np.array(best_indices[0])[None,],indices.T))
    
    return chosen_pts, chosen_indices

def SLRGP_Batch_Group0_3(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 3):
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    
    n_pts -= 1
    SX = []
    comb = combinations(range(1,n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        indices = best_indices[list(i)]
        pts = U[indices] # 
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
            S_x += np.linalg.norm(pts[j] - U[best_indices[0]])
        SX.append(S_x)
    comb = combinations(range(1,n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = U[indices]
            print(eval_pts)
            
    chosen_pts = np.vstack((U[np.argmax(ave_reductions)][:,None].T,eval_pts))
    chosen_indices = np.hstack((np.array(best_indices[0])[None,],indices.T))
    
    return chosen_pts, chosen_indices

def SLRGP_Batch_Group0_4(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 4):
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    
    n_pts -= 1
    SX = []
    comb = combinations(range(1,n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        indices = best_indices[list(i)]
        pts = U[indices] # 
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
            S_x += np.linalg.norm(pts[j] - U[best_indices[0]])
        SX.append(S_x)
    comb = combinations(range(1,n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = U[indices]
            print(eval_pts)
            
    chosen_pts = np.vstack((U[np.argmax(ave_reductions)][:,None].T,eval_pts))
    chosen_indices = np.hstack((np.array(best_indices[0])[None,],indices.T))
    
    return chosen_pts, chosen_indices

def SLRGP_Batch_Group0_5(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 5):
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    
    n_pts -= 1
    SX = []
    comb = combinations(range(1,n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        indices = best_indices[list(i)]
        pts = U[indices] # 
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
            S_x += np.linalg.norm(pts[j] - U[best_indices[0]])
        SX.append(S_x)
    comb = combinations(range(1,n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = U[indices]
            print(eval_pts)
            
    chosen_pts = np.vstack((U[np.argmax(ave_reductions)][:,None].T,eval_pts))
    chosen_indices = np.hstack((np.array(best_indices[0])[None,],indices.T))
    
    return chosen_pts, chosen_indices

def SLRGP_Batch_Group0_6(Z,U,model,k,n_pts = 5,Y_vals=None,fac = 6):
    
    ave_reductions = SLRGP_ave_reductions2(Z,U,model,k)
    
    # Order reversed so starts with Best pt
    best_indices = np.argpartition(ave_reductions, -n_pts*fac)[-n_pts*fac:][::-1]
    
    n_pts -= 1
    SX = []
    comb = combinations(range(1,n_pts*fac), n_pts)
    # Print the obtained combinations
    for i in list(comb):
        indices = best_indices[list(i)]
        pts = U[indices] # 
        S_x = 0
        for j in range(n_pts):
            for jj in range(n_pts):
                    S_x += np.linalg.norm(pts[j] - pts[jj])
            S_x += np.linalg.norm(pts[j] - U[best_indices[0]])
        SX.append(S_x)
    comb = combinations(range(1,n_pts*fac), n_pts)
    for _,i in enumerate(list(comb)):
        if _ == np.argmax(np.array(SX)):
            indices = best_indices[list(i)]
            eval_pts = U[indices]
            print(eval_pts)
            
    chosen_pts = np.vstack((U[np.argmax(ave_reductions)][:,None].T,eval_pts))
    chosen_indices = np.hstack((np.array(best_indices[0])[None,],indices.T))
    
    return chosen_pts, chosen_indices

def SLRGP_z_variances(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses the next evaluation point based on the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    

    q = U.shape[0] # No. Unevaluated pts
    
    # # All points -> Evaluated and Unevaluated points
    X = np.vstack((Z,U))
    
    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    #Lambda = optimize_SLRGP_parameter(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
      
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    current_vars = []
    potential_vars = []
    
    for i in range(q):
        reduction = 0
        c_var = 0
        p_var = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += T1 -  T2
                c_var += T1
                p_var += T2
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        current_vars.append(c_var)
        potential_vars.append(p_var)
        
       #np.array(ave_reductions).squeeze(),      
    return np.array(current_vars).squeeze(), np.array(potential_vars).squeeze()

def SLRGP_ave_reductions(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses the next evaluation point based on the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    

    q = U.shape[0] # No. Unevaluated pts
    
    # # All points -> Evaluated and Unevaluated points
    X = np.vstack((Z,U))
    
    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
        
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    Lambda_1 = Lambda
    Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 
    
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        #Lambda_2 = Lambda*k.rbf.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += T1 -  T2
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
            
    return np.array(ave_reductions).squeeze()


def calculate_laplacian_parts(x_new,Z,model):
    """This is a helper function for SLRGP which calculates the Laplacian as presented in the original SLRGP paper. 
    
    TODO: EDIT AND UPDATE
    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained

    :return: (numpy array), the Laplacian
    """ 
    
    # All points -> Evaluated and Unevaluated points
    X = np.vstack((x_new,Z))
    Y, _ = model.predict(X)
    
    # Calculate Laplacian Matrix
    S_x = np.empty((Z.shape[0]))
    S_y = np.empty((Z.shape[0]))

    for j in range(Z.shape[0]):
        S_x[j] = np.linalg.norm(x_new - Z[j])
        S_y[j] = np.linalg.norm(Y[0] - Y[j])
    D_x =  S_x.sum()
    D_y =  S_y.sum()
    
    return D_x, D_y,S_x,S_y



def SLRGP_modified_penalty2_4B(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - np.exp(-D_y/D_x)
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-D_y/D_x) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]


def SLRGP_modified_penalty2_5(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - np.exp(-D_y/D_y)
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-D_y) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]


def SLRGP_modified_penalty2_6(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - np.exp(-D_y/D_y)
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-D_y)*np.exp(-D_x) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]


def SLRGP_modified_penalty2_7(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - np.exp(-D_y/D_y)
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-np.min(S_x)) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]


def SLRGP_modified_penalty2_8(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - np.exp(-D_y/D_y)
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-D_x/np.min(S_y)) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]


def SLRGP_modified_penalty2_9(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - np.exp(-D_y/D_y)
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-np.min(S_y)/D_x ) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]

def SLRGP_modified_penalty2_1(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - exp(- dx)
        
        where dx is the sum of distances from the proposed point x and all evaluated points"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(- D_x) ) 
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]


def SLRGP_modified_penalty2_2(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - exp(-np.min(S_x)) 
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-np.min(S_x)) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]


def SLRGP_modified_penalty2_3(Z,U, model,k, n_pts = None,Y_vals=None,n_fac = None):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - np.exp(-D_x/D_y) ) 
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-D_x/D_y) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]


def SLRGP_modified_penalty2_4(Z,U, model,k):
    """Modification: 
        
        Average reduction in variance (eq's 5, 6 in SLRGP paper) is scaled by 
        (1 - np.exp(-D_y/D_y)
        
        where dx is the sum of distances from the proposed point x and all evaluated points
        Sx is the collection of all distances from the proposed point to all evaluatd pts"""   
    

    q = U.shape[0] 

    X = np.vstack((Z,U))


    #print("calculating L...")
    L = calculate_laplacian(Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]   
                D_x, D_y,S_x,S_y = calculate_laplacian_parts(xj, Z, model)
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += (T1 -  T2)*(1 - np.exp(-D_y/D_y) )
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]

def ALM(eval_pts, uneval_pts,model,k ):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel. It then selects the point with highest predictive uncertainty.


    :param eval_pts: (numpy array), The set of evaluated points
    :param uneval_pts: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """
    
    post_means, post_vars = model.predict(uneval_pts)
    
    # Point with highest variance
    new_point = uneval_pts[np.argmax(post_vars)]
    indx = np.argmax(post_vars)
    return [new_point], [indx]

def ALC_old(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel. It then selects the point that gives largest average reduction in predictive uncertainty


    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """    

    #k = model.kern
    m = Z.shape[0] # No. evaluated pts
    n = U.shape[0] # No. Unevaluated pts
    
    ave_reductions = []
    for i in range(n):
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi))) # K_{ z + x1, z + x1}
        K_z_xi = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) ) 
    
        reduction = 0
        for j in range(m):
            xj = Z[j][None,:]
            reduction +=  ( k.K(xj,z_xi)* K_z_xi*k.K(z_xi,xj) ).sum()
    
        ave_reductions.append( reduction/(n-1) ) 
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    return new_point, indx

def ALC(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel. It then selects the point that gives largest average reduction in predictive uncertainty


    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    
    #k = model.kern
    #m = Z.shape[0] # No. evaluated pts
    q = U.shape[0] # No. Unevaluated pts
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    try:
        K_zz_inv = linalg.inv( k.K(Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 
        
    except:
        # Add jitter
        K_zz_inv = linalg.inv( k.K(Z) + k.variance*np.eye(k.K(Z).shape[0]) + np.eye(k.K(Z).shape[0])*1e-04  ) 
    
    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        try:
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )

        
        except:
            # Add jitter
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04  )
        
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += T1 -  T2
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
        
        # xi = U[i][None,:]
        # z_xi = k.K(np.vstack((Z,xi))) # K_{ z + x1, z + x1}
        # K_z_xi = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) ) 
    
        # reduction = 0
        # for j in range(m):
        #     xj = Z[j][None,:]
        #     reduction +=  ( k.K(xj,z_xi)* K_z_xi*k.K(z_xi,xj) ).sum()
    
        # ave_reductions.append( reduction/(q-1) ) 
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    return new_point, indx
    
def SLRGP_modified_penalty(Z,U, model,k,n_pts = None,Y_vals=None,fac = None):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses the next evaluation point based on the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    

    q = U.shape[0] # No. Unevaluated pts
    
    # # All points -> Evaluated and Unevaluated points
    X = np.vstack((Z,U))


    
    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_modified_penalty(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    print("optimised  Lambda...")
    #print("Shape of Z = ",Z.shape )
    #print("Lambda = ",Lambda )
    #print("k.rbf.variance = ",k.rbf.variance )
    #print("Shape of k.K(Z,Z) = ",k.K(Z).shape )
    #print("linalg.inv( k.K(Z,Z) ) with jitter = ",linalg.inv( k.K(Z,Z) + np.eye(k.K(Z,Z).shape[0])*1e-04  ) )
    
    pred_X, _ = model.predict(X)
    pred_U, _ = model.predict(U)
    pred_Z, _ = model.predict(Z)
    
    kerY = GPy.kern.RBF(1, variance=k.variance, lengthscale=np.mean(k.lengthscale))
    
    # Elementwise prediction of 
    new_KZZ = k.K(Z)*kerY.K(pred_Z)
    new_KXZ = k.K(X,Z)*kerY.K(pred_X,pred_Z)
    new_KZX = k.K(Z,X)*kerY.K(pred_Z,pred_X)
    #new_KZxi
    
    
    #Lambda_1 = Lambda*k.rbf.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    Lambda_1 = Lambda*k.variance*linalg.inv( new_KZZ + np.eye(new_KZZ.shape[0])*1e-04  )

    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( new_KZZ + Lambda_1@new_KZX@L@new_KXZ + k.variance*np.eye(new_KZZ.shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        pred_z_xi, _ = model.predict(z_xi)
        new_KZxi = k.K(z_xi)*kerY.K(pred_z_xi)
        
    
        pred_z_xi, _ = model.predict(z_xi)
        new_kZxiX = k.K(z_xi,X)*kerY.K(pred_z_xi, pred_X)
        new_kXZxi = k.K(X,z_xi)*kerY.K(pred_X,pred_z_xi)
        
        Lambda_2 = Lambda*k.variance*linalg.inv( new_KZxi + np.eye(new_KZxi.shape[0])*1e-04 )
        
        #Lambda_2 = Lambda*k.rbf.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )

        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )

        K_z_xi_inv = linalg.inv( new_KZxi + Lambda_2@new_kZxiX@L@new_kXZxi + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]
                
                pred_xj, _ = model.predict(xj)
                new_KxjZ = k.K(xj,Z)*kerY.K(pred_xj, pred_Z)
                new_KZxj = k.K(Z,xj)*kerY.K(pred_Z,pred_xj)
                new_xj_Zxi = k.K(xj,z_xi)*kerY.K(pred_xj,pred_z_xi)
                new_KZxi_xj = k.K(z_xi,xj)*kerY.K(pred_z_xi,pred_xj)
                
                T1 = k.K(xj)*kerY.K(pred_xj) - new_KxjZ @ K_zz_inv @ new_KZxj
                
                T2 = k.K(xj)*kerY.K(pred_xj) - new_xj_Zxi @ K_z_xi_inv @ new_KZxi_xj
                
                
                reduction += T1 -  T2
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        

        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return [new_point], [indx]

def SLRGP(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses the next evaluation point based on the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    

    q = U.shape[0] # No. Unevaluated pts
    
    # # All points -> Evaluated and Unevaluated points
    X = np.vstack((Z,U))


    
    #print("calculating L...")
    L = calculate_laplacian(X,Z,U,model)
    #L = calculate_alternative_laplacian(Z,U,model)
    #print("optimising  Lambda...")
    #Lambda = optimize_SLRGP_parameter(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    
    #print("Shape of Z = ",Z.shape )
    #print("Lambda = ",Lambda )
    #print("k.rbf.variance = ",k.rbf.variance )
    #print("Shape of k.K(Z,Z) = ",k.K(Z).shape )
    #print("linalg.inv( k.K(Z,Z) ) with jitter = ",linalg.inv( k.K(Z,Z) + np.eye(k.K(Z,Z).shape[0])*1e-04  ) )
    
    
    Lambda_1 = Lambda*k.variance*linalg.inv( k.K(Z) + np.eye(k.K(Z).shape[0])*1e-04  )
    
    #Lambda_1 = Lambda
    #Lambda_2 = Lambda
    
    #print("Shape of Lambda 1:", Lambda_1.shape)
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + Lambda_1*k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 
    
    K_zz_inv = linalg.inv( k.K(Z) + Lambda_1@k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 

    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        Lambda_2 = Lambda*k.variance*linalg.inv( k.K(z_xi) + np.eye(k.K(z_xi).shape[0])*1e-04 )
        
        #print("Shape of Lambda 2:", Lambda_2.shape)
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda_2@k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += T1 -  T2
                #print(reduction)
        
        ave_reductions.append( reduction/(q - 1 ) )
        
        
        # xi = U[i][None,:]
        # z_xi = k.K(np.vstack((Z,xi))) # K_{ z + x1, z + x1}
        # K_z_xi = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) ) 
    
        # reduction = 0
        # for j in range(m):
        #     xj = Z[j][None,:]
        #     reduction +=  ( k.K(xj,z_xi)* K_z_xi*k.K(z_xi,xj) ).sum()
    
        # ave_reductions.append( reduction/(q-1) ) 
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    if len([indx]) > 1:
        indx = indx[0]
        new_point = U[indx]
    
    #print("SLRGP INDX", indx)
    #print("SLRGP new point", new_point)
    
    return new_point, indx

def SLRGP_FIXED(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses the next evaluation point based on the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    

    q = U.shape[0] # No. Unevaluated pts
    
    # # All points -> Evaluated and Unevaluated points
    X = np.vstack((Z,U))
    
    #print("calculating L...")
    L = calculate_laplacian(Z,U,model)

    #L = calculate_laplacian_FIXED(Z,Y,model)
    #print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    

    ave_reductions = []
    K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    for i in range(q):
        reduction = 0
        
        # Term 1
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
    
        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += T1 -  T2
        
        ave_reductions.append( reduction/(q - 1 ) )
        
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    return [new_point], [indx]
    


def optimize_SLRGP_parameter_modified_penalty(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2]):
    """This is a helper function for SLRGP for optimization of the Tuning Parameter of the Laplacian Regularization Penalty. 
    This is presented as algorithm 2 in the original SLRGP paper. 
    

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param X: (numpy array), The set of all points in the domain - both evaluated and unevaluated points
    :param Lambda_vals: (list), List of pre-determined values to choose for lambda
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (float), the optimised value of Lambda
    """  
    
    P = np.empty((len(Lambda_vals)))
    
    q = U.shape[0] # No. Unevaluated pts
    
    x_star = []
    
    kerY = GPy.kern.RBF(1, variance=k.variance, lengthscale=np.mean(k.lengthscale))
    
    pred_X, _ = model.predict(X)
 
    
    for indx,Lambda in enumerate(Lambda_vals):
        reduction = 0
        
        L = calculate_laplacian(X,Z,U,model)
        #L = calculate_alternative_laplacian(Z,U,model)
        
        m_i = np.empty((U.shape[0]))
        
        for i in range(U.shape[0]):
            xi = U[i]
            U_nxi = np.delete(U, obj=i, axis=0) # remove xi from U
            z_xi = np.vstack((Z,xi)) # add xi to Z
            
            pred_z_xi, _ = model.predict(z_xi)
            new_KZxi = k.K(z_xi)*kerY.K(pred_z_xi)
        
    
            pred_z_xi, _ = model.predict(z_xi)
            new_kZxiX = k.K(z_xi,X)*kerY.K(pred_z_xi, pred_X)
            new_kXZxi = k.K(X,z_xi)*kerY.K(pred_X,pred_z_xi)
                        
             # np.argwhere(numpy.isnan( k.K(z_xi) ))
            # print("k.K(z_xi):", np.max( k.K(z_xi) ))
            # print("Lambda:",(np.max( Lambda)) )
            # print("k.K(X,z_xi):", (np.max( k.K(X,z_xi)))  )
            # print("k.K(z_xi,X):", (np.max( k.K(z_xi,X))) )
            # print("L:", np.argwhere(np.isnan( L )))
            # print("L:", (np.max( L )))
            # print("k.K(z_xi,X)@L :", k.K(z_xi,X)@L)
            #print("k.rbf.variance*np.eye(k.K(z_xi).shape[0]):", k.rbf.variance*np.eye(k.K(z_xi).shape[0]))
            #K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.rbf.variance*np.eye(k.K(z_xi).shape[0]) )
            try:
                #K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.rbf.variance*np.eye(k.K(z_xi).shape[0]) )
                K_X_Zxi_inv = linalg.inv( new_KZxi + Lambda*new_kZxiX@L@new_kXZxi + k.variance*np.eye(k.K(z_xi).shape[0]) )
 
            except:
                print("adding jitter")
                # Add jitter
                #K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.rbf.variance*np.eye(k.K(z_xi).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04  )
                K_X_Zxi_inv = linalg.inv( new_KZxi + Lambda*new_kZxiX@L@new_kXZxi + k.variance*np.eye(k.K(z_xi).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04  )

            pred_KU_nxi, _ = model.predict(U_nxi)
            new_KU_nxi = k.K(U_nxi)*kerY.K(pred_KU_nxi)
            new_U_nxi_z_xi = k.K(U_nxi,z_xi)*kerY.K(pred_KU_nxi,pred_z_xi)
            new_z_xi_U_nxi = k.K(z_xi,U_nxi)*kerY.K(pred_z_xi,pred_KU_nxi)
            
            #print("going to get the max of", k.K(U_nxi,U_nxi) - k.K(U_nxi,z_xi)@K_X_Zxi_inv@k.K(z_xi,U_nxi))
            #m_i[i] = np.max( k.K(U_nxi,U_nxi) - k.K(U_nxi,z_xi)@K_X_Zxi_inv@k.K(z_xi,U_nxi) )
            m_i[i] = np.max( new_KU_nxi - new_U_nxi_z_xi@K_X_Zxi_inv@new_z_xi_U_nxi )
     
        x_star.append( np.argmax(m_i) )
        
        xi = U[np.argmax(m_i)][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        pred_z_xi, _ = model.predict(z_xi)
        new_KZxi = k.K(z_xi)*kerY.K(pred_z_xi)
        
    
        pred_z_xi, _ = model.predict(z_xi)
        new_kZxiX = k.K(z_xi,X)*kerY.K(pred_z_xi, pred_X)
        new_kXZxi = k.K(X,z_xi)*kerY.K(pred_X,pred_z_xi)
        
        pred_U, _ = model.predict(U)
        pred_Z, _ = model.predict(Z)
        
        #kerY = GPy.kern.RBF(1, variance=k.rbf.variance, lengthscale=np.mean(k.rbf.lengthscale))
        
        # Elementwise prediction of 
        new_KZZ = k.K(Z)*kerY.K(pred_Z)
        new_KXZ = k.K(X,Z)*kerY.K(pred_X,pred_Z)
        new_KZX = k.K(Z,X)*kerY.K(pred_Z,pred_X)
        
        try:
            #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
            #K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
            K_z_xi_inv = linalg.inv( new_KZxi + Lambda*new_kZxiX@L@new_kXZxi + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
            K_zz_inv = linalg.inv( new_KZZ + Lambda*new_KZX@L@new_KXZ + k.variance*np.eye(k.K(Z).shape[0]) ) 
        
        except:
            #print("adding jitter")
            #K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04 )
            #K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) + np.eye(k.K(Z).shape[0])*1e-04 ) 
            print("adding jitter")
            K_z_xi_inv = linalg.inv( new_KZxi + Lambda*new_kZxiX@L@new_kXZxi + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04 )
            K_zz_inv = linalg.inv( new_KZZ + Lambda*new_KZX@L@new_KXZ + k.variance*np.eye(k.K(Z).shape[0]) + np.eye(k.K(Z).shape[0])*1e-04 ) 

        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                #T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                #T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                pred_xj, _ = model.predict(xj)
                new_KxjZ = k.K(xj,Z)*kerY.K(pred_xj, pred_Z)
                new_KZxj = k.K(Z,xj)*kerY.K(pred_Z,pred_xj)
                new_xj_Zxi = k.K(xj,z_xi)*kerY.K(pred_xj,pred_z_xi)
                new_KZxi_xj = k.K(z_xi,xj)*kerY.K(pred_z_xi,pred_xj)
                
                T1 = k.K(xj)*kerY.K(pred_xj) - new_KxjZ @ K_zz_inv @ new_KZxj
                
                T2 = k.K(xj)*kerY.K(pred_xj) - new_xj_Zxi @ K_z_xi_inv @ new_KZxi_xj
                
                
                reduction += T1 -  T2
                #print(reduction)
        
        P[indx] = reduction/(q - 1 )
    
    return np.argmax(P)
    
    
def optimize_SLRGP_parameter(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2]):
    """This is a helper function for SLRGP for optimization of the Tuning Parameter of the Laplacian Regularization Penalty. 
    This is presented as algorithm 2 in the original SLRGP paper. 
    

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param X: (numpy array), The set of all points in the domain - both evaluated and unevaluated points
    :param Lambda_vals: (list), List of pre-determined values to choose for lambda
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (float), the optimised value of Lambda
    """  
    #print(Z)
    #print(U)
    #print(model)
    #print(k)
    
    P = np.empty((len(Lambda_vals)))
    
    q = U.shape[0] # No. Unevaluated pts
    
    x_star = []
    
    L = calculate_laplacian(X,Z,U,model)
    
    for indx,Lambda in enumerate(Lambda_vals):
        reduction = 0
        
        #L = calculate_laplacian(Z,U,model)
        #L = calculate_alternative_laplacian(Z,U,model)
        
        m_i = np.empty((U.shape[0]))
        
        for i in range(U.shape[0]):
            xi = U[i]
            U_nxi = np.delete(U, obj=i, axis=0) # remove xi from U
            z_xi = np.vstack((Z,xi)) # add xi to Z
             # np.argwhere(numpy.isnan( k.K(z_xi) ))
            # print("k.K(z_xi):", np.max( k.K(z_xi) ))
            # print("Lambda:",(np.max( Lambda)) )
            # print("k.K(X,z_xi):", (np.max( k.K(X,z_xi)))  )
            # print("k.K(z_xi,X):", (np.max( k.K(z_xi,X))) )
            # print("L:", np.argwhere(np.isnan( L )))
            # print("L:", (np.max( L )))
            # print("k.K(z_xi,X)@L :", k.K(z_xi,X)@L)
            #print("k.rbf.variance*np.eye(k.K(z_xi).shape[0]):", k.rbf.variance*np.eye(k.K(z_xi).shape[0]))
            #K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.rbf.variance*np.eye(k.K(z_xi).shape[0]) )
            try:
                K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.variance*np.eye(k.K(z_xi).shape[0]) )
            except:
                print("adding jitter")
                # Add jitter
                K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.variance*np.eye(k.K(z_xi).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04  )

            
            #print("going to get the max of", k.K(U_nxi,U_nxi) - k.K(U_nxi,z_xi)@K_X_Zxi_inv@k.K(z_xi,U_nxi))
            m_i[i] = np.max( k.K(U_nxi,U_nxi) - k.K(U_nxi,z_xi)@K_X_Zxi_inv@k.K(z_xi,U_nxi) )
     
        x_star.append( np.argmax(m_i) )
        
        xi = U[np.argmax(m_i)][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        try:
            K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
            K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 
        
        except:
            print("adding jitter")
            K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04 )
            K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) + np.eye(k.K(Z).shape[0])*1e-04 ) 

        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += T1 -  T2
                #print(reduction)
        
        P[indx] = reduction/(q - 1 )
    
    return np.argmax(P)

def optimize_SLRGP_parameter_FIXED(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2]):
    """This is a helper function for SLRGP for optimization of the Tuning Parameter of the Laplacian Regularization Penalty. 
    This is presented as algorithm 2 in the original SLRGP paper. 
    

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param X: (numpy array), The set of all points in the domain - both evaluated and unevaluated points
    :param Lambda_vals: (list), List of pre-determined values to choose for lambda
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (float), the optimised value of Lambda
    """  
    
    P = np.empty((len(Lambda_vals)))
    
    q = U.shape[0] # No. Unevaluated pts
    
    x_star = []
    
    L = calculate_laplacian(X,Z,U,model)
    
    for indx,Lambda in enumerate(Lambda_vals):
        reduction = 0
        
        #L = calculate_laplacian(Z,U,model)
        #L = calculate_laplacian_FIXED(Z,Y,model)
        
        m_i = np.empty((U.shape[0]))
        
        for i in range(U.shape[0]):
            xi = U[i]
            U_nxi = xi[:,None].T #np.delete(U, obj=i, axis=0) # remove xi from U
            z_xi = Z #np.vstack((Z,xi)) # add xi to Z
             # np.argwhere(numpy.isnan( k.K(z_xi) ))
            # print("k.K(z_xi):", np.max( k.K(z_xi) ))
            # print("Lambda:",(np.max( Lambda)) )
            # print("k.K(X,z_xi):", (np.max( k.K(X,z_xi)))  )
            # print("k.K(z_xi,X):", (np.max( k.K(z_xi,X))) )
            # print("L:", np.argwhere(np.isnan( L )))
            # print("L:", (np.max( L )))
            # print("k.K(z_xi,X)@L :", k.K(z_xi,X)@L)
            #print("k.rbf.variance*np.eye(k.K(z_xi).shape[0]):", k.rbf.variance*np.eye(k.K(z_xi).shape[0]))
            #K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.rbf.variance*np.eye(k.K(z_xi).shape[0]) )
            try:
                K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.variance*np.eye(k.K(z_xi).shape[0]) )
            except:
                print("adding jitter")
                # Add jitter
                K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.variance*np.eye(k.K(z_xi).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04  )

            
            #print("going to get the max of", k.K(U_nxi,U_nxi) - k.K(U_nxi,z_xi)@K_X_Zxi_inv@k.K(z_xi,U_nxi))
            m_i[i] = k.K(U_nxi) - k.K(U_nxi,z_xi)@K_X_Zxi_inv@k.K(z_xi,U_nxi)
     
        x_star.append( np.argmax(m_i) )
        
        xi = U[np.argmax(m_i)][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        try:
            K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
            K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) ) 
        
        except:
            print("adding jitter")
            K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04 )
            K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.variance*np.eye(k.K(Z).shape[0]) + np.eye(k.K(Z).shape[0])*1e-04 ) 

        for j in range(q - 1 ):
            if j != i:
                
                xj = U[j][None,:]               
                T1 = k.K(xj) - k.K(xj,Z) @ K_zz_inv @ k.K(Z,xj)
                
                T2 = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)
                
                
                reduction += T1 -  T2
                #print(reduction)
        
        P[indx] = reduction/(q - 1 )
    
    return np.argmax(P)
        
        #for l in Lambda_vals:
        
    
def calculate_laplacian(X,Z,U,model):
    """This is a helper function for SLRGP which calculates the Laplacian as presented in the original SLRGP paper. 
    

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained

    :return: (numpy array), the Laplacian
    """ 
    #if not Lambda:
    #    Lambda = 1
        
    #k = model.kern
    #m = Z.shape[0] # No. evaluated pts
    #q = U.shape[0] # No. Unevaluated pts
    
    # All points -> Evaluated and Unevaluated points
    #X = np.vstack((Z,U))
    Y, _ = model.predict(X)
    
    # Calculate Laplacian Matrix
    S_x = np.empty((X.shape[0], X.shape[0]))
    S_y = np.empty((X.shape[0], X.shape[0]))
    D_x = np.eye(X.shape[0])
    D_y = np.eye(X.shape[0])
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            S_x[i,j] = np.linalg.norm(X[i] - X[j])
            S_y[i,j] = np.linalg.norm(Y[i] - Y[j])
        D_x[i,i] =  S_x[i,:].sum()
        #if D_x[i,i] == 0:
        #    D_x[i,i] = 0.001 # To avoid division by zero
        D_y[i,i] =  S_y[i,:].sum()
    

    
    L = ( D_y - S_y)/(D_x - S_x)
    
    #print("Number of Nans", np.count_nonzero(np.isnan(L)) )
    #print("Average L", np.nanmean(L) )
    
    # Check there will be no division by zero:
    for i in range((D_x - S_x).shape[0]):
        for j in range((D_x - S_x).shape[0]):
            if (D_x - S_x)[i,j] == 0:
                L[i,j] = np.nanmean(L)
                
    # print("L:", L)
    # print("D_y:", D_y)
    # print("S_y:", S_y)
    # print("D_x:", D_x)
    # print("S_x:", S_x)
    
    return L

def calculate_laplacian_FIXED(X,Y,model):
    """This is a helper function for SLRGP which calculates the Laplacian as presented in the original SLRGP paper. 
    

    :param X: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained

    :return: (numpy array), the Laplacian
    """ 
    #if not Lambda:
    #    Lambda = 1
        
    #k = model.kern
    #m = Z.shape[0] # No. evaluated pts
    #q = U.shape[0] # No. Unevaluated pts
    
    # All points -> Evaluated and Unevaluated points
    #X = np.vstack((Z,U))
    ##Y, _ = model.predict(X)
    
    # Calculate Laplacian Matrix
    S_x = np.empty((X.shape[0], X.shape[0]))
    S_y = np.empty((X.shape[0], X.shape[0]))
    D_x = np.eye(X.shape[0])
    D_y = np.eye(X.shape[0])
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            S_x[i,j] = np.linalg.norm(X[i] - X[j])
            S_y[i,j] = np.linalg.norm(Y[i] - Y[j])
        D_x[i,i] =  S_x[i,:].sum()
        D_y[i,i] =  S_y[i,:].sum()
    
    L = ( D_y - S_y)/(D_x - S_x)
    # print("L:", L)
    # print("D_y:", D_y)
    # print("S_y:", S_y)
    # print("D_x:", D_x)
    # print("S_x:", S_x)
    
    return L


def calculate_laplacian_in_parts(Z,U,model):
    """This is edited from actual paper. Just for testing!!

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained

    :return: (numpy array), the Laplacian
    """ 
    #if not Lambda:
    #    Lambda = 1
        
    #k = model.kern
    #m = Z.shape[0] # No. evaluated pts
    #q = U.shape[0] # No. Unevaluated pts
    
    # All points -> Evaluated and Unevaluated points
    X = np.vstack((Z,U))
    Y, _ = model.predict(X)
    
    # Calculate Laplacian Matrix
    S_x = np.empty((X.shape[0], X.shape[0]))
    S_y = np.empty((X.shape[0], X.shape[0]))
    D_x = np.eye(X.shape[0])
    D_y = np.eye(X.shape[0])
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            S_x[i,j] = np.linalg.norm(X[i] - X[j])
            S_y[i,j] = np.linalg.norm(Y[i] - Y[j])
        D_x[i,i] =  S_x[i,:].sum()
        D_y[i,i] =  S_y[i,:].sum()
    
    Ly = ( D_y - S_y)
    Lx = (D_x - S_x)
    
    #L = S_y/S_x
    
    #A = ( D_y - S_y)/( D_y - S_y).max()
    #B = (D_x - S_x)/(D_x - S_x).max()
    #L = A/B
    
    # print("L:", L)
    # print("D_y:", D_y)
    # print("S_y:", S_y)
    # print("D_x:", D_x)
    # print("S_x:", S_x)
    
    return Lx,Ly
    

def SLRGP_Batch(Z,U, model,k, n_pts = 2,Y=None):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    chosen_pts, chosen_indices = [],[]
        
    for pt in range(n_pts):
        pt, index = SLRGP(Z,U, model,k)
        Z = np.vstack((Z,pt))
        #print("pt",np.array(pt))
        #print("shape", pt.shape)
        #print("shape", pt[:,None].shape)
        pt_mean, _ = model.predict(np.array(pt)[:,None].T)
        #print("predicted pt, got",pt_mean)
        Y = np.vstack((Y,pt_mean))
        
        # Remove the new point from the set of unevaluated points
        U = np.delete(U, index,axis = 0)
        
        chosen_pts.append(pt)
        chosen_indices.append(index)
        
        model = GPy.models.GPRegression(Z,Y,kernel = k)
        model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
    
        # optimise parameters
        model.optimize()
        model.optimize_restarts(num_restarts = 10, verbose=False)
        
    return chosen_pts, chosen_indices

from sklearn.cluster import KMeans
def SLRGP_Batch_Clustering(Z,U, model,k, n_fac = 2,Y=None):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    n_pts = 5
    chosen_pts, chosen_indices = [],[]
    
    ave_reductions = SLRGP_ave_reductions(Z,U,model,k) 
    
    ind = np.argpartition(ave_reductions, -n_pts*n_fac)[-n_pts*n_fac:]
    clusters = KMeans(n_clusters = n_pts)
    clusters.fit(U[ind])
    # clusters.labels_
    
    for cluster in range(n_pts):
        for i in np.where(clusters.labels_ == cluster):
            #print("cluster min = ",np.argmax(ave_reductions[ind][i]))
            print("Index = ", i[np.argmax(ave_reductions[ind][i])])
            chosen_indices.append(i[np.argmax(ave_reductions[ind][i])])
            chosen_pts.append(U[i[np.argmax(ave_reductions[ind][i])]])
    
        
    # for pt in range(n_pts):
    #     pt, index = SLRGP(Z,U, model,k)
    #     Z = np.vstack((Z,pt))
    #     #print("pt",np.array(pt))
    #     #print("shape", pt.shape)
    #     #print("shape", pt[:,None].shape)
    #     pt_mean, _ = model.predict(np.array(pt)[:,None].T)
    #     #print("predicted pt, got",pt_mean)
    #     Y = np.vstack((Y,pt_mean))
        
    #     # Remove the new point from the set of unevaluated points
    #     U = np.delete(U, index,axis = 0)
        
    #     chosen_pts.append(pt)
    #     chosen_indices.append(index)
        
    #     model = GPy.models.GPRegression(Z,Y,kernel = k)
    #     model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
    
    #     # optimise parameters
    #     model.optimize()
    #     model.optimize_restarts(num_restarts = 10, verbose=False)
        
    return chosen_pts, chosen_indices

def SLRGP_Batch3(Z,U, model,k, n_pts = 3):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    chosen_pts, chosen_indices = [],[]
        
    for pt in range(n_pts):
        pt, index = SLRGP(Z,U, model,k)
        Z = np.vstack((Z,pt))
        
        # Remove the new point from the set of unevaluated points
        U = np.delete(U, index,axis = 0)
        
        chosen_pts.append(pt)
        chosen_indices.append(index)
    
    return chosen_pts, chosen_indices

def SLRGP_Batch5(Z,U, model,k, n_pts = 5):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses a batch of future evaluation points found by recursively running the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    chosen_pts, chosen_indices = [],[]
        
    for pt in range(n_pts):
        pt, index = SLRGP(Z,U, model,k)
        Z = np.vstack((Z,pt))
        
        # Remove the new point from the set of unevaluated points
        U = np.delete(U, index,axis = 0)
        
        chosen_pts.append(pt)
        chosen_indices.append(index)
    
    return chosen_pts, chosen_indices

def IMSE(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel. It then selects the 
    point that gives largest average reduction in predictive uncertainty


    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """  
    
    #k = model.kern
    #m = Z.shape[0] # No. evaluated pts
    q = U.shape[0] # No. Unevaluated pts
    
    ave_reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    for i in range(q):
        reduction = 0
        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        
        try:
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.variance*np.eye(z_xi.shape[0] ) )
        except:
            # Add jitter
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.variance*np.eye(z_xi.shape[0] ) + np.eye(z_xi.shape[0])*1e-04 )

        
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(z_xi.shape[0] ) )
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0] ) )
    
        for j in range(q):
            if j != i:
                
                xj = U[j][None,:]               
                
                reduction += k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)               
                        
        ave_reductions.append( reduction )
                
    indx = np.argmax(ave_reductions)
    new_point = U[indx]
    
    return new_point, indx

def MMSE(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel. It then selects the 
    point that minimises the maximum predicted variance of the unevaluated points.


    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    
    #k = model.kern
    #m = Z.shape[0] # No. evaluated pts
    q = U.shape[0] # No. Unevaluated pts
    
    reductions = []
    #K_zz_inv = linalg.inv( k.K(Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
    for i in range(q):
        max_reduction = 0
        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        try:
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.variance*np.eye(z_xi.shape[0] ) )
        except:
            # Add jitter
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.variance*np.eye(z_xi.shape[0] ) + np.eye(z_xi.shape[0])*1e-04 )
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0] ) )
    
        for j in range(q):
            if j != i:
                
                xj = U[j][None,:]               
                
                new_reduction = k.K(xj) - k.K(xj,z_xi) @ K_z_xi_inv @ k.K(z_xi,xj)    
                
                max_reduction = max(max_reduction, new_reduction )
                        
        reductions.append( max_reduction )
                
    indx = np.argmin(reductions)
    new_point = U[indx]
    
    return new_point, indx
    
def SLRV_laplacian(X,knn):
    
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    S = np.zeros(((X.shape[0]),X.shape[0])) 
    for i,j in enumerate(indices):

        S[i,j] = 1

    # Set diagonal enteries to zero
    S = S - np.eye((X.shape[0]))
    
    D = np.eye((X.shape[0]))
        
    for i in range(X.shape[0]):
        D[i,i] = S[i,:].sum()
        
    L = D - S
    
    return L
    
def SLRV_get_W(Z,xi,m,h):
    # Possibly incorrect -> go through p vs m in shapes!
    W = np.zeros((m,m))
    for i in range(m):
        if np.linalg.norm((xi - Z[i])/h, ord=1) < 1:
            W[i,i] = (1 - np.linalg.norm((xi - Z[i])/h, ord=1)**3 )**3
            
    return W

def SLRV(Z,U, model,k = 5, h = 0.01, l1 = 1, l2 = 1):
    
    #X,Z,xj,q,h
    X = np.vstack((Z,U)) # All possible points
    q = U.shape[0] # No. Unevaluated pts
    m = Z.shape[0]
    n = X.shape[0]
    # Get the Laplacian -> Blindly using KNN
    # Blindly using KNN
    knn = 5
    
    print("X shape", X.shape)
    # nbrs = NearestNeighbors(n_neighbors=knn, algorithm='ball_tree').fit(X)
    # distances, indices = nbrs.kneighbors(X)
    
    # S = np.zeros(((X.shape[0]),X.shape[0])) 
    # for i,j in enumerate(indices):

    #     S[i,j] = 1

    # # Set diagonal enteries to zero
    # S = S - np.eye((X.shape[0]))
    
    # D = np.eye((X.shape[0]))
        
    # for i in range(X.shape[0]):
    #     D[i,i] = S[i,:].sum()
    
    criteria = []
        
    L = SLRV_laplacian(X,knn) #D - S
    
    print("L shape:", L.shape)
    
    X_star = np.ones((n, Z.shape[1]+1))
    print("X_star shape:", X_star.shape)
    for i in range(n):
        X_star[i,1:] = X[i]
        
    for _, xi in enumerate(U):
        
        W = SLRV_get_W(Z,xi,m,h)
        print("w shape:", W.shape )
        
        Z_star = np.ones((m, Z.shape[1]+1))
        for i in range(m):
            Z_star[i,1:] = Z[i] - xi
        
        print("Z_star.T @ W @ Z_star:", Z_star.T @ W @ Z_star )
        print("l1*X.T@L@X:", l1*X.T@L@X )
        print("np.eye((n)):", np.eye((n)) )
        #prod_inv= linalg.inv( Z_star.T @ W @ Z_star + l1*X.T@L@X + l2*np.eye((n)) )
        
        # Replaced X with X_star
        prod_inv= linalg.inv( Z_star.T @ W @ Z_star + l1*X_star.T@L@X_star + l2*np.eye((Z.shape[1]+1)) )
        print("prod_inv shape:", prod_inv.shape)
        res = X_star.T @ prod_inv @ X_star
        
        criteria.append( res.mean().sum() )  # Is this correct?
            
        
    indx = np.argmin(criteria)
    new_point = U[indx]
        
    return new_point, indx
    

