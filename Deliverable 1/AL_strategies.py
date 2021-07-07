# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:52:21 2021

@author: Cillian
"""
import numpy as np
from scipy import linalg
#from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

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
    return new_point, indx

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
        K_zz_inv = linalg.inv( k.K(Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
        
    except:
        # Add jitter
        K_zz_inv = linalg.inv( k.K(Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) + np.eye(k.K(Z).shape[0])*1e-04  ) 
    
    
    for i in range(q):
        reduction = 0
        
        # Term 1

        
        xi = U[i][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        #K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
        
        try:
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )

        
        except:
            # Add jitter
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04  )
        
    
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
    


def SLRGP(Z,U, model,k):
    """This function takes the set of evaluated points, unevaluated points, the GPy object and it's kernel.
    It then choses the next evaluation point based on the Sequential Laplacian Regularized Gaussian Process active learning method.

    :param Z: (numpy array), The set of evaluated points
    :param U: (numpy array), The set of unevaluated points
    :param model: (GPy object), model currently being trained
    :param k: (GPy kernel), kernel of GPy model being trained

    :return: (array, int), the new point to evaluate, the index of the point in array of unevaluated points
    """   
    
    # if not Lambda:
    #     Lambda = 1
    # #k = model.kern
    # #m = Z.shape[0] # No. evaluated pts
    q = U.shape[0] # No. Unevaluated pts
    
    # # All points -> Evaluated and Unevaluated points
    X = np.vstack((Z,U))
    # Y, _ = model.predict(X)
    
    # # Calculate Laplacian Matrix
    # S_x = np.empty((X.shape[0], X.shape[0]))
    # S_y = np.empty((X.shape[0], X.shape[0]))
    # D_x = np.eye(X.shape[0])
    # D_y = np.eye(X.shape[0])
    
    # for i in tqdm(range(X.shape[0])):
    #     for j in range(X.shape[0]):
    #         S_x[i,j] = np.linalg.norm(X[i] - X[j])
    #         S_y[i,j] = np.linalg.norm(Y[i] - Y[j])
    #     D_x[i,i] =  S_x[i,:].sum()
    #     D_y[i,i] =  S_y[i,:].sum()
    
    # L = ( D_y - S_y)/(D_x - S_x)
    
    print("calculating L...")
    L = calculate_laplacian(Z,U,model)
    print("optimising  Lambda...")
    Lambda = optimize_SLRGP_parameter(Z,U,X,k,model,Lambda_vals = [1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2])
    

    
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
    
    P = np.empty((len(Lambda_vals)))
    
    q = U.shape[0] # No. Unevaluated pts
    
    x_star = []
    
    for indx,Lambda in enumerate(Lambda_vals):
        reduction = 0
        
        L = calculate_laplacian(Z,U,model)
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
                K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.rbf.variance*np.eye(k.K(z_xi).shape[0]) )
            except:
                print("adding jitter")
                # Add jitter
                K_X_Zxi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(z_xi,X)@L@k.K(X,z_xi) + k.rbf.variance*np.eye(k.K(z_xi).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04  )

            
            #print("going to get the max of", k.K(U_nxi,U_nxi) - k.K(U_nxi,z_xi)@K_X_Zxi_inv@k.K(z_xi,U_nxi))
            m_i[i] = np.max( k.K(U_nxi,U_nxi) - k.K(U_nxi,z_xi)@K_X_Zxi_inv@k.K(z_xi,U_nxi) )
     
        x_star.append( np.argmax(m_i) )
        
        xi = U[np.argmax(m_i)][None,:]
        z_xi = k.K(np.vstack((Z,xi)))
        try:
            K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) )
            K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) ) 
        
        except:
            print("adding jitter")
            K_z_xi_inv = linalg.inv( k.K(z_xi) + Lambda*k.K(np.vstack((Z,xi)),X)@L@k.K(X,np.vstack((Z,xi))) + k.rbf.variance*np.eye(k.K(np.vstack((Z,xi))).shape[0]) + np.eye(k.K(z_xi).shape[0])*1e-04 )
            K_zz_inv = linalg.inv( k.K(Z) + Lambda*k.K(Z,X)@L@k.K(X,Z) + k.rbf.variance*np.eye(k.K(Z).shape[0]) + np.eye(k.K(Z).shape[0])*1e-04 ) 

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
        
    
def calculate_laplacian(Z,U,model):
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
    
    L = ( D_y - S_y)/(D_x - S_x)
    # print("L:", L)
    # print("D_y:", D_y)
    # print("S_y:", S_y)
    # print("D_x:", D_x)
    # print("S_x:", S_x)
    
    return L
    

def SLRGP_Batch(Z,U, model,k, n_pts = 2):
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
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(z_xi.shape[0] ) )
        except:
            # Add jitter
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(z_xi.shape[0] ) + np.eye(z_xi.shape[0])*1e-04 )

        
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
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(z_xi.shape[0] ) )
        except:
            # Add jitter
            K_z_xi_inv = linalg.inv( k.K(z_xi) + k.rbf.variance*np.eye(z_xi.shape[0] ) + np.eye(z_xi.shape[0])*1e-04 )
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
    

