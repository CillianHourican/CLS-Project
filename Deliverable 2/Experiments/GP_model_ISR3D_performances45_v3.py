# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:05:38 2021

@author: Cillian
"""

import GPy
import AL_strategies
#from utils import *
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,max_error,r2_score,mean_absolute_percentage_error


# def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
#     """Check that y_true and y_pred belong to the same regression task.
#     Parameters
#     ----------
#     y_true : array-like
#     y_pred : array-like
#     multioutput : array-like or string in ['raw_values', uniform_average',
#         'variance_weighted'] or None
#         None is accepted due to backward compatibility of r2_score().
#     Returns
#     -------
#     type_true : one of {'continuous', continuous-multioutput'}
#         The type of the true target data, as output by
#         'utils.multiclass.type_of_target'.
#     y_true : array-like of shape (n_samples, n_outputs)
#         Ground truth (correct) target values.
#     y_pred : array-like of shape (n_samples, n_outputs)
#         Estimated target values.
#     multioutput : array-like of shape (n_outputs) or string in ['raw_values',
#         uniform_average', 'variance_weighted'] or None
#         Custom output weights if ``multioutput`` is array-like or
#         just the corresponding argument if ``multioutput`` is a
#         correct keyword.
#     dtype : str or list, default="numeric"
#         the dtype argument passed to check_array.
#     """
#     check_consistent_length(y_true, y_pred)
#     y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
#     y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)


#     if y_true.ndim == 1:
#         y_true = y_true.reshape((-1, 1))

#     if y_pred.ndim == 1:
#         y_pred = y_pred.reshape((-1, 1))

#     if y_true.shape[1] != y_pred.shape[1]:
#         raise ValueError("y_true and y_pred have different number of output "
#                          "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

#     n_outputs = y_true.shape[1]
#     allowed_multioutput_str = ('raw_values', 'uniform_average',
#                                'variance_weighted')
#     if isinstance(multioutput, str):
#         if multioutput not in allowed_multioutput_str:
#             raise ValueError("Allowed 'multioutput' string values are {}. "
#                              "You provided multioutput={!r}".format(
#                                  allowed_multioutput_str,
#                                  multioutput))
#     elif multioutput is not None:
#         multioutput = check_array(multioutput, ensure_2d=False)
#         if n_outputs == 1:
#             raise ValueError("Custom weights are useful only in "
#                              "multi-output cases.")
#         elif n_outputs != len(multioutput):
#             raise ValueError(("There must be equally many custom weights "
#                               "(%d) as outputs (%d).") %
#                              (len(multioutput), n_outputs))
#     y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

#     return y_type, y_true, y_pred, multioutput

# from sklearn import *
# def mean_absolute_percentage_error(y_true, y_pred,
#                                    sample_weight=None,
#                                    multioutput='uniform_average'):
#     """Mean absolute percentage error regression loss.
#     Note here that we do not represent the output as a percentage in range
#     [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in the
#     :ref:`User Guide <mean_absolute_percentage_error>`.
#     .. versionadded:: 0.24
#     Parameters
#     ----------
#     y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
#         Ground truth (correct) target values.
#     y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
#         Estimated target values.
#     sample_weight : array-like of shape (n_samples,), default=None
#         Sample weights.
#     multioutput : {'raw_values', 'uniform_average'} or array-like
#         Defines aggregating of multiple output values.
#         Array-like value defines weights used to average errors.
#         If input is list then the shape must be (n_outputs,).
#         'raw_values' :
#             Returns a full set of errors in case of multioutput input.
#         'uniform_average' :
#             Errors of all outputs are averaged with uniform weight.
#     Returns
#     -------
#     loss : float or ndarray of floats in the range [0, 1/eps]
#         If multioutput is 'raw_values', then mean absolute percentage error
#         is returned for each output separately.
#         If multioutput is 'uniform_average' or an ndarray of weights, then the
#         weighted average of all output errors is returned.
#         MAPE output is non-negative floating point. The best value is 0.0.
#         But note the fact that bad predictions can lead to arbitarily large
#         MAPE values, especially if some y_true values are very close to zero.
#         Note that we return a large value instead of `inf` when y_true is zero.
#     Examples
#     --------
#     >>> from sklearn.metrics import mean_absolute_percentage_error
#     >>> y_true = [3, -0.5, 2, 7]
#     >>> y_pred = [2.5, 0.0, 2, 8]
#     >>> mean_absolute_percentage_error(y_true, y_pred)
#     0.3273...
#     >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
#     >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
#     >>> mean_absolute_percentage_error(y_true, y_pred)
#     0.5515...
#     >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
#     0.6198...
#     """
#     y_type, y_true, y_pred, multioutput = _check_reg_targets(
#         y_true, y_pred, multioutput)
#     check_consistent_length(y_true, y_pred, sample_weight)
#     epsilon = np.finfo(np.float64).eps
#     mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
#     output_errors = np.average(mape,
#                                weights=sample_weight, axis=0)
#     if isinstance(multioutput, str):
#         if multioutput == 'raw_values':
#             return output_errors
#         elif multioutput == 'uniform_average':
#             # pass None as weights to np.average: uniform mean
#             multioutput = None

#     return np.average(output_errors, weights=multioutput)



#     if y_true.ndim == 1:
#         y_true = y_true.reshape((-1, 1))

#     if y_pred.ndim == 1:
#         y_pred = y_pred.reshape((-1, 1))

#     if y_true.shape[1] != y_pred.shape[1]:
#         raise ValueError("y_true and y_pred have different number of output "
#                          "({0}!={1})".format(y_true.shape[1], y_pred.shape[1]))

#     n_outputs = y_true.shape[1]
#     allowed_multioutput_str = ('raw_values', 'uniform_average',
#                                'variance_weighted')
#     if isinstance(multioutput, str):
#         if multioutput not in allowed_multioutput_str:
#             raise ValueError("Allowed 'multioutput' string values are {}. "
#                              "You provided multioutput={!r}".format(
#                                  allowed_multioutput_str,
#                                  multioutput))
#     elif multioutput is not None:
#         multioutput = check_array(multioutput, ensure_2d=False)
#         if n_outputs == 1:
#             raise ValueError("Custom weights are useful only in "
#                              "multi-output cases.")
#         elif n_outputs != len(multioutput):
#             raise ValueError(("There must be equally many custom weights "
#                               "(%d) as outputs (%d).") %
#                              (len(multioutput), n_outputs))
#     y_type = 'continuous' if n_outputs == 1 else 'continuous-multioutput'

#     return y_type, y_true, y_pred, multioutput

# from sklearn import *
# def mean_absolute_percentage_error(y_true, y_pred,
#                                    sample_weight=None,
#                                    multioutput='uniform_average'):
#     """Mean absolute percentage error regression loss.
#     Note here that we do not represent the output as a percentage in range
#     [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in the
#     :ref:`User Guide <mean_absolute_percentage_error>`.
#     .. versionadded:: 0.24
#     Parameters
#     ----------
#     y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
#         Ground truth (correct) target values.
#     y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
#         Estimated target values.
#     sample_weight : array-like of shape (n_samples,), default=None
#         Sample weights.
#     multioutput : {'raw_values', 'uniform_average'} or array-like
#         Defines aggregating of multiple output values.
#         Array-like value defines weights used to average errors.
#         If input is list then the shape must be (n_outputs,).
#         'raw_values' :
#             Returns a full set of errors in case of multioutput input.
#         'uniform_average' :
#             Errors of all outputs are averaged with uniform weight.
#     Returns
#     -------
#     loss : float or ndarray of floats in the range [0, 1/eps]
#         If multioutput is 'raw_values', then mean absolute percentage error
#         is returned for each output separately.
#         If multioutput is 'uniform_average' or an ndarray of weights, then the
#         weighted average of all output errors is returned.
#         MAPE output is non-negative floating point. The best value is 0.0.
#         But note the fact that bad predictions can lead to arbitarily large
#         MAPE values, especially if some y_true values are very close to zero.
#         Note that we return a large value instead of `inf` when y_true is zero.
#     Examples
#     --------
#     >>> from sklearn.metrics import mean_absolute_percentage_error
#     >>> y_true = [3, -0.5, 2, 7]
#     >>> y_pred = [2.5, 0.0, 2, 8]
#     >>> mean_absolute_percentage_error(y_true, y_pred)
#     0.3273...
#     >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
#     >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
#     >>> mean_absolute_percentage_error(y_true, y_pred)
#     0.5515...
#     >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
#     0.6198...
#     """
#     y_type, y_true, y_pred, multioutput = _check_reg_targets(
#         y_true, y_pred, multioutput)
#     check_consistent_length(y_true, y_pred, sample_weight)
#     epsilon = np.finfo(np.float64).eps
#     mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
#     output_errors = np.average(mape,
#                                weights=sample_weight, axis=0)
#     if isinstance(multioutput, str):
#         if multioutput == 'raw_values':
#             return output_errors
#         elif multioutput == 'uniform_average':
#             # pass None as weights to np.average: uniform mean
#             multioutput = None

#     return np.average(output_errors, weights=multioutput)


def GP_model(batch_size,retrieved_Inputs,Outputs,AL_method,test_inputs, test_outputs,parameter_ranges, max_eval = 45):
    """Train Gaussian Process surrigate for a given function

    Parameters UPDATE and add SHAPES for inputs/outputs
    ----------
    retrieved_Inputs : (Mxn) numpy array
        Collection of possible input values to the black-box
    Outputs : (Mx1) numpy array
        Collection of corresponding output values from the black-box
    AL_method : str
        Active learning stratrgy to use. See AL_strategies for available options
    parameter_ranges: list of lists (length n)
        Specifies the ranges of each parameter
    M : int 
        Number of equally spaced points to use for each input parameter
    max_eval: int
        The number of times the active learning strategy is used to pick a new sample point
    
    
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
    #kb = GPy.kern.Linear(input_dim=n, ARD=True)
    k = kg#*kb
    
    
    # Randomly pick 20 starting points
    indx = np.random.randint(retrieved_Inputs.shape[0], size = 5)
    #indx = np.random.choice(retrieved_Inputs.shape[0], size = 3, replace=False)
    Inputs = retrieved_Inputs[indx,:]
    
    # Evaluate LHS points
    Y_vals =  Outputs[indx][:,None]#.squeeze()
    Outputs = np.delete(Outputs,indx,axis = 0)
    
    # Define set of unevaluated points
    uneval_pts = np.delete(retrieved_Inputs,indx,axis = 0)
    
    # Train regression function
    model = GPy.models.GPRegression(Inputs,Y_vals,kernel = k)
    model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)

    # optimise parameters
    model.optimize()
    # model.Gaussian_noise.variance = model.Y.var()*0.01
    # model.Gaussian_noise.variance.fix()
    # model.optimize(max_iters=100, messages=False)
    # model.Gaussian_noise.variance.unfix()
    # model.optimize(max_iters=400, messages=False)
    model.optimize_restarts(num_restarts = 10, verbose=False)
    
    RMSE = []
    MAX = []
    r2 = []
    MAPE = []
    
    GP_means, GP_vars = model.predict(test_inputs)
              
    # Get RMSE of Validation data
    RMSE.append( mean_squared_error(GP_means, test_outputs, squared=False) )
      
    # Get Max Error
    MAX.append(max_error(GP_means, test_outputs) )
      
    # R2 Score
    r2.append(r2_score(test_outputs,GP_means) )
    
    MAPE.append(mean_absolute_percentage_error(test_outputs,GP_means))

    it = 0
    while it < int(max_eval/batch_size): #int(max_eval):#
               
        # Apply active learning strategy to find new points
        #print("Inputs:", Inputs.shape)
        #print("Y_vals:", Y_vals.shape)
        new_point, new_indx = AL_method(Inputs, uneval_pts, model,k,batch_size,Y_vals )
    
        # # Remove the new point from the set of unevaluated points
                            #uneval_pts = np.delete(uneval_pts,new_indx,axis = 0)
        
        #print("New indices", new_indx)
        #print("New points", new_point)
        
                            # # Evaluate new point
                            # New_eval = Outputs[new_indx].squeeze()
                            # Outputs = np.delete(Outputs,new_indx,axis = 0)
                        
                            # # Add new point to inputs and outputs
                            # Inputs = np.vstack((Inputs,new_point))
                            # Y_vals = np.vstack((Y_vals,New_eval))
        
        for pt in range(len(new_indx)):
                        
            # Evaluate new point
            New_eval = Outputs[new_indx[pt]]#.squeeze()[pt]
            #New_eval = Outputs[new_indx][pt]
            Outputs = np.delete(Outputs,new_indx[pt],axis = 0)
        
            # Add new point to inputs and outputs
            Inputs = np.vstack((Inputs,new_point[pt]))
            Y_vals = np.vstack((Y_vals,New_eval))
        
        # Train regression model
        model = GPy.models.GPRegression(Inputs,Y_vals,k)
        model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
        #model.constrain_positive('.*') ######## WHY is this here?  

        # optimize the model
        model.optimize()
        # model.Gaussian_noise.variance = model.Y.var()*0.01
        # model.Gaussian_noise.variance.fix()
        # model.optimize(max_iters=100, messages=False)
        # model.Gaussian_noise.variance.unfix()
        # model.optimize(max_iters=400, messages=False)
        model.optimize_restarts(num_restarts = 10, verbose=False)
        
        GP_means, GP_vars = model.predict(test_inputs)
        
              
        # Get RMSE of Validation data
        RMSE.append( mean_squared_error(GP_means, test_outputs, squared=False) )
          
        # Get Max Error
        MAX.append(max_error(GP_means, test_outputs) )
          
        # R2 Score
        r2.append(r2_score(test_outputs,GP_means) )
        
        MAPE.append(mean_absolute_percentage_error(test_outputs,GP_means))




        # Remove the new point from the set of unevaluated points
        #uneval_pts = np.delete(uneval_pts,new_indx,axis = 0)
        
        for pt in range(len(new_indx)):
            # Remove the new point from the set of unevaluated points
            uneval_pts = np.delete(uneval_pts,new_indx[pt],axis = 0)
            
            # Batch Sampling Problem: If smaller indices are removed before larger ones
    
        # Print something so you have soething to stare at while the code runs..
        print("completed IT;",it)
        it += 1
    
        # if it % 50 ==0: 
        #     # Use the model to make predictions for unevaluated points
        #     post_means, post_vars = model.predict(uneval_pts) ##########################
        #     print("variances range:", min(post_vars), max(post_vars))
            
    
    return(model, Inputs,Outputs, RMSE, MAX,r2,MAPE )




def GP_model_FULL(batch_size,retrieved_Inputs,Outputs,AL_method,test_inputs, test_outputs,parameter_ranges, max_eval = 45):
    """Train Gaussian Process surrigate for a given function

    Parameters UPDATE and add SHAPES for inputs/outputs
    ----------
    retrieved_Inputs : (Mxn) numpy array
        Collection of possible input values to the black-box
    Outputs : (Mx1) numpy array
        Collection of corresponding output values from the black-box
    AL_method : str
        Active learning stratrgy to use. See AL_strategies for available options
    parameter_ranges: list of lists (length n)
        Specifies the ranges of each parameter
    M : int 
        Number of equally spaced points to use for each input parameter
    max_eval: int
        The number of times the active learning strategy is used to pick a new sample point
    
    
    Returns
    -------
    GPy regression model
        a trained gaussian process model
    """ 

    # Number of parameters
    n = len(parameter_ranges)
    
    # Define kernel
    kg = GPy.kern.RBF(n, .5, np.ones(n) * 2., ARD=True)

    k = kg#*kb

    Inputs = retrieved_Inputs
    
    # Evaluate LHS points
    Y_vals =  Outputs[:,None]#.squeeze()
    
    # Train regression function
    model = GPy.models.GPRegression(Inputs,Y_vals,kernel = k)
    model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)

    # optimise parameters
    model.optimize()
    model.optimize_restarts(num_restarts = 10, verbose=False)
    
    GP_means, GP_vars = model.predict(test_inputs)
              
    # Get RMSE of Validation data
    RMSE = mean_squared_error(GP_means, test_outputs, squared=False)
      
    # Get Max Error
    MAX = max_error(GP_means, test_outputs) 
      
    # R2 Score
    r2 = r2_score(test_outputs,GP_means) 
    
    MAPE = mean_absolute_percentage_error(test_outputs,GP_means)
    
    
    return(model, Inputs,Outputs, RMSE, MAX,r2,MAPE )
            


def GP_model_batchClustering(batch_size,retrieved_Inputs,Outputs,AL_method,test_inputs, test_outputs,parameter_ranges, max_eval = 45):
    """Train Gaussian Process surrigate for a given function

    Parameters UPDATE and add SHAPES for inputs/outputs
    ----------
    retrieved_Inputs : (Mxn) numpy array
        Collection of possible input values to the black-box
    Outputs : (Mx1) numpy array
        Collection of corresponding output values from the black-box
    AL_method : str
        Active learning stratrgy to use. See AL_strategies for available options
    parameter_ranges: list of lists (length n)
        Specifies the ranges of each parameter
    M : int 
        Number of equally spaced points to use for each input parameter
    max_eval: int
        The number of times the active learning strategy is used to pick a new sample point
    
    
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
    #kb = GPy.kern.Linear(input_dim=n, ARD=True)
    k = kg#*kb
    
    
    # Randomly pick 20 starting points
    indx = np.random.randint(retrieved_Inputs.shape[0], size = 5)
    #indx = np.random.choice(retrieved_Inputs.shape[0], size = 3, replace=False)
    Inputs = retrieved_Inputs[indx,:]
    
    # Evaluate LHS points
    Y_vals =  Outputs[indx][:,None]#.squeeze()
    Outputs = np.delete(Outputs,indx,axis = 0)
    
    # Define set of unevaluated points
    uneval_pts = np.delete(retrieved_Inputs,indx,axis = 0)
    
    # Train regression function
    model = GPy.models.GPRegression(Inputs,Y_vals,kernel = k)
    model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)

    # optimise parameters
    model.optimize()
    # model.Gaussian_noise.variance = model.Y.var()*0.01
    # model.Gaussian_noise.variance.fix()
    # model.optimize(max_iters=100, messages=False)
    # model.Gaussian_noise.variance.unfix()
    # model.optimize(max_iters=400, messages=False)
    model.optimize_restarts(num_restarts = 10, verbose=False)
    
    RMSE = []
    MAX = []
    r2 = []
    MAPE = []
    
    GP_means, GP_vars = model.predict(test_inputs)
              
    # Get RMSE of Validation data
    RMSE.append( mean_squared_error(GP_means, test_outputs, squared=False) )
      
    # Get Max Error
    MAX.append(max_error(GP_means, test_outputs) )
      
    # R2 Score
    r2.append(r2_score(test_outputs,GP_means) )
    
    MAPE.append(mean_absolute_percentage_error(test_outputs,GP_means))
    #new_indx =1
    it = 0
    while it < int(max_eval/batch_size): #int(max_eval):#
               
        # Apply active learning strategy to find new points
        #print("Inputs:", Inputs.shape)
        #print("Y_vals:", Y_vals.shape)
        new_point, new_indx = AL_method(Inputs, uneval_pts, model,k,batch_size,Y_vals )
    
        # # Remove the new point from the set of unevaluated points
                            #uneval_pts = np.delete(uneval_pts,new_indx,axis = 0)
        
        #print("New indices", new_indx)
        #print("New points", new_point)
        
                            # # Evaluate new point
                            # New_eval = Outputs[new_indx].squeeze()
                            # Outputs = np.delete(Outputs,new_indx,axis = 0)
                        
                            # # Add new point to inputs and outputs
                            # Inputs = np.vstack((Inputs,new_point))
                            # Y_vals = np.vstack((Y_vals,New_eval))
        
        for pt in range(len(new_indx)):
                        
            # Evaluate new point
            New_eval = Outputs[new_indx[pt]]#.squeeze()[pt]
            #New_eval = Outputs[new_indx][pt]
            #Outputs = np.delete(Outputs,new_indx[pt],axis = 0)
        
            # Add new point to inputs and outputs
            Inputs = np.vstack((Inputs,new_point[pt]))
            Y_vals = np.vstack((Y_vals,New_eval))
            
        for pt in np.sort(new_indx)[::-1]:
            Outputs = np.delete(Outputs,pt,axis = 0)
        
        # Train regression model
        model = GPy.models.GPRegression(Inputs,Y_vals,k)
        model['Gaussian_noise.variance'].constrain_bounded(1e-1,1e2)
        #model.constrain_positive('.*') ######## WHY is this here?  

        # optimize the model
        model.optimize()
        # model.Gaussian_noise.variance = model.Y.var()*0.01
        # model.Gaussian_noise.variance.fix()
        # model.optimize(max_iters=100, messages=False)
        # model.Gaussian_noise.variance.unfix()
        # model.optimize(max_iters=400, messages=False)
        model.optimize_restarts(num_restarts = 10, verbose=False)
        
        GP_means, GP_vars = model.predict(test_inputs)
        
              
        # Get RMSE of Validation data
        RMSE.append( mean_squared_error(GP_means, test_outputs, squared=False) )
          
        # Get Max Error
        MAX.append(max_error(GP_means, test_outputs) )
          
        # R2 Score
        r2.append(r2_score(test_outputs,GP_means) )
        
        MAPE.append(mean_absolute_percentage_error(test_outputs,GP_means))




        # Remove the new point from the set of unevaluated points
        #uneval_pts = np.delete(uneval_pts,new_indx,axis = 0)
        
        for pt in np.sort(new_indx)[::-1]:
            # Remove the new point from the set of unevaluated points
            uneval_pts = np.delete(uneval_pts,pt,axis = 0)
    
        # Print something so you have soething to stare at while the code runs..
        print("completed IT;",it)
        it += 1
    
        # if it % 50 ==0: 
        #     # Use the model to make predictions for unevaluated points
        #     post_means, post_vars = model.predict(uneval_pts) ##########################
        #     print("variances range:", min(post_vars), max(post_vars))
            
    
    return(model, Inputs,Outputs, RMSE, MAX,r2,MAPE )


            