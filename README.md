# CLS Project
The aim of this project is to develop Gaussian Process (GP) regression surrogate models with Active Learning methods and apply it to the non-intrusive uncertainty quantification of ISR3D, a complex multiscale computational model to simulate In-Stent Restenosis in 3D. 

Active learning provides an iterative approach when building the surrogate model; at each step, the sampling technique identifies the next sample based on some predefined metric computed using the surrogate model. The surrogate model is then retrained using the newly sampled point, and the process is repeated until a maximum number of samples were used.


## Deliverable 1: Implementation of Active Learning Methods

Five active learning methods have been implemented for GP models using the [GPy](http://sheffieldml.github.io/GPy/) package. 


* Active learning MacKay (ALM)[[1]](#1)
* Active Learning Cohn (ALC)[[1]](#1)
* Integrated Mean Squared Error (IMSE) [[2]](#2)
* Maximum Mean Squared Error (MMSE)[[2]](#2)
* sequential Laplacian regularized Gaussian process (SLRGP)[[3]](#3)


## Example Experiments
Run_Sobol_GP.py and Run_Sobol_GS.py provide example functions for calculating Sobol Indices using Gaussian Processes and the provided Gray-Scott model. 
train_GS provides example functions for training multiple surrogate models to emulate the Gray_Scott model with different active learning strategies
train_ISR3D provides example functions for training multiple surrogate models to emulate an ISR3D dataset with different active learning strategies

## Deliverable 2: Uncertainty Quantification of ISR3D Data (IN PROGRESS)

The generated data will be studied and a sensitivity analysis performed on input parameters. 

## References
<a id="1">[1]</a> 
Sambu Seo, Marko Wallat, Thore Graepel, and Klaus Obermayer. Gaussian processregression: Active data selection and test point rejection. InMustererkennung 2000,pages 27–34. Springer, 2000.

<a id="2">[2]</a> 
Jerome Sacks, William J Welch, Toby J Mitchell, and Henry P Wynn.  Design andanalysis of computer experiments.Statistical science, pages 409–423, 1989

<a id="3">[3]</a> 
Rajitha  Meka,  Adel  Alaeddini,  Sakiko  Oyama,  and  Kristina  Langer.   An  activelearning methodology for efficient estimation of expensive noisy black-box functionsusing gaussian process regression.IEEE Access, 8:111460–111474, 2020
