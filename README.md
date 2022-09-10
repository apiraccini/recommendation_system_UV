# recomendation_system_UV

## English description

This repository contains the project for the course Informatical Methods for Statistics and Data Science, where we built a recommendation system using UV decomposition.

In the present work we try to apply the theory of recommendation systems to real data, consisting of music ratings available on the Yahoo! platform. The utility matrix is estimated through UV decomposition, following different methods of gradient descent, specifically enriched with additional features to insert a form of control over the high sparsity rate of the matrix. The techniques proposed and implemented lead to the same conclusions from an efficacy point of view, while presenting, among the considerable variations, significant differences in efficiency.

Contents:
- user_song.tgz:  The file contains the dataset "Yahoo! Music User Ratings of Songs with Artist, Album, and Genre Meta Information, v. 1.0 (1.4 Gbyte & 1.1 Gbyte) "in .tgz format;
- user_song.py: The module contains the create_utility_matrix function, which takes in input the path to the folder in .tgz format and the subpath to the data of interest in .txt format contained therein, and returns the corresponding utility matrix. Optionally, data in the format can also be returned in .txt. The module is meant to be imported from a main module that you use the create_utility_matrix function, if it is run as main module creates the utility matrix and prints size and percentage of unassigned ratings;
- extra_functions.py: The module contains several accessory functions used by the modules GDmat.py, SGD.py, SGD2.py The functions are accompanied by comments explaining how they work. The module is meant to be imported from the three aforementioned modules they use the contained functions, if it is launched as main it does not give output.
- GD.py, SGD.py, SGD2.py: The modules contain the functions gradient_descent_mat, stochastic_gradient_descent_mat and stochastic_gradient_descent_mat2 which calculate the UV decomposition of a utility matrix respectively by means of a Gradient Descent algorithm and two versions of a Stochastic Gradient Descent algorithm. Functions work with the same input and output parameters, the modules have the same structure and are separated only to be able to allow single execution of command line algorithms
Parameters:
  - M: Initial utility matrix
  - d: Number of latent dimensions
  - age: Learning rate of the algorithm
  - onlyUV: If True the function returns only U and V trailing, otherwise also Final RMSE, number of iterations to convergence and percentage of correct classification
  - perturb: If True the starting matrices U and V are perturbed, the function returns the final predicted array obtained as averages of the forecasts obtained according to the various perturbations, in addition to the final RMSE and percentage of correct classification
  - nperturb: Number of perturbations to be carried out
  - init: string indicating the array initialization method for U and V: 'ones' (default) initializes U and V by assigning 1 on each entry, 'mean' initializes them so that the matrix estimated starting has each income equal to the average of entries of the starting matrix, 'meanrow' and 'meancol' initialize U and V so that the starting estimated matrix has row (or column) values equal to row averages (or column) of the initial utility matrix
  - method: string indicating the perturbation method to be adopted: 'n' (default) uses a standard normal distribution, 'u' uses a uniform distribution defined from -1 to 1
These odules are meant to be imported from a main module which use the gradient descent function contained in it, for using them individually they must be launched from the command line. In this case the function is performed on a dummy 5 * 5 matrix saved in the L.txt file inside the compressed folder and on a randomly created matrix with a fixed share of empty revenue per row, equal to approximately 25% of revenue.
Functions are to be used as follows:
  - usage: GDmat.py [-h] [--path PATH] -d D [-eta ETA] [-n N] [-p P] [-perturb PERTURB] [-nperturb NPERTURB] [-init INIT] [-method METHOD] (In the same way using SGD.py or SGD2.py)
  - optional arguments:
    -h, --help show this help message and exit
    --path PATH path to the file containing the 5 * 5 test matrix
    -d D number of latent dimensions
    -eta ETA learning rate of the GD algorithm (default = 0.005)
    -n N number of rows of the random test matrix (default = 10)
    -p P number of columns in the random test matrix (default = 20)
    -perturb PERTURB do you want to perform a perturbation of the initial U and V matrices? (default = False)
    -nperturb NPERTURB number of perturbations if perturb == True (default = 3)
    -init INIT method of initialization of matrices U and V (default = 'ones', alternative = 'mean')
    -method METHOD perturbation method of matrices U and V (default = 'n', alternative = 'u')

- search_for_d_GD.py, search_for_d_SGD.py, search_for_d_SGD2.py (search_for_d_GD.txt, search_for_d_SGD.txt, search_for_d_SGD2.txt):  DO NOT RUN - computationally burdensome. The code output is saved in the corresponding files search_for_d_GD.txt, search_for_d_SGD.txt, search_for_d_SGD2.txt. The modules create the utility matrix corresponding to the user_song dataset using the create_utility_matrix function, they subsequently perform the estimate of empty entries by UV decomposition for each dimension 2 to 6 (with three perturbations and using the default methods for initialization and perturbation). Each module corresponds to one of the three gradient descent algorithms considered, and for each in a text file the results are reported (for each dimension) related to RMSE, percentage of correct classification, average number of iterations and overall execution time. The goal of this analysis is to heuristically identify the best value of d to be used in the final analys.
