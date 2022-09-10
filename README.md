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
