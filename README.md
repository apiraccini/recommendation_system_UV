## English description

This repository contains the project for the course Informatical Methods for Statistics and Data Science, where we built a recommendation system using UV decomposition. The group was composed by myself and my University colleagues Matteo Gasparin and [Pietro Scanzi](https://github.com/pietroscanzi).

In the present work we try to apply the theory of recommendation systems to real data, consisting of music ratings available on the Yahoo! platform. The utility matrix is estimated through UV decomposition, following different methods of gradient descent, specifically enriched with additional features to insert a form of control over the high sparsity rate of the matrix. The techniques proposed and implemented lead to the same conclusions from an efficacy point of view, while presenting, among the considerable variations, significant differences in efficiency.

Contents:
- user_song.tgz:  The file contains the dataset "Yahoo! Music User Ratings of Songs with Artist, Album, and Genre Meta Information, v. 1.0 (1.4 Gbyte & 1.1 Gbyte)" in .tgz format;
- user_song.py: The module contains the create_utility_matrix function, which takes as input the path to the folder in .tgz format and the subpath to the data of interest in .txt format contained therein, and returns the corresponding utility matrix. Optionally, data can also be returned in the format .txt. The module is meant to be imported from a main module that you use the create_utility_matrix function, if it is run as main module creates the utility matrix and prints size and percentage of unassigned ratings;
- extra_functions.py: The module contains several accessory functions used by the modules GDmat.py, SGD.py, SGD2.py The functions are accompanied by comments explaining how they work. The module is meant to be imported from the three aforementioned modules that use the contained functions, if it is launched as main it does not give output.
- GD.py, SGD.py, SGD2.py: The modules contain the functions gradient_descent_mat, stochastic_gradient_descent_mat and stochastic_gradient_descent_mat2 which calculate the UV decomposition of a utility matrix respectively by means of a Gradient Descent algorithm and two versions of a Stochastic Gradient Descent algorithm. These modules are meant to be imported from a main module which use the gradient descent function contained in it, for using them individually they must be launched from the command line. In this case the function is performed on a dummy 5 * 5 matrix saved in the L.txt file inside the compressed folder and on a randomly created matrix with a fixed share of empty revenue per row, equal to approximately 25% of revenue. Functions work with the same input and output parameters, the modules have the same structure and are separated only to be able to allow single execution of command line algorithms.
Parameters:
  - M: Initial utility matrix
  - d: Number of latent dimensions
  - age: Learning rate of the algorithm
  - onlyUV: If True the function returns only U and V trailing, otherwise also Final RMSE, number of iterations to convergence and percentage of correct classification
  - perturb: If True the starting matrices U and V are perturbed, the function returns the final predicted array obtained as averages of the forecasts obtained according to the various perturbations, in addition to the final RMSE and percentage of correct classification
  - nperturb: Number of perturbations to be carried out
  - init: string indicating the array initialization method for U and V: 'ones' (default) initializes U and V by assigning 1 on each entry, 'mean' initializes them so that the matrix estimated starting has each income equal to the average of entries of the starting matrix, 'meanrow' and 'meancol' initialize U and V so that the starting estimated matrix has row (or column) values equal to row averages (or column) of the initial utility matrix
  - method: string indicating the perturbation method to be adopted: 'n' (default) uses a standard normal distribution, 'u' uses a uniform distribution defined from -1 to 1
  - global usage: GDmat.py [-h] [--path PATH] -d D [-eta ETA] [-n N] [-p P] [-perturb PERTURB] [-nperturb NPERTURB] [-init INIT] [-method METHOD] (In the same way using SGD.py or SGD2.py)

- search_for_d_GD.py, search_for_d_SGD.py, search_for_d_SGD2.py (search_for_d_GD.txt, search_for_d_SGD.txt, search_for_d_SGD2.txt):  DO NOT RUN - computationally burdensome. The code output is saved in the corresponding files search_for_d_GD.txt, search_for_d_SGD.txt, search_for_d_SGD2.txt. The modules create the utility matrix corresponding to the user_song dataset using the create_utility_matrix function, they subsequently perform the estimate of empty entries by UV decomposition for each dimension 2 to 6 (with three perturbations and using the default methods for initialization and perturbation). Each module corresponds to one of the three gradient descent algorithms considered, and for each in a text file the results are reported (for each dimension) related to RMSE, percentage of correct classification, average number of iterations and overall execution time. The goal of this analysis is to heuristically identify the best value of d to be used in the final analys.

- analysis1.py, analysis2.py, analysis3.py, analysis4.py (analysis1.txt, analysis4.txt): DO NOT RUN - computationally burdensome. The code output is saved in the corresponding analysis.txt file. The modules implement the analysis of the user_song dataset according to the three proposed methodologies. The four modules correspond to the four initialization methods of the starting matrices U and V ('mean', 'meanrow', 'meancol' and 'ones') and for each method analysis of the matrices thus initialized without perturbations and with three perturbations, either according to a normal distribution and according to a uniform distribution. Estimates of the utility matrix corresponding to the dataset using the GD, SGD and SGD2 methodologies according to 4 * 3 = 12 configurations the parameters of the defined gradient descent functions. The output is saved in tabular format in the analysis.txt file, reporting the following results for each analysis: Algorithm, RMSE, classification, iterations, total time. Based on the results, the methodology and configuration of parameters which leads to the best estimate of the initial utility matrix are chosen according to RMSE and classification rate. At the moment the initialization functions needed to perform the analyzes 2 and 3 are not optimized, so only the analyzes 1 and 4, the results of which are saved in their respective text files.

- analysis_part2.py (analysis_part2.txt, Matrix0.txt): DO NOT RUN - computationally burdensome. The module loads the user_song dataset and then implements the procedure gradient descent (with related parameters) selected as best thanks to the analysis.py module. Given the possibility to focus on a specific configuration of parameters, it is decided to carry out 5 perturbations. The results of the procedure in terms of RMSE, correct classification of observed revenue, average number of iterations and average execution time come saved in the file analysis_part2.txt; the revenue of the corresponding estimated matrix empty cells in the starting utility matrix is saved in text in the Matrix0.txt file.

-reccomender.py: The module defines an object of the MRJob class which takes as input an estimated rating matrix in textual format and returns for each user the three highest rated ratings and the corresponding song IDs.

- presentation_analysis.py (reccomended.png): The module executes the Job MapReduce defined in the module inside it reccomender.py and creates a dictionary with matching key-value pairs to the outputs. Starting from this, we save frequencies and IDs of the suggested songs which were in first place for at least one user, in descending order according to frequency, and the corresponding bar graph is created showing the ID and frequencies, which is saved in the reccomended.png file

- other text files: Obtained by extracting the contents of the user_song.tgz compressed folder using the create_utility_matrix function contained in the user_song.py module.

## Italian description

Questa repository contiene il progetto per il corso Metodi Informatici per la Statistica ed il Data Science, dove abbiamo costruito un sistema di raccomandazione utilizzando la decomposizione UV. Il gruppo era composto da me e i miei colleghi di Università Matteo Gasparin and [Pietro Scanzi](https://github.com/pietroscanzi).

Nel presente lavoro cerchiamo di applicare la teoria dei sistemi di raccomandazione a dati reali, costituiti da valutazioni musicali disponibili sulla piattaforma Yahoo!. La matrice di utilità viene stimata attraverso la decomposizione UV, seguendo diverse modalità di discesa del gradiente, specificamente arricchite di funzionalità aggiuntive per inserire una forma di controllo sull'alto tasso di sparsità della matrice. Le tecniche proposte e implementate portano alle stesse conclusioni dal punto di vista dell'efficacia, pur presentando, tra le diverse varianti, significative differenze di efficienza.
