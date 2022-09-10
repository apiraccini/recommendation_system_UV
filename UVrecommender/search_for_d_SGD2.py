# -*- coding: utf-8 -*-

import time
import datetime as dt
import numpy as np

import SGD2

import user_song as us
#import extra_functions as ef

utility = us.create_utility_matrix()
#utility = ef.test_matrix(5, 5)

d = [2, 3, 4, 5, 6]


"""Analisi su SGD2"""
delta_time = list()
rmse = list()
num_iter = list()
perc_corretti = list()

for i in range(len(d)):
    print("Dim = {}".format(d[i]))
    time.sleep(1)
    t1 = dt.datetime.now()
    U_i, V_i, rmse_i, num_iter_i, perc_corretti_i = SGD2.stochastic_gradient_descent_mat2(utility, d = d[i], eta = 0.0001, perturb = True, nperturb = 3)
    t2 = dt.datetime.now()
    rmse.append(rmse_i)
    num_iter.append(num_iter_i)
    perc_corretti.append(perc_corretti_i)
    delta = t2 - t1
    delta_time.append(delta)
    
path = "search_for_d_SGD2.txt"
f = open(path, "w") 

    
f.write("Analisi delle prestazioni di SGD2 (eta = 0.0001, 3 perturbazioni secondo una distribuzione normale standard) sul dataset user_song in base al numero di dimensioni latenti\n\n")    
print("\nAnalisi delle prestazioni di SGD2 (eta = 0.0001, 3 perturbazioni secondo una distribuzione normale standard) sul dataset user_song in base al numero di dimensioni latenti\n")    
f.write("Dimensione\tRMSE\t\tclassificazione\t\titerazioni\ttempo totale\n")    
for i in range(len(d)):
    f.write("{}\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(d[i], np.round(rmse[i],5), np.round(perc_corretti[i],2), num_iter[i], delta_time[i]))
    print("{}\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(d[i], np.round(rmse[i],5), np.round(perc_corretti[i],2), num_iter[i], delta_time[i]))
f.close()