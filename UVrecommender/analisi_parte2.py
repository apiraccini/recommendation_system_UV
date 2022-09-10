# -*- coding: utf-8 -*-

import numpy as np
import math
import time
import datetime as dt

#import user_song as us
import extra_functions as ef

import SGD2


"""Dataset"""

#utility, dati = us.create_utility_matrix(True)

#n = np.shape(utility)[0]
#p = np.shape(utility)[1]
#m = np.shape(dati)[0]
#nan_perc = 100*np.round(1 - m/(n*p),5)      
    
#print("\nMatrice di utilità (n*p) con n = {} users e p = {} canzoni\nPercentuale di rating non assegnati : {} %\n".format(n,p, nan_perc))



"""Metodologia di analisi e parametri"""

utility = ef.test_matrix(200, 30)
d = 2
eta = 0.0005
perturb = True
nperturb = 5
method = "u"
init = "ones"

print("""\nMetodologia di analisi scelta:\n Algoritmo SGD con matrici U e V inizializzate 
      con valori tutti uguali e pari a 1, con due perturbazioni
      secondo una distribuzione uniforme tra -1 e +1 delle matrici U e V inizializzate
      in questo modo""")



"""Analisi"""

path = "analisi_parte2.txt"
f = open(path, "w") 
f.write("Algoritmo\tRMSE\t\tclassificazione\t\titerazioni\ttempo totale\n")
time.sleep(1)
t1 = dt.datetime.now()
U, V, rmse, num_iter, perc_corretti = SGD2.stochastic_gradient_descent_mat2(utility, d, eta, perturb = True, nperturb = 5, method = "u", init = "ones")
t2 = dt.datetime.now()
delta = t2 - t1
f.write("GD\t\t{}\t\t{}%\t\t\t{}\t\t{}\n".format(np.round(rmse,5), np.round(perc_corretti,2), num_iter, delta)) 




utility_hat = np.dot(U, V)
print("Dimensioni matrice stimata: ", utility_hat.shape)
"""Si salva la matrice stimata in formato testuale, ignorando i valori che
   erano già presenti nella matrice di utilità iniziale
   I file hanno un numero di righe non maggiore di 10000, in modo da poter essere
   processati mediante MapReduce"""


ef.matrix_to_text(utility_hat, utility, begin = 0, end = utility_hat.shape[0], filenum = 0)    