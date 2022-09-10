# -*- coding: utf-8 -*-

import tarfile
import numpy as np


def create_utility_matrix(keep_original_data = False):
    
    """Yahoo! Music ratings for User Selected and 
    Randomly Selected songs, version 1.0 (1.2 MB)"""
    tar = tarfile.open("user_song.tgz","r")
    tar.getmembers()
    tar.extractall()
    tar.close()

    dati = np.genfromtxt("./ydata-ymusic-rating-study-v1_0-train.txt",dtype=int)
    """Dati organizzati in: ID user, ID song, Rating"""
    n = max(dati[:,0])
    p = max(dati[:,1])
    m = np.shape(dati)[0]      
    """15400 users x 1000 items ,311704 ratings totali"""

    utility = np.empty((n,p))  
    utility[:] = float('nan')
       
    """Creazione matrice di utilita'"""
    for i in range(m):
        utility[dati[i,0]-1,dati[i,1]-1] = dati[i,2]
    if keep_original_data == True:    
        return utility, dati    
    return utility

   
if __name__ == "__main__":
    
    utility, dati = create_utility_matrix(True)
        
    n = np.shape(utility)[0]
    p = np.shape(utility)[1]
    m = np.shape(dati)[0]
    nan_perc = 100*np.round(1 - m/(n*p),5)      
    
    print("\nMatrice di utilità (n*p) con n = {} users e p = {} canzoni".format(n,p))
    print("Percentuale di rating non assegnati : {} %".format(nan_perc))