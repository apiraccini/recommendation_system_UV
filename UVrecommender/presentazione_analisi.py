import numpy as np
import matplotlib.pyplot as plt

import user_song as us
from recommender import MRJ_recommender


"""Dataset"""

utility, dati = us.create_utility_matrix(True)

n = np.shape(utility)[0]
p = np.shape(utility)[1]


"""Utilizzando un modulo che implementa un algoritmo di Map-reduce prendendo come input
la matrice stimata in formato testuale, si ottiene un dizionario chiave - valore che
ha come chiave l'ID dell'utente e come valore tre liste contenenti rating e ID delle
tre canzoni maggiormente suggerite per quell'utente 
Map-reduce processa al massimo 10000 righe"""

v = dict()
mr_job = MRJ_recommender(args=['--runner', 'inline', 'Matrix0.txt'])
with mr_job.make_runner() as runner:
    runner.run()
    job_output = mr_job.parse_output(runner.cat_output())
    for key, value in job_output:
        v[int(key)] = value
 

"""Tabella di frequenza riportante le canzoni che sono state suggerite
al primo posto a un qualsiasi utente in ordine decrescente e le relative 
frequenze"""

table = np.zeros((p,2)) 
top_counts = list()
top_counts_songsID = list()

toplot_y = list()

for i in range(n):
    table[int(v[i][0][1]),0] += 1
    table[int(v[i][0][1]),1]  = v[i][0][1]

sorted_table = table[table[:,0].argsort()[::-1]]
for i in range(len(sorted_table)):
    if sorted_table[i,0] > 0:
        top_counts.append(sorted_table[i,0])
        top_counts_songsID.append(int(sorted_table[i,1]))
        
        

"""Plot"""

toplot_x = np.arange(len(top_counts))

plt.bar(toplot_x, top_counts, color = '#1ED760', edgecolor = 'black')
 
plt.title('Tabella di frequenza delle canzoni più suggerite')
plt.xlabel('ID Canzoni suggerite più volte al primo posto') 
plt.ylabel('Frequenza di raccomandazione')
plt.xticks(list(range(len(top_counts))),top_counts_songsID)
for i in range(len(top_counts)):
    plt.text(x = list(range(len(top_counts)))[i]-0.3 , y = top_counts[i]+100, s = 'n={}'.format(int(top_counts[i])), size = 11) 
plt.subplots_adjust(bottom= 0.2, top = 1.05)
plt.show()

plt.savefig("reccomended.png")