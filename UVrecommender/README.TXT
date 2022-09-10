Contenuto della cartella compressa:

- README.TXT

	Il presente file di testo, che illustra il contenuto della cartella compressa


- user_song.tgz

	Il file contiene il dataset "Yahoo! Music User Ratings of Songs with 
	Artist, Album, and Genre Meta Information, v. 1.0 
	(1.4 Gbyte & 1.1 Gbyte)" in formato .tgz


- user_song.py

	Il modulo contiene la funzione create_utility_matrix, che prende in 
	input il percorso alla cartella in formato .tgz e il sottopercorso 
	ai dati di interesse in formato .txt in essa contenuti, e restituisce 
	la matrice di utilità corrispondente

	Opzionalmente possono essere restituiti anche i dati nel formato 
	.txt di origine

	Il modulo è pensato per essere importato da un modulo principale che 
	utilizzi la funzione create_utility_matrix, se viene eseguito come
	modulo principale crea la matrice di utilità e ne stampa a schermo
	dimensioni e percentuale di rating non assegnati


- extra_functions.py
	
	Il modulo contiene diverse funzioni accessorie utilizzate dai moduli
	GDmat.py, SGD.py, SGD2.py

	Le funzioni sono accompagnate da commenti che ne spiegano il funzionamento

	Il modulo è pensato per essere importato dai tre moduli sopracitati che ne
	utilizzano le funzioni contenute, se viene lanciato come main non da output


- GD.py, SGD.py, SGD2.py
	
	I moduli contengono le funzioni gradient_descent_mat, 
	stochastic_gradient_descent_mat e stochastic_gradient_descent_mat2
	che calcolano la decomposizione UV di una matrice di utilità 
	rispettivamente mediante un'algoritmo di Gradient Descent e due 
	versioni di un algoritmo di Stochastic Gradient Descent.

	Le funzioni lavorano con gli stessi parametri di input e output, 
	i moduli hanno la medesima struttura e sono separati solo per poter 
	permettere l'esecuzione singola degli algoritmi da riga di comando

	Parametri:
	M:	Matrice di utilità iniziale
	d:	Numero di dimensioni latenti
	eta:	Learning rate dell'algoritmo
	onlyUV:	Se True la funzione ritorna solo U e V finali, altrimenti anche 
		RMSE finale, numero di iterazioni a convergenza e percentuale di 
		corretta classificazione
	perturb: Se True le matrici U e V di partenza vengono perturbate, la 
		funzione restituisce la matrice prevista finale ottenuta come
		medie delle previsioni ottenute secondo le varie perturbazioni, 
		oltre a RMSE e percentuale di corretta classificazione finali
	nperturb: Numero di perturbazioni da effettuare
	init: stringa indicante il metodo di inizializzazione delle matrici 
	     	U e V di partenza: 'ones'(default) inizializza U e V assegnando 
		1 ad ogni entrata, 'mean' le inizializza in modo che la matrice 
		stimata di partenza abbia ogni entrata pari alla media delle 
		entrate della matrice di partenza, 'meanrow' e 'meancol' 
		inizializzano U e V in modo che la matrice stimata di partenza 
		abbia valori di riga (o di colonna) pari alle medie di riga 
		(o di colonna) della matrice di utilità iniziale	
	method: stringa indicante il metodo di perturbazione da adottare:
		'n'(default) utilizza una distribuzione normale standard,
		'u' utilizza una distribuzione uniforme definita da -1 a 1 

	I moduli sono pensati per essere importati da un modulo principale che 
	utilizzi la funzione di discesa del gradiente in essa contenuti, per 
	utilizzarli singolarmente devono essere lanciati da riga di comando

	In tal caso la funzione viene eseguita su una matrice fittizia 5*5 
	salvata nel file L.txt all'interno della cartella compressa e su
	una matrice creata casualmente con una quota fissa di entrate vuote
	per riga, pari al 25% circa delle entrate

	Il funzionamento è il seguente :
	
	usage: GDmat.py [-h] [--path PATH] -d D [-eta ETA] [-n N] [-p P] 
			[-perturb PERTURB] [-nperturb NPERTURB] [-init INIT]
                	[-method METHOD]
	(Nello stesso modo utilizzando SGD.py o SGD2.py)	

	optional arguments:
  	-h, --help          show this help message and exit
  	--path PATH         percorso al file contenente la matrice di prova 5*5
  	-d D                numero di dimensioni latenti
  	-eta ETA            learning rate dell'algoritmo GD (default = 0.005)
 	-n N                numero di righe della matrice di prova casuale (default = 10)
  	-p P                numero di colonne della matrice di prova casuale (default = 20)
 	-perturb PERTURB    si vuole effettuare una perturbazione delle matrici U e V iniziali ? (default = False)
  	-nperturb NPERTURB  numero di perturbazioni se perturb==True (default = 3)
  	-init INIT          metodo di inizializzazione delle matrici U e V (default='ones',alternativa='mean')
  	-method METHOD      metodo di perturbazione delle matrici U e V (default='n', alternativa='u')


- search_for_d_GD.py, search_for_d_SGD.py, search_for_d_SGD2.py
- (search_for_d_GD.txt, search_for_d_SGD.txt, search_for_d_SGD2.txt)

	NON ESEGUIRE - computazionalmente oneroso
	L'output del codice viene salvato nei corrispondenti file
	search_for_d_GD.txt, search_for_d_SGD.txt, search_for_d_SGD2.txt

	I moduli creano la matrice di utilita' corrispondente al dataset user_song 
	utilizzando la funzione create_utility_matrix, successivamente effettuano 
	la stima delle entrate vuote mediante decomposizione UV per ogni dimensione
	da 2 a 6 (con tre perturbazioni e utilizzando i metodi di default per 
	inizializzazione e perturbazione)

	Ogni modulo corrisponde ad uno dei tre algoritmi di discesa del gradiente 
	considerati, e per ognuno in un file di testo vengono riportati i risultati 
	(per ogni dimensione) relativi a RMSE, percentuale di corretta classificazione,
	numero medio di iterazioni e tempo complessivo di esecuzione

	L'obiettivo di queta analisi e' individuare euristicamente il miglior valore
	di d da utilizzare nell'analisi finale


- analisi1.py, analisi2.py, analisi3.py, analisi4.py
- (analisi1.txt, analisi4.txt)

	NON ESEGUIRE - computazionalmente oneroso
	L'output del codice viene salvato nel corrispondente file analisi.txt

	I moduli implementano l'analisi del dataset user_song secondo le tre 
	metodologie proposte. I quattro moduli corrispondono ai quattro
	metodi di inizializzazione delle matrici U e V di partenza ('mean', 
	'meanrow', 'meancol' e 'ones') e per ogni metodo si effettuano analisi delle 
	matrici così inizializzate senza perturbazioni e con tre perturbazioni, sia
	secondo una distribuzione normale che secondo una distribuzione uniforme

	Vengono quindi eseguite stime della matrice di utilità corrispondente al
	dataset mediante le metodologie GD, SGD e SGD2 secondo 4*3=12 configurazioni
	dei parametri delle funzioni di discesa del gradiente definite

	L'output viene salvato in formato tabellare nel file analisi.txt,
	riportando per ogni analisi i seguenti risultati:
		Algoritmo, RMSE, classificazione, iterazioni, tempo totale

	Sulla base dei risultati, si sceglie dunque la metodologia e la configurazione 
	di parametri che porta alla migliore stima della matrice di utilità iniziale
	secondo RMSE e classificazione

	Al momento le funzioni di inizializzazione necessarie per effettuare le analisi 
	2 e 3 non sono ottimizzate, pertanto è stato possibile effettuare solo le 
	analisi 1 e 4, i quali risultati vengono salvati nei rispettivi file testuali


- analisi_parte2.py
- (analisi_parte2.txt, Matrix0.txt)

	NON ESEGUIRE - computazionalmente oneroso

	Il modulo carica il dataset user_song e successivamente implementa la procedura 
	di discesa del gradiente (con i relativi parametri) selezionata come migliore 
	grazie al modulo analisi.py

	Vista la possibilità di focalizzarsi su una configurazione specifica di 
	parametri, si decide di effettuare 5 perturbazioni

	I risultati della procedura in termini di RMSE, corretta classificazione delle 
	entrate osservate, numero medio di iterazioni e tempo medio di esecuzione vengono
	salvati nel file analisi_parte2.txt; le entrate della matrice stimata corrispondenti
	a celle vuote nella matrice di utilità di partenza viene salvata in formato
	testuale nel file Matrix0.txt 


- reccomender.py

	Il modulo definisce un oggetto della classe MRJob che prende in input una 
	matrice di rating stimati in formato testuale e restituisce per ogni utente 
	i tre rating maggiori stimati e gli ID delle canzoni corrispondenti


- presentazione_analisi.py
- (reccomended.png)

	Il modulo fa eseguire al suo interno il Job MapReduce definito nel modulo 
	reccomender.py e crea un dizionario con le coppie chiave-valore corrispondenti
	agli output

	A partire da questo, si salvano le frequenze e gli ID delle canzoni suggerite
	al primo posto ad almeno un utente, in ordine discendente secondo la frequenza,
	e si crea il corrispondente grafico a barre riportante ID e frequenze, che viene
	salvato nel file reccomended.png


- altri file di testo
	
	Ottenuti estraendo il contenuto della cartella compressa user_song.tgz
	mediante la funzione create_utility_matrix contenuta nel modulo user_song.py
