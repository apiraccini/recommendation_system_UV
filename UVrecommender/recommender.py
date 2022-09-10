from mrjob.job import MRJob


class MRJ_recommender(MRJob): 
    """Restituisce per ogni utente i primi tre item 
    raccomandati secondo la matrice di utilità stimata
    in ingresso"""
    
    def mapper(self, _, line):
        l=line.split(",")
        yield l[0], [l[2],l[1]]
    
    def reducer(self, key, values):
        yield key, sorted(values, reverse=True)[0:3]  
        
        
if __name__=="__main__":
    MRJ_recommender.run()