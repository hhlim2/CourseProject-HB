import nltk
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords

#class to create bm25 scores of the textbook chunks based on the input query

class BM25_algorithm:
     
    #initalize bm25 variables
    def __init__(self, corpus, k, b):
        self.num_docs = len(corpus)
        self.avgdl = sum(len(i) for i in corpus) / self.num_docs
        self.k = k
        self.b = b
        self.corpus = corpus


    #https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    #function to remove stopwords from corpus
    def remove_stopwords(self):
        corpus_updated = []
        corpus = self.corpus

        stop_words = stopwords.words('english')
        for doc in corpus: 
            '''doc = doc.lower().split(' ')
            filtered_doc= []
            for w in doc:
                if w not in stop_words:
                    filtered_doc.append(w)'''
            filtered_doc = [w for w in doc.lower().split(' ') if w not in stop_words]

            corpus_updated.append(filtered_doc)
        return corpus_updated

    #function to calculate the bm25 scores of the textbook chunks based on the input query
    def calculate_scores(self, query):
        query = query.split(' ')

        cleaned_corpus = self.remove_stopwords()
      
        avg_dl = self.avgdl
        k1 = self.k
        b = self.b
        score = []
    
        for doc in cleaned_corpus:
            freq_qd = []
            dl = len(doc)

            for query_term in query:
                freq_qd.append(doc.count(query_term))
            
 
            idf = [0 if i == 0 else 1/i for i in freq_qd]
            
            score_doc = 0
            for i in range(len(freq_qd)):
                score_doc += idf[i] * ((freq_qd[i] * (k1 + 1))/(freq_qd[i] + k1*(1-b + b*dl/avg_dl)))

            score.append(score_doc)
        
        return score
