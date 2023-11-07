#modified from https://github.com/AIAnytime/Create-Vector-Store-from-Scratch/blob/main/vector_store.py

import numpy as np


class VectorStore:
    def __init__(self):
        self.vector_data = {}  # A dictionary to store vectors
        self.vector_index = {}  # An indexing structure for retrieval

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the store.

        Args:
            vector_id (str or int): A unique identifier for the vector.
            vector (numpy.ndarray): The vector data to be stored.
        """
        self.vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id):
        """
        Retrieve a vector from the store.

        Args:
            vector_id (str or int): The identifier of the vector to retrieve.

        Returns:
            numpy.ndarray: The vector data if found, or None if not found.
        """
        return self.vector_data.get(vector_id)

    def _update_index(self, vector_id, vector):
        """
        Update the index with the new vector.

        Args:
            vector_id (str or int): The identifier of the vector.
            vector (numpy.ndarray): The vector data.
        """
        # In this simple example, we use brute-force cosine similarity for indexing
        for existing_id, existing_vector in self.vector_data.items():
            similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector))
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity
     
    def calculate_bm25(self, k1, b, query, corpus_vector):
        corpus_vector = corpus_vector.split()
        dl = len(corpus_vector)
        avgdl = sum(len(self.vector_data.items()[1]))/len(self.vector_data.items()[1])
        freq_qd = []
        for i in range(len(query)):
            freq_qd.append(corpus_vector.count(query[i]))
        
        idf = 1/freq_qd

        score = 0
        for i in range(len(freq_qd)):
            score += idf[i] * ((freq_qd[i] * (k1 + 1))/(freq_qd[i] + k1*(1-b + b*dl/avgdl)))
        return score


    def find_similar_vectors(self, query_vector, num_results=5):
        """
        Find similar vectors to the query vector calculated by BM25.

        Args:
            query_vector (numpy.ndarray): The query vector for similarity search.
            num_results (int): The number of similar vectors to return.

        Returns:
            list: A list of (vector_id, similarity_score) tuples for the most similar vectors.
        """
        results = []
        k1 = 1.2
        b = 0.75
        for vector_id, vector in self.vector_data.items():
            #similarity = self.calculate_bm25(k1, b, query_vector, vector)
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        # Sort by similarity in descending order
        #results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        #return results[:num_results]
        return results



