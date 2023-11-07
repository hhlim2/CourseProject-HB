from newscatcherapi import NewsCatcherApiClient
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from rank_bm25 import BM25Okapi
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM


from vectorstore import VectorStore
import numpy as np

#import environment variables
load_dotenv()

api_key = os.getenv('news_api', None)
newscatcherapi = NewsCatcherApiClient(x_api_key=api_key) 


st.title('CS410 News Retrieval')
query = st.text_input('news question')


def summarize(articles, input_query):
    links = []
    titles = []
    for i in range(len(articles['articles'])):
        links.append(articles['articles'][i]['link'])
        titles.append(articles['articles'][i]['title'])

    loader = WebBaseLoader(links)
    loader.requests_kwargs = {'verify':False, 'timeout':15}
    docs = loader.load()

    CHUNK_SIZE = 3000
    CHUNK_OVERLAP = 100

    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
    texts = text_splitter.split_documents(docs)

    metadata = []
    doc_content = []
    for i in range(len(texts)):
        metadata.append(texts[i].metadata)
        doc_content.append(texts[i].page_content)
    
    tokenized_corpus = [doc.split(" ") for doc in doc_content]

    bm25_corpus = BM25Okapi(tokenized_corpus)

    vector_store = VectorStore()

    vocabulary = set()
    for sentence in doc_content:
        tokens = sentence.lower().split()
        vocabulary.update(tokens)

    # Assign unique indices to words in the vocabulary
    word_to_index = {word: i for i, word in enumerate(vocabulary)}

    # Vectorization
    sentence_vectors = {}
    for sentence in doc_content:
        tokens = sentence.lower().split()
        vector = np.zeros(len(vocabulary))
        for token in tokens:
            vector[word_to_index[token]] += 1
        sentence_vectors[sentence] = vector

    # Storing in VectorStore
    for sentence, vector in sentence_vectors.items():
        vector_store.add_vector(sentence, vector)

    # Searching for Similarity
    query_sentence = input_query
    query_vector = np.zeros(len(vocabulary))
    query_tokens = query_sentence.lower().split()
    for token in query_tokens:
        if token in word_to_index:
            query_vector[word_to_index[token]] += 1

    bm25_similarity = vector_store.find_similar_vectors(query_vector, num_results=2)

    #tokenized_query = input_query.split(" ")
    #bm25_scores = bm25_corpus.get_scores(tokenized_query)

    #relevant_passages = doc_content[np.argmax(bm25_similarity)]
    #relevant_metadata = metadata[np.argmax(bm25_similarity)]
    #similarity = np.max(bm25_similarity)



    # Print similar sentences

    relevant_passages = bm25_similarity[0][0]
    
    similarity = bm25_similarity[0][1]
    #print(relevant_passages)

    
    # Load model directly

    flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")


   
    falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40b")
    falcon_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")


    tokenizer = falcon_tokenizer
    model= falcon_model

    input_text = f"using the following context: {relevant_passages}, answer the following question: {query}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids,
                             do_sample=True,
                             min_length = 50,
                             max_length = 300,
                             temperature = 0.7,
                             top_p = 0.15,
                             top_k = 50)
    res = tokenizer.decode(outputs[0])


    # pipeline function Get predictions
    '''nlp = pipeline('question-answering', model=flan_t5, tokenizer=flan_t5)
    QA_input = {
        'question': query,
        'context': relevant_passages
    }
    res = nlp(QA_input)'''
    #need to return the source

    
    return res, similarity, 

if query:
    all_articles = newscatcherapi.get_search(q=query,
                                        lang='en',
                                        countries='US',
                                        page_size=5,
                                        )
    response, similarity = summarize(all_articles, query)
    st.subheader(response)
    st.caption(similarity)
    #st.caption(relevant_metadata)


        
    


    

    