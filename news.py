from newscatcherapi import NewsCatcherApiClient
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from rank_bm25 import BM25Okapi
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from bm25 import BM25_algorithm

import json
# Python 3
import http.client, urllib.parse
#from vectorstore import VectorStore
import numpy as np


from huggingface_hub import login
login(token='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')

import warnings
warnings.filterwarnings("ignore")

#import environment variables
load_dotenv()

st.title('CS410 News Retrieval')
query = st.text_input('news question')




# Load model directly
#tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
#model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")



tokenizer = tokenizer
model= model




with st.sidebar:
    min_length_entry = st.number_input('minimum length of output', min_value=1, value = 50)
    max_length_entry = st.number_input('maximum length of output', min_value= min_length_entry + 1, value = 150)
    temperature_entry = st.slider('temperature', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    top_p_entry = st.slider('top_p', min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    top_k_entry = st.number_input('top_k', min_value=0, max_value=100, value=50)
    k1_entry = st.number_input('bm25 k1', min_value=0.0, max_value= 100.0, value=1.2)
    b_entry = st.slider('bm25 b', min_value=0.0, max_value= 1.0, value=.75, step=0.01)
    submit = st.button(label='submit')

if submit:
    #model parameters
    min_length = min_length_entry
    max_length = max_length_entry
    temperature = temperature_entry
    top_p = k1_entry
    top_k = top_k_entry

    #bm25_parameters
    k1 = k1_entry
    b = b_entry
else:
    #default model parameters
    min_length = 50
    max_length = 150
    temperature = 0.2
    top_p = 0.15
    top_k = 50

    #default bm25_parameters
    k1 = 1.2
    b = 0.75

loader = PyPDFLoader("CS441 Textbook.pdf")

docs = loader.load()

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

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


#bm25_corpus = BM25_algorithm(doc_content, k=k1, b=b)



def summarize(input_query):

    bm25_corpus = BM25_algorithm(doc_content, k=1.2, b=0.75)

    bm25_scores = bm25_corpus.calculate_scores(input_query)

    ind = np.argpartition(bm25_scores, -3)[-3:]
    ind = ind.tolist()

    relevant_passages = ' ,'.join(np.array(doc_content)[ind])
    #bm25_context = str([i.page_content for i in relevant_passages])
    #metadata = np.array(metadata)[ind]
    metadata = 'placeholder'
    similarity = np.array(bm25_scores)[ind]


    input_text = f"given the context: {relevant_passages}, answer the question: {input_query}."

    #print(bm25_context)

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    outputs = model.generate(input_ids,
                                do_sample=True,
                                min_length = 50,
                                max_length = 150,
                                temperature = 0.2,
                                top_p = 0.15,
                                top_k = 50
                                
                                )
    res = tokenizer.decode(outputs[0])

        
    return res, similarity, metadata

if query:
    response, similarity, metadata = summarize(query)
    st.write(response)
    st.write('BM25 Score: ' , str(similarity))
    #st.write(metadata)



        
    


    

    