#project imports
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import T5Tokenizer, T5ForConditionalGeneration
from bm25 import BM25_algorithm
from bs4 import BeautifulSoup

import base64
import numpy as np


from huggingface_hub import login

from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv('HF_APIKEY'))

import warnings
warnings.filterwarnings("ignore")

#function to load the cs441 textbook and chunk it into sections
#returns 2 lists: metadata containing the page number and pdf name and doc_content containing the textbook chunks
def load_pdf():
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
    return metadata, doc_content

#if the metadata and doc_content arrays have not been loaded, load them
if ('metadata' not in st.session_state) or ('doc_content' not in st.session_state):
    m, dc = load_pdf()
    st.session_state.metadata = m
    st.session_state.doc_content = dc

#if the flan-t5 tokenizer and model have not been loaded, load them
#you can swap out this model for any huggingface model just ensure you use the right transformers
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

if 'model' not in st.session_state:
    st.session_state.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
                                                              

tokenizer = st.session_state.tokenizer
model= st.session_state.model


st.title('CS441 Assistant')
st.caption('Hannah Benig (hhlim2) CS410 Final Project')
col1, col2 = st.columns(2)


with col1:
    query = st.text_area('Applied Machine Learning Question')


#from https://discuss.streamlit.io/t/go-to-specific-page-in-pdf-after-loading-it/39546
#loads the pdf display for specified page
def show_pdf(new_page_num):
    with open("CS441 Textbook.pdf","rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={new_page_num}" width="500" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    

#sets default model parameters
if 'min_length' not in st.session_state:
    st.session_state.min_length = 50
if 'max_length' not in st.session_state:
    st.session_state.max_length = 150
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.2
if 'top_p' not in st.session_state:
    st.session_state.top_p = 0.15
if 'top_k' not in st.session_state:
    st.session_state.top_k = 50

#sets  default bm25_parameters
if 'k1' not in st.session_state:
    st.session_state.k1 = 1.2
if 'b' not in st.session_state:
    st.session_state.b = 0.75


#initalizes the 441 textbook as the corpus
if 'bm25_corpus' not in st.session_state:
    st.session_state.bm25_corpus = BM25_algorithm(st.session_state.doc_content, k=st.session_state.k1, b=st.session_state.b)


#if parameters are changed on the sidebar, the global variables are updated
def set_params(min_length_entry, max_length_entry, temperature_entry, top_p_entry, top_k_entry, k1_entry, b_entry):
    #Model parameters
    st.session_state.min_length = min_length_entry
    st.session_state.max_length = max_length_entry
    st.session_state.temperature = temperature_entry
    st.session_state.top_p = top_p_entry
    st.session_state.top_k = top_k_entry

    #bm25 parameters
    st.session_state.k1 = k1_entry
    st.session_state.b = b_entry


#sets up the sidebar to change model and bm25 parameters
with st.sidebar:
    with st.form('params_form'):
        min_length_entry = st.number_input('minimum length of output', min_value=1, value = st.session_state.min_length)
        max_length_entry = st.number_input('maximum length of output', min_value= min_length_entry + 1, value = st.session_state.max_length)
        temperature_entry = st.slider('temperature', min_value=0.0, max_value=1.0, value=st.session_state.temperature, step=0.01)
        top_p_entry = st.slider('top_p', min_value=0.0, max_value=1.0, value=st.session_state.top_p, step=0.01)
        top_k_entry = st.number_input('top_k', min_value=0, max_value=100, value=st.session_state.top_k)
        k1_entry = st.number_input('bm25 k1', min_value=0.0, max_value= 100.0, value=st.session_state.k1)
        b_entry = st.slider('bm25 b', min_value=0.0, max_value= 1.0, value=st.session_state.b, step=0.01)

        submit = st.form_submit_button(label='submit')

        if submit:
            set_params(min_length_entry, max_length_entry, temperature_entry, top_p_entry, top_k_entry, k1_entry, b_entry)
            st.session_state.bm25_corpus = BM25_algorithm(st.session_state.doc_content, k=k1_entry, b=b_entry)
    

#sets up the answering function
def summarize(input_query):
   
    #calculates the bm25 scores for the textbook chunks based on their relevance to the input query
    bm25_scores = st.session_state.bm25_corpus.calculate_scores(input_query)

    #gets the top 3 documents 
    #(this can be changed if you want to use more than 3 relevant chunks to construct the answer)
    ind = np.argpartition(bm25_scores, -3)[-3:]
    ind = ind.tolist()

    #gets the most relevant passages and page numbers based on their bm25 scores
    relevant_passages = ' ,'.join(np.array(st.session_state.doc_content)[ind])
    page_meta = np.array(st.session_state.metadata)[ind]
    page_num = [page_meta[i]['page'] for i in range(len(page_meta))]

    #bm25 scores
    similarity = np.average(np.array(bm25_scores)[ind])
    
    #model prompt
    #input_text = f"Use the following context: {relevant_passages} to answer the question: {input_query}."
    input_text = f"{relevant_passages} \n\n {input_query}"


    #tokenizes the model prompt to be taken in by the model
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    #passes the input_ids and parameters to the model and generates a tokenized answer
    outputs = model.generate(input_ids,
                                do_sample=True,
                                min_length = st.session_state.min_length,
                                max_length = st.session_state.max_length,
                                temperature = st.session_state.temperature,
                                top_p = st.session_state.top_p,
                                top_k = st.session_state.top_k
                                
                                )
    #tokenized model output is decoded back to natural english
    res = tokenizer.decode(outputs[0])

        
    return res, similarity, page_num

#displays the answer and relevant textbook page
page_num = None
if query:
    with col1:
        response, similarity, page_num = summarize(query)
        response = BeautifulSoup(response, "html.parser").text
        st.write(response)
        st.write('BM25 Score: ' , str(similarity))
        st.write('Page Numbers: ' , str(page_num))
with col2:
    if page_num is not None:
        show_pdf(page_num[0])
    else:
        show_pdf(0)





        
    


    

    
