# CS410 CourseProject - Hannah Benig

Instructions on how to install and run the web app locally, along with example questions that can be used for testing. Two methods of running the application are provided. 

1. Google Colab:
- Will run slowly but does not depend on personal computer specs
2. Locally:
- Recommended specs: 32GB RAM,  i7-12700H Processor
    - A computer with worse specs might run very slowly or not at all

# Setup and Run Streamlit App in Google Colab
The Colab notebook contains the same code from the main.py and bm25.py files but allows you to utilize the T4 GPU runtime.

![alt text](https://github.com/hhlim2/CourseProject-HB/blob/main/photos/colab_1.png?raw=true)
*Figure 1: Google Colab Notebook Setup*

Colab Instructions
1. Download the CS441 Textbook.pdf file from https://github.com/hhlim2/CourseProject-HB/blob/main/CS441%20Textbook.pdf 
2. Open Google Colab Link in **Firefox** (make a copy and run in your personal Google account): https://colab.research.google.com/drive/1lxnP1LTVJ0fClyiLbsI97CLCa1EieNg4?usp=sharing
3. Upload the CS441 Textbook.pdf file into the Google Colab Notebook Files folder (numbers 1 & 2 in Figure 1)
4. Connect to the T4 GPU runtime in the Google Colab Notebook (number 3 in Figure 1)
5. Run all cells

![alt text](https://github.com/hhlim2/CourseProject-HB/blob/main/photos/colab_2.png?raw=true)
*Figure 2: IP Address and Localtunnel Link*

6. Copy the IP Address contained in the output of cell 3 (number 1 in Figure 2)
7. Click on the link in the output of cell 4 (number 2 in Figure 2) 

![alt text](https://github.com/hhlim2/CourseProject-HB/blob/main/photos/colab_3.png?raw=true)
*Figure 3: Example Localtunnel Webpage*

8. The link from the previous step will bring you to a page similar to the one in Figure 3. Enter the IP Address you copied in step 6 in the “Endpoint IP” box and click Submit.
9. You should be brought to the Streamlit App. 

# Setup and Run Streamlit App Locally
Prerequisites 
- Mac OS, Linux, or WSL (on a Windows PC)
    - WSL installation: https://learn.microsoft.com/en-us/windows/wsl/install
    - type 'wsl' in terminal to activate
- Python

1. Clone this repository
```
git clone https://github.com/hhlim2/CourseProject-HB.git
```

2. Create a virtual environment within the repository directory
```
cd CourseProject-HB
python3 -m venv env
```

3. Activate the virtual environment
```
source env/bin/activate
```

4. Install all dependencies
```
pip install -r requrements.txt
```

5. Run the application
```
streamlit run main.py
```
Open the localhost link in Firefox for full features (PDF display does not work in Chrome)

# Usage
1. Input your question regarding topics covered in the CS441 textbook and receive an answer. The PDF viewer will bring you to the page/section where the answer can be found.
2. You can change the hyperparameters in the left window to see how tweaking parameters changes the answer. The default parameters have been set to values where the best response is generated.

Example Questions
1. What is agglomerative clustering?
2. What is the difference between agglomerative and divisive clustering?
3. Explain how object detection works.

