# CS410 CourseProject - Hannah Benig

# Prerequisites
- Mac OS, Linux, or WSL (on a Windows PC)
    - WSL installation: https://learn.microsoft.com/en-us/windows/wsl/install
    - type 'wsl' in terminal to activate
    - WSL can be quite slow; another option is to remote into a Linux machine via FastX from UIUC EWS: https://answers.uillinois.edu/illinois.engineering/page.php?id=81693
- Python

# Setup and Run Streamlit App

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

