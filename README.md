# CS410 CourseProject - Hannah Benig


# Setup and Run Streamlit App
1. Create venv
```
python3 -m venv 410env
```

3. Activate the virtual environment
```
mac: source 410env/bin/activate
windows: & "./410env/Scripts/Activate.ps1"
```

5. Install all dependencies
```
pip install -r requrements.txt
```

7. Run the application
```
streamlit run final_project.py
```

# Usage
1. Input your question regarding topics covered in the cs441 textbook and receive an answer. The PDF viewer will bring you to the page/section where the answer can be found.
2. You can change the hyperparameters in the left window to see how tweaking parameters changes the answer. The default parameters have been set to values where the best response is generated.
