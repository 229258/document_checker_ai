# document_checker_ai

Please use Python 3.9

### Important after cloning the repository:

Find and extract archive ```dokumenciki5/variables/variables.zip``` (splitted to 2 files) and make sure that extracted file (```variables.data-00000-of-00001```) is the same folder as zip archive.

This is because github doesn't accept files bigger than 100MB.

### Before trying to run locally:
1. After installing Python libraries from ```requirements.txt``` additionally you need to install manually:
```
python -m pip install --upgrade Pillow
```
```
python -m pip install tensorflow
```

2. Make sure you have MongoDB database on Atalssian cluster. Modify you db address if needed. Currently its hardcoded in files `main.py` - variable `MONGODB_URI`.

### Running locally
Type in console (make sure is `uvicorn` in your Python libraries)
```
python -m uvicorn main:app --reload
```
Server should start on port 8000.
