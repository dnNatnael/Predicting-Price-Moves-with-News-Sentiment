# Package Installation Guide

## Installation Location
**Project Folder**: `C:\Users\HomePC\Desktop\Second\Predicting-Price-Moves-with-News-Sentiment`

## Installation Steps

### 1. Activate Virtual Environment
```powershell
cd C:\Users\HomePC\Desktop\Second\Predicting-Price-Moves-with-News-Sentiment
.\venv\Scripts\Activate.ps1
```

### 2. Install All Packages
```powershell
pip install -r requirements.txt
```

### 3. Download spaCy English Model
```powershell
python -m spacy download en_core_web_sm
```

## Packages to Install

All packages listed in `requirements.txt` will be installed. Key packages include:

- **pandas, numpy** - Data processing
- **matplotlib, seaborn, plotly** - Visualization
- **nltk, spacy, textblob** - NLP
- **gensim, bertopic** - Topic modeling
- **scikit-learn** - Machine learning utilities
- **jupyter, notebook** - Jupyter notebook support

## Note
The packages should be installed in the virtual environment located at:
`C:\Users\HomePC\Desktop\Second\Predicting-Price-Moves-with-News-Sentiment\venv`



