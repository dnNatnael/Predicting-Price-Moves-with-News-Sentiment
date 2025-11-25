# Installation Instructions

## Package Installation

### Location
Install all packages in the project root directory:
```
C:\Users\HomePC\Desktop\Second\Predicting-Price-Moves-with-News-Sentiment
```

### Steps

1. **Activate Virtual Environment** (if using venv):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Required Packages**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Download spaCy English Model**:
   ```powershell
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK Data** (will be downloaded automatically on first run, but you can pre-download):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Required Packages

All packages are listed in `requirements.txt`. Here's what will be installed:

### Core Data Processing
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing

### Visualization
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualization
- `plotly` - Interactive visualizations

### Natural Language Processing
- `nltk` - Natural Language Toolkit
- `spacy` - Advanced NLP library
- `textblob` - Text processing library

### Topic Modeling
- `gensim` - Topic modeling (LDA)
- `bertopic` - BERT-based topic modeling
- `sentence-transformers` - Sentence embeddings for BERTopic
- `umap-learn` - Dimensionality reduction for BERTopic
- `hdbscan` - Clustering for BERTopic
- `pyLDAvis` - Interactive LDA visualization

### Machine Learning
- `scikit-learn` - Machine learning utilities

### Utilities
- `wordcloud` - Word cloud generation
- `tqdm` - Progress bars
- `python-dateutil` - Date parsing
- `jupyter` - Jupyter notebook support
- `notebook` - Jupyter notebook server

## Verification

After installation, verify by running:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import gensim
print("All packages installed successfully!")
```

## Troubleshooting

1. **If spaCy model download fails**: Try downloading manually:
   ```powershell
   python -m spacy download en_core_web_sm
   ```

2. **If BERTopic installation fails**: It requires PyTorch. Install PyTorch first if needed.

3. **Memory issues with BERTopic**: BERTopic is optional. The analysis will work with just LDA if BERTopic fails.




