# Predicting Price Moves with News Sentiment

Financial News and Stock Price Integration Dataset - Exploratory Data Analysis

## Project Overview

This project performs comprehensive Exploratory Data Analysis (EDA) on the Financial News and Stock Price Integration Dataset (FNSPID) for Nova Financial Solutions.

## Dataset Structure

The dataset should contain the following columns:
- `headline`: The financial news headline
- `url`: Link to the full article
- `publisher`: Author or news source
- `date`: Publication date and time (UTC-4 timezone)
- `stock`: Stock ticker symbol (e.g., AAPL)

## Installation Instructions

### 1. Install Packages

**Location**: Install packages in the project root directory:
```
C:\Users\HomePC\Desktop\Second\Predicting-Price-Moves-with-News-Sentiment
```

**Steps**:
1. Activate your virtual environment (if using venv):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install all required packages:
   ```powershell
   pip install -r requirements.txt
   ```

3. Download spaCy English model:
   ```powershell
   python -m spacy download en_core_web_sm
   ```

### 2. Required Packages

All packages are listed in `requirements.txt`. Key packages include:
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **NLP**: nltk, spacy, textblob
- **Topic Modeling**: gensim, bertopic, sentence-transformers
- **Machine Learning**: scikit-learn
- **Utilities**: tqdm, python-dateutil

## Usage

### Option 1: Run Python Script

1. Place your dataset CSV file in the `data/` folder as `financial_news.csv`
2. Run the main EDA script:
   ```powershell
   python scripts/run_eda.py
   ```

### Option 2: Use Jupyter Notebook

1. Start Jupyter Notebook:
   ```powershell
   jupyter notebook
   ```
2. Open `notebooks/EDA_Analysis.ipynb`
3. Update the data path in the first cell
4. Run all cells

## Output Structure

All outputs are saved to the `output/` directory:

```
output/
├── figures/              # All visualization images
│   ├── headline_length_distribution.png
│   ├── top_publishers.png
│   ├── publication_frequency.png
│   ├── news_spikes.png
│   ├── frequent_keywords.png
│   ├── lda_topics.png
│   ├── publisher_analysis.png
│   └── publisher_topic_preferences.png
├── data/                 # Extracted features and statistics
│   ├── descriptive_statistics.csv
│   ├── top_publishers.csv
│   ├── frequent_keywords.csv
│   ├── lda_topics.csv
│   ├── articles_with_topics.csv
│   ├── publisher_rankings.csv
│   ├── publisher_topic_preferences.csv
│   └── news_spikes.csv
├── lda_visualization.html    # Interactive LDA visualization
├── bertopic_topics.html      # BERTopic visualization (if available)
└── eda_summary_report.txt    # Text summary report
```

## Analysis Components

### 1. Descriptive Statistics
- Headline length distribution (min, max, mean, median)
- Word count statistics
- Dataset overview (total articles, unique publishers, unique stocks)

### 2. Publisher Analysis
- Top 10 most active publishers
- Publisher rankings by article count
- Reporting style differences
- Domain extraction (if email-like values)

### 3. Publication Frequency
- Daily, weekly, monthly, yearly patterns
- Hour-of-day distribution
- Time-series analysis
- News activity spike detection

### 4. Topic Modeling
- Frequent keywords and phrases extraction
- LDA topic modeling (10 topics)
- BERTopic modeling (optional)
- Topic assignment to articles
- Topic category identification (earnings, mergers, FDA approvals, etc.)

### 5. Publisher-Specific Analysis
- Topic preferences by publisher
- Reporting style analysis
- Stock coverage patterns

## Next Steps for Sentiment Analysis

1. **Sentiment Scoring**: Implement VADER or fine-tune BERT for financial sentiment
2. **Feature Engineering**: Create time-based and topic-based features
3. **Correlation Analysis**: Link sentiment scores to price movements
4. **Model Development**: Build predictive models for price direction
5. **Validation**: Use time-series cross-validation for robust evaluation

## Project Structure

```
Predicting-Price-Moves-with-News-Sentiment/
├── data/                  # Dataset folder (place your CSV here)
├── src/                   # Source code modules
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── eda_analyzer.py    # EDA analysis and visualizations
│   ├── topic_modeling.py  # Topic modeling (LDA, BERTopic)
│   └── publisher_analyzer.py  # Publisher analysis
├── scripts/               # Execution scripts
│   └── run_eda.py         # Main EDA execution script
├── notebooks/             # Jupyter notebooks
│   └── EDA_Analysis.ipynb # Interactive EDA notebook
├── output/                # Generated outputs (created automatically)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Notes

- Ensure your dataset CSV file is properly formatted with the required columns
- The analysis may take several minutes depending on dataset size
- BERTopic training is optional and may take longer but provides better topic quality
- All visualizations are saved as high-resolution PNG files (300 DPI)

## Support

For questions or issues, please refer to the code comments or contact the development team.
