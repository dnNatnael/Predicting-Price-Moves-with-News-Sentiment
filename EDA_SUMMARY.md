# EDA Solution Summary

## Overview
A comprehensive Exploratory Data Analysis solution has been created for the Financial News and Stock Price Integration Dataset (FNSPID).

## Package Installation Information

### Installation Location
**Project Root Directory**: 
```
C:\Users\HomePC\Desktop\Second\Predicting-Price-Moves-with-News-Sentiment
```

### Installation Steps
1. Activate virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Install all packages:
   ```powershell
   pip install -r requirements.txt
   ```

3. Download spaCy English model:
   ```powershell
   python -m spacy download en_core_web_sm
   ```

### Required Packages (from requirements.txt)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **NLP**: nltk, spacy, textblob
- **Topic Modeling**: gensim, bertopic, sentence-transformers, umap-learn, hdbscan, pyLDAvis
- **ML**: scikit-learn
- **Utilities**: wordcloud, tqdm, python-dateutil, jupyter, notebook

## Solution Components

### 1. Source Modules (`src/`)

#### `data_loader.py`
- Loads CSV dataset
- Preprocesses data (date parsing, feature extraction)
- Validates required columns
- Extracts temporal features

#### `eda_analyzer.py`
- Computes descriptive statistics
- Analyzes headline length distributions
- Identifies top publishers
- Analyzes publication frequency (daily, monthly, yearly, hourly)
- Detects news activity spikes
- Generates comprehensive visualizations
- Creates summary reports

#### `topic_modeling.py`
- Extracts frequent keywords and phrases
- Implements LDA topic modeling
- Implements BERTopic (optional, advanced)
- Preprocesses text (tokenization, lemmatization, stopword removal)
- Creates topic visualizations
- Assigns topics to articles
- Identifies financial topic categories (earnings, mergers, FDA approvals, etc.)

#### `publisher_analyzer.py`
- Ranks publishers by activity
- Extracts publisher domains
- Analyzes reporting styles
- Identifies topic preferences by publisher
- Creates publisher-specific visualizations

### 2. Execution Scripts

#### `scripts/run_eda.py`
Main execution script that:
- Loads and preprocesses data
- Runs all EDA analyses
- Generates all visualizations
- Saves all results to CSV files
- Creates summary report
- Provides recommendations for next steps

### 3. Jupyter Notebook

#### `notebooks/EDA_Analysis.ipynb`
Interactive notebook for step-by-step analysis (can be created manually from the script)

## Output Structure

All outputs are saved to the `output/` directory:

```
output/
├── figures/
│   ├── headline_length_distribution.png
│   ├── top_publishers.png
│   ├── publication_frequency.png
│   ├── news_spikes.png
│   ├── frequent_keywords.png
│   ├── lda_topics.png
│   ├── publisher_analysis.png
│   └── publisher_topic_preferences.png
├── data/
│   ├── descriptive_statistics.csv
│   ├── top_publishers.csv
│   ├── frequent_keywords.csv
│   ├── lda_topics.csv
│   ├── articles_with_topics.csv
│   ├── publisher_rankings.csv
│   ├── publisher_topic_preferences.csv
│   └── news_spikes.csv
├── lda_visualization.html
├── bertopic_topics.html (if available)
└── eda_summary_report.txt
```

## Analysis Features

### 1. Descriptive Statistics ✓
- Headline length distribution (min, max, mean, median, std, quartiles)
- Word count statistics
- Dataset overview (total articles, unique publishers, unique stocks)
- Date range analysis

### 2. Publisher Analysis ✓
- Top 10 most active publishers
- Publisher rankings with statistics
- Reporting style differences
- Domain extraction (if email-like values)
- Topic preferences by publisher

### 3. Publication Frequency Analysis ✓
- Daily, weekly, monthly, yearly patterns
- Hour-of-day distribution (pre-market, market hours, after-hours)
- Time-series analysis
- News activity spike detection (with threshold)
- Major market event identification suggestions

### 4. Text Analysis & Topic Modeling ✓
- Frequent keywords and phrases extraction
- LDA topic modeling (10 topics, configurable)
- BERTopic modeling (optional, advanced)
- Topic assignment to articles
- Topic category identification:
  - Earnings
  - Mergers & Acquisitions
  - FDA Approvals
  - Price Targets
  - Product Launches
  - Partnerships
  - Regulations
  - Market Movements

### 5. Visualizations ✓
- Headline length distribution (histogram, box plots)
- Top publishers (bar charts, pie charts)
- Publication frequency (multiple time-series plots)
- News spikes visualization
- Frequent keywords (bar charts)
- LDA topics visualization
- Publisher analysis (multiple charts)
- Topic preferences heatmap

## Usage

### Option 1: Run Python Script
```powershell
python scripts/run_eda.py
```
Make sure your dataset is at: `data/financial_news.csv`

### Option 2: Use Jupyter Notebook
1. Start Jupyter: `jupyter notebook`
2. Open `notebooks/EDA_Analysis.ipynb`
3. Update data path and run cells

## Recommendations for Next Steps

The script provides detailed recommendations for:
1. **Sentiment Analysis Preparation**
   - Topic-based sentiment features
   - Publisher-specific sentiment patterns
   - Time-based sentiment features

2. **Feature Engineering**
   - Headline length/word count
   - Topic labels
   - Publisher credibility factors
   - Temporal features

3. **Correlation Analysis**
   - Sentiment-price correlations
   - Lag effects analysis
   - Publisher influence on market reactions

4. **Model Preparation**
   - Temporal train/test splits
   - Class imbalance handling
   - Feature selection

5. **Sentiment Scoring Approaches**
   - VADER (financial-aware)
   - Fine-tuned BERT/RoBERTa
   - Custom financial dictionary

6. **Validation Strategy**
   - Time-series cross-validation
   - Walk-forward analysis
   - Out-of-sample testing

## Notes

- All code is well-commented and follows best practices
- Results are reproducible (random seeds set)
- Visualizations are high-resolution (300 DPI)
- BERTopic is optional and may take longer but provides better quality
- The solution handles missing data gracefully
- All outputs are structured and well-organized

## Dataset Requirements

Your dataset CSV should have these columns:
- `headline`: Financial news headline
- `url`: Link to full article
- `publisher`: Author or news source
- `date`: Publication date and time (UTC-4 timezone)
- `stock`: Stock ticker symbol (e.g., AAPL)




