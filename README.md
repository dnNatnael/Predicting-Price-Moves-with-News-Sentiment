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

---

## Experiment Tracking and Outputs

### Understanding Output Directory Structure

The `output/` directory contains all generated analysis results. Here's what each subdirectory and file contains:

```
output/
├── figures/                      # Static visualization images (PNG)
├── data/                         # Processed data files (CSV)
├── technical_analysis/          # Technical analysis outputs
│   ├── plots/                   # TA-Lib indicator charts
│   ├── summary.json             # Key metrics (volatility, Sharpe ratio)
│   ├── correlation_matrix.csv   # Price correlations
│   └── dataframe_snapshot.csv   # Last 10 rows of processed data
├── lda_visualization.html        # Interactive LDA topic visualization
├── bertopic_topics.html          # Interactive BERTopic visualization
├── bertopic_barchart.html        # Topic frequency bar chart
├── eda_summary_report.txt        # Text summary of EDA findings
└── data/
    ├── articles_with_topics.csv # News with topic labels
    ├── descriptive_statistics.csv # Summary statistics
    ├── top_publishers.csv       # Publisher rankings
    ├── frequent_keywords.csv     # Keyword frequencies
    ├── lda_topics.csv           # Topic-word mappings
    ├── publisher_rankings.csv    # Full publisher rankings
    ├── publisher_topic_preferences.csv  # Topic distribution by publisher
    └── news_spikes.csv          # Detected news spikes
```

### Interpreting Key Outputs

1. **Technical Analysis Summary** (`output/technical_analysis/summary.json`):
   - `annualized_volatility`: 20-day rolling volatility annualized (higher = more volatile)
   - `sharpe_ratio`: Risk-adjusted return (higher = better, >1 is good, >2 is excellent)
   - `last_cumulative_return`: Total return over the analysis period

2. **Correlation Analysis**:
   - Pearson correlation ranges from -1 (perfect negative) to +1 (perfect positive)
   - p-value < 0.05 indicates statistical significance
   - 95% CI shows the uncertainty range

3. **Topic Models**:
   - LDA visualization shows topic clusters and word distributions
   - BERTopic provides hierarchical topic structure
   - Use coherence score to evaluate topic quality (higher = better, typically 0.3-0.7)

### Running Experiments

```bash
# EDA with default settings
python scripts/run_eda.py

# Technical analysis with custom ticker
python scripts/technical_analysis.py --ticker GOOGL

# Technical analysis with custom date range
python scripts/technical_analysis.py --ticker AAPL --start-date 2022-01-01 --end-date 2023-12-31

# Technical analysis with custom indicators
python scripts/technical_analysis.py --ticker AAPL --sma-window 10 20 50 --rsi-window 21

# Correlation analysis with timezone handling
python scripts/news_sentiment_stock_correlation.py --ticker AAPL

# Correlation with lagged sentiment (previous day)
python scripts/news_sentiment_stock_correlation.py --ticker AAPL --use-lagged --lag 1

# Comprehensive correlation analysis
python scripts/news_sentiment_stock_correlation.py --ticker AAPL --comprehensive
```

---

## Troubleshooting

### Common Environment Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'talib'`

**Solution**:
```bash
# Install TA-Lib system dependencies first, then:
pip install TA-Lib
```

**Problem**: `ModuleNotFoundError: No module named 'bertopic'`

**Solution**:
```bash
pip install bertopic sentence-transformers
```

#### 2. Data Loading Issues

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/financial_news.csv'`

**Solution**:
- Ensure your CSV file is in the `data/` directory
- Update the path in the script or use `--data-path` flag

**Problem**: `KeyError: 'headline'` - Missing required column

**Solution**:
- Check that your CSV has all required columns: `headline`, `url`, `publisher`, `date`, `stock`
- Run with verbose output to see exact error

#### 3. Memory Issues with Large Datasets

**Problem**: `MemoryError` during topic modeling

**Solution**:
- Reduce number of topics: `train_lda(num_topics=5)`
- Sample the dataset: `df.sample(n=10000)`
- Use `min_topic_size` parameter in BERTopic

#### 4. Timezone Warnings

**Problem**: `RuntimeWarning: datetime.datetime ... is not timezone-aware`

**Solution**:
- The correlation analysis handles this automatically with `--timezone` flag
- Ensure news timestamps are properly localized

#### 5. NLTK Data Issues

**Problem**: `LookupError: 'punkt' not found`

**Solution**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Performance Tips

1. **Speed up EDA**: Use `--skip-bertopic` if not needed
2. **Parallel processing**: Gensim's LdaMulticore uses multiple cores
3. **Reduce plot resolution**: Change `dpi=300` to `dpi=100` in savefig calls
4. **Batch processing**: Process multiple tickers in a loop

### Getting Help

1. Check the output logs for detailed error messages
2. Run with `--verbose` flag for more information
3. Review the analysis summary in `output/*/summary.json`
4. Examine the interactive HTML visualizations for data insights
