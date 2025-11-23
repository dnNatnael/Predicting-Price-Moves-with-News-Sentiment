"""
Main EDA Execution Script
Runs comprehensive exploratory data analysis on the Financial News dataset
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
from data_loader import DataLoader
from eda_analyzer import EDAAnalyzer
from topic_modeling import TopicModeler
from publisher_analyzer import PublisherAnalyzer
import warnings
warnings.filterwarnings('ignore')


def create_output_directory(output_dir: str = "output"):
    """Create output directory if it doesn't exist"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir, "figures").mkdir(parents=True, exist_ok=True)
    Path(output_dir, "data").mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    """Main execution function"""
    print("=" * 80)
    print("FINANCIAL NEWS AND STOCK PRICE INTEGRATION DATASET - EDA")
    print("Nova Financial Solutions")
    print("=" * 80)
    print()
    
    # Configuration
    DATA_PATH = "data/financial_news.csv"  # Update this path to your dataset
    OUTPUT_DIR = create_output_directory("output")
    
    # Check if data file exists
    if not Path(DATA_PATH).exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Please ensure your dataset is located at the specified path.")
        print("Expected columns: headline, url, publisher, date, stock")
        return
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    print("-" * 80)
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    df = loader.preprocess_data()
    print(f"Loaded {len(df):,} articles")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    # Step 2: Descriptive Statistics and Basic EDA
    print("Step 2: Computing descriptive statistics...")
    print("-" * 80)
    eda = EDAAnalyzer(df, OUTPUT_DIR)
    stats = eda.compute_descriptive_stats()
    print(f"Total Articles: {stats['total_articles']:,}")
    print(f"Unique Publishers: {stats['unique_publishers']}")
    print(f"Unique Stocks: {stats['unique_stocks']}")
    print(f"Mean Headline Length: {stats['headline_length']['mean']:.2f} characters")
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    eda.plot_headline_length_distribution()
    print("  ✓ Headline length distribution saved")
    
    top_publishers = eda.analyze_top_publishers(10)
    eda.plot_top_publishers(10)
    print("  ✓ Top publishers analysis saved")
    
    freq_stats = eda.analyze_publication_frequency()
    eda.plot_publication_frequency()
    print("  ✓ Publication frequency analysis saved")
    
    spikes = eda.detect_news_spikes(threshold_std=2.0)
    eda.plot_news_spikes(threshold_std=2.0)
    print(f"  ✓ News spikes analysis saved ({len(spikes)} spikes detected)")
    print()
    
    # Step 3: Topic Modeling
    print("Step 3: Topic Modeling...")
    print("-" * 80)
    topic_modeler = TopicModeler(df, OUTPUT_DIR)
    
    # Extract frequent keywords
    keywords_df = topic_modeler.extract_frequent_keywords(50)
    topic_modeler.plot_frequent_keywords(30)
    print("  ✓ Frequent keywords extracted and visualized")
    
    # Train LDA
    print("Training LDA model (this may take a few minutes)...")
    lda_model = topic_modeler.train_lda(num_topics=10, passes=10)
    lda_topics = topic_modeler.get_lda_topics(num_words=10)
    topic_modeler.plot_lda_topics(num_words=10)
    print("  ✓ LDA model trained and topics extracted")
    
    # Create LDA visualization
    try:
        topic_modeler.create_lda_visualization()
        print("  ✓ Interactive LDA visualization created")
    except Exception as e:
        print(f"  ⚠ LDA visualization creation failed: {e}")
    
    # Train BERTopic (optional, may take longer)
    print("\nTraining BERTopic model (this may take several minutes)...")
    try:
        bertopic_model = topic_modeler.train_bertopic(min_topic_size=10)
        if bertopic_model:
            bertopic_topics = topic_modeler.get_bertopic_topics()
            topic_modeler.plot_bertopic_topics()
            print("  ✓ BERTopic model trained")
    except Exception as e:
        print(f"  ⚠ BERTopic training failed: {e}")
    
    # Assign topics to articles
    df_with_topics = topic_modeler.assign_topics_to_articles(method='lda')
    print()
    
    # Step 4: Publisher Analysis
    print("Step 4: Publisher Analysis...")
    print("-" * 80)
    publisher_analyzer = PublisherAnalyzer(df, OUTPUT_DIR)
    
    rankings = publisher_analyzer.rank_publishers()
    print(f"  ✓ Ranked {len(rankings)} publishers")
    
    domains = publisher_analyzer.extract_publisher_domains()
    if len(domains) > 0:
        print(f"  ✓ Extracted {len(domains)} publisher domains")
    
    styles = publisher_analyzer.analyze_reporting_styles(top_n=10)
    print(f"  ✓ Analyzed reporting styles for top 10 publishers")
    
    # Topic preferences
    topic_categories = topic_modeler.identify_topic_categories()
    topic_prefs = publisher_analyzer.identify_topic_preferences(topic_categories, top_n=10)
    print(f"  ✓ Identified topic preferences by publisher")
    
    # Generate publisher visualizations
    publisher_analyzer.plot_publisher_analysis(top_n=10)
    publisher_analyzer.plot_topic_preferences(topic_categories, top_n=10)
    print("  ✓ Publisher analysis visualizations saved")
    print()
    
    # Step 5: Save results
    print("Step 5: Saving results...")
    print("-" * 80)
    
    # Save summary report
    report = eda.generate_summary_report()
    with open(f"{OUTPUT_DIR}/eda_summary_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)
    print("  ✓ Summary report saved")
    
    # Save descriptive statistics
    stats_df = pd.DataFrame([stats['descriptive']])
    stats_df.to_csv(f"{OUTPUT_DIR}/data/descriptive_statistics.csv", index=False)
    print("  ✓ Descriptive statistics saved")
    
    # Save top publishers
    top_publishers.to_csv(f"{OUTPUT_DIR}/data/top_publishers.csv", index=False)
    print("  ✓ Top publishers data saved")
    
    # Save keywords
    keywords_df.to_csv(f"{OUTPUT_DIR}/data/frequent_keywords.csv", index=False)
    print("  ✓ Frequent keywords saved")
    
    # Save LDA topics
    lda_topics_df = pd.DataFrame([
        {'topic': k, 'words': ', '.join(v['top_words'])} 
        for k, v in lda_topics.items()
    ])
    lda_topics_df.to_csv(f"{OUTPUT_DIR}/data/lda_topics.csv", index=False)
    print("  ✓ LDA topics saved")
    
    # Save articles with topic assignments
    df_with_topics.to_csv(f"{OUTPUT_DIR}/data/articles_with_topics.csv", index=False)
    print("  ✓ Articles with topic assignments saved")
    
    # Save publisher rankings
    rankings.to_csv(f"{OUTPUT_DIR}/data/publisher_rankings.csv", index=False)
    print("  ✓ Publisher rankings saved")
    
    # Save topic preferences
    topic_prefs.to_csv(f"{OUTPUT_DIR}/data/publisher_topic_preferences.csv", index=False)
    print("  ✓ Publisher topic preferences saved")
    
    # Save news spikes
    spikes.to_csv(f"{OUTPUT_DIR}/data/news_spikes.csv", index=False)
    print("  ✓ News spikes data saved")
    print()
    
    # Step 6: Recommendations
    print("Step 6: Recommendations for Next Steps")
    print("=" * 80)
    print("""
    RECOMMENDATIONS FOR SENTIMENT ANALYSIS AND MODEL PREPARATION:
    
    1. SENTIMENT ANALYSIS PREPARATION:
       - Use the extracted topics to create topic-based sentiment features
       - Consider publisher-specific sentiment patterns
       - Analyze sentiment trends around news spikes
       - Create time-based sentiment features (hour, day of week)
    
    2. FEATURE ENGINEERING:
       - Headline length and word count (already extracted)
       - Topic labels from LDA/BERTopic
       - Publisher credibility/weighting factors
       - Temporal features (time since market open, day of week)
       - Stock-specific news frequency
    
    3. CORRELATION ANALYSIS:
       - Correlate sentiment scores with stock price movements
       - Analyze lag effects (news impact on next day/week prices)
       - Study publisher influence on market reactions
       - Identify which topics drive price movements
    
    4. MODEL PREPARATION:
       - Create train/validation/test splits respecting temporal order
       - Handle class imbalance if predicting price direction
       - Consider multi-class classification (up/down/neutral)
       - Feature selection based on correlation analysis
    
    5. SENTIMENT SCORING APPROACHES:
       - Use VADER (financial lexicon-aware)
       - Fine-tune BERT/RoBERTa on financial news
       - Create custom financial sentiment dictionary
       - Combine multiple sentiment scores
    
    6. VALIDATION STRATEGY:
       - Time-series cross-validation
       - Walk-forward analysis
       - Out-of-sample testing on recent data
    """)
    
    print("=" * 80)
    print("EDA COMPLETE!")
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()



