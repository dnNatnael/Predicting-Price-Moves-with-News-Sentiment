"""
Publisher Analysis Module
Analyzes publisher patterns and reporting styles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import re
import warnings
warnings.filterwarnings('ignore')


class PublisherAnalyzer:
    """Analyze publisher patterns and reporting styles"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "output"):
        """
        Initialize the Publisher Analyzer
        
        Args:
            df: DataFrame with publisher data
            output_dir: Directory to save outputs
        """
        self.df = df
        self.output_dir = output_dir
        self.analysis_results = {}
    
    def rank_publishers(self) -> pd.DataFrame:
        """
        Rank publishers by number of articles
        
        Returns:
            DataFrame with publisher rankings
        """
        publisher_rankings = self.df.groupby('publisher').agg({
            'headline': 'count',
            'stock': 'nunique',
            'headline_length': ['mean', 'std'],
            'date': ['min', 'max']
        }).reset_index()
        
        publisher_rankings.columns = [
            'publisher', 'article_count', 'unique_stocks',
            'avg_headline_length', 'std_headline_length',
            'first_article', 'last_article'
        ]
        
        publisher_rankings = publisher_rankings.sort_values('article_count', ascending=False)
        publisher_rankings['rank'] = range(1, len(publisher_rankings) + 1)
        
        self.analysis_results['rankings'] = publisher_rankings
        return publisher_rankings
    
    def extract_publisher_domains(self) -> pd.DataFrame:
        """
        Extract domains from publisher field (if email-like)
        
        Returns:
            DataFrame with domain statistics
        """
        # Check if publisher field contains email-like values
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        
        domain_stats = []
        for publisher in self.df['publisher'].unique():
            if re.match(email_pattern, str(publisher)):
                domain = publisher.split('@')[1]
                count = len(self.df[self.df['publisher'] == publisher])
                domain_stats.append({
                    'publisher': publisher,
                    'domain': domain,
                    'article_count': count
                })
        
        if domain_stats:
            domain_df = pd.DataFrame(domain_stats)
            domain_summary = domain_df.groupby('domain')['article_count'].sum().sort_values(ascending=False)
            self.analysis_results['domains'] = domain_summary
            return domain_df
        else:
            # Use publisher_domain column if available
            if 'publisher_domain' in self.df.columns:
                domain_summary = self.df.groupby('publisher_domain')['headline'].count().sort_values(ascending=False)
                self.analysis_results['domains'] = domain_summary
                return pd.DataFrame({
                    'domain': domain_summary.index,
                    'article_count': domain_summary.values
                })
            return pd.DataFrame()
    
    def analyze_reporting_styles(self, top_n: int = 10) -> Dict:
        """
        Analyze differences in reporting styles across publishers
        
        Args:
            top_n: Number of top publishers to analyze
            
        Returns:
            Dictionary with style analysis
        """
        top_publishers = self.rank_publishers().head(top_n)['publisher'].tolist()
        
        style_analysis = {}
        
        for publisher in top_publishers:
            pub_df = self.df[self.df['publisher'] == publisher]
            
            # Analyze keywords
            keywords = []
            for headline in pub_df['headline']:
                keywords.extend(str(headline).lower().split())
            
            from collections import Counter
            top_keywords = Counter(keywords).most_common(10)
            
            style_analysis[publisher] = {
                'article_count': len(pub_df),
                'avg_headline_length': pub_df['headline_length'].mean(),
                'unique_stocks': pub_df['stock'].nunique(),
                'top_keywords': [kw[0] for kw in top_keywords],
                'keyword_frequencies': dict(top_keywords),
                'publication_frequency': pub_df.groupby('date_only')['headline'].count().mean()
            }
        
        self.analysis_results['styles'] = style_analysis
        return style_analysis
    
    def identify_topic_preferences(self, topic_keywords: Dict, top_n: int = 10) -> pd.DataFrame:
        """
        Identify which publishers focus on which topics
        
        Args:
            topic_keywords: Dictionary mapping topic names to keyword lists
            top_n: Number of top publishers to analyze
            
        Returns:
            DataFrame with topic preferences
        """
        top_publishers = self.rank_publishers().head(top_n)['publisher'].tolist()
        
        topic_preferences = []
        
        for publisher in top_publishers:
            pub_df = self.df[self.df['publisher'] == publisher]
            headlines = pub_df['headline'].str.lower().str.cat(sep=' ')
            
            publisher_topics = {'publisher': publisher, 'total_articles': len(pub_df)}
            
            for topic, keywords in topic_keywords.items():
                # Count occurrences of topic keywords
                count = sum(headlines.count(keyword) for keyword in keywords)
                publisher_topics[f'{topic}_mentions'] = count
                publisher_topics[f'{topic}_percentage'] = (count / len(pub_df)) * 100 if len(pub_df) > 0 else 0
            
            topic_preferences.append(publisher_topics)
        
        topic_df = pd.DataFrame(topic_preferences)
        self.analysis_results['topic_preferences'] = topic_df
        return topic_df
    
    def plot_publisher_analysis(self, top_n: int = 10, save: bool = True):
        """Create comprehensive publisher analysis visualizations"""
        rankings = self.rank_publishers().head(top_n)
        styles = self.analyze_reporting_styles(top_n)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Publisher rankings
        axes[0, 0].barh(range(len(rankings)), rankings['article_count'], 
                       color=sns.color_palette("husl", len(rankings)))
        axes[0, 0].set_yticks(range(len(rankings)))
        axes[0, 0].set_yticklabels(rankings['publisher'], fontsize=9)
        axes[0, 0].set_xlabel('Number of Articles')
        axes[0, 0].set_title(f'Top {top_n} Publishers by Article Count')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # 2. Average headline length by publisher
        avg_lengths = [styles[pub]['avg_headline_length'] for pub in rankings['publisher']]
        axes[0, 1].bar(range(len(rankings)), avg_lengths, 
                      color=sns.color_palette("coolwarm", len(rankings)))
        axes[0, 1].set_xticks(range(len(rankings)))
        axes[0, 1].set_xticklabels(rankings['publisher'], rotation=45, ha='right', fontsize=9)
        axes[0, 1].set_ylabel('Average Headline Length (characters)')
        axes[0, 1].set_title('Average Headline Length by Publisher')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Unique stocks covered
        unique_stocks = rankings['unique_stocks'].tolist()
        axes[1, 0].scatter(range(len(rankings)), unique_stocks, 
                          s=100, alpha=0.6, color='purple')
        axes[1, 0].set_xticks(range(len(rankings)))
        axes[1, 0].set_xticklabels(rankings['publisher'], rotation=45, ha='right', fontsize=9)
        axes[1, 0].set_ylabel('Number of Unique Stocks')
        axes[1, 0].set_title('Stock Coverage by Publisher')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Article count vs unique stocks
        axes[1, 1].scatter(rankings['article_count'], rankings['unique_stocks'], 
                          s=150, alpha=0.6, color='green')
        for idx, row in rankings.iterrows():
            axes[1, 1].annotate(row['publisher'], 
                              (row['article_count'], row['unique_stocks']),
                              fontsize=7, alpha=0.7)
        axes[1, 1].set_xlabel('Total Articles')
        axes[1, 1].set_ylabel('Unique Stocks Covered')
        axes[1, 1].set_title('Publisher Activity vs Coverage')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/publisher_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_topic_preferences(self, topic_keywords: Dict, top_n: int = 10, save: bool = True):
        """Plot topic preferences by publisher"""
        topic_df = self.identify_topic_preferences(topic_keywords, top_n)
        
        # Prepare data for heatmap
        topic_cols = [col for col in topic_df.columns if col.endswith('_percentage')]
        heatmap_data = topic_df.set_index('publisher')[topic_cols]
        heatmap_data.columns = [col.replace('_percentage', '') for col in heatmap_data.columns]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage of Articles'}, ax=ax)
        ax.set_title('Topic Preferences by Publisher (Percentage of Articles)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Topic Category', fontsize=12)
        ax.set_ylabel('Publisher', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/publisher_topic_preferences.png', dpi=300, bbox_inches='tight')
        plt.close()



