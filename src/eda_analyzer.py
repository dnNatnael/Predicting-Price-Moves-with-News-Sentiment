"""
Exploratory Data Analysis Module
Performs comprehensive EDA on the financial news dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


class EDAAnalyzer:
    """Perform comprehensive EDA on financial news data"""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "output"):
        """
        Initialize the EDA Analyzer
        
        Args:
            df: Preprocessed DataFrame
            output_dir: Directory to save outputs
        """
        self.df = df
        self.output_dir = output_dir
        self.stats = {}
        
    def compute_descriptive_stats(self) -> Dict:
        """
        Compute descriptive statistics for headline lengths
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'headline_length': {
                'min': self.df['headline_length'].min(),
                'max': self.df['headline_length'].max(),
                'mean': self.df['headline_length'].mean(),
                'median': self.df['headline_length'].median(),
                'std': self.df['headline_length'].std(),
                'q25': self.df['headline_length'].quantile(0.25),
                'q75': self.df['headline_length'].quantile(0.75)
            },
            'headline_word_count': {
                'min': self.df['headline_word_count'].min(),
                'max': self.df['headline_word_count'].max(),
                'mean': self.df['headline_word_count'].mean(),
                'median': self.df['headline_word_count'].median(),
                'std': self.df['headline_word_count'].std(),
                'q25': self.df['headline_word_count'].quantile(0.25),
                'q75': self.df['headline_word_count'].quantile(0.75)
            },
            'total_articles': len(self.df),
            'unique_publishers': self.df['publisher'].nunique(),
            'unique_stocks': self.df['stock'].nunique(),
            'date_range': {
                'start': self.df['date'].min(),
                'end': self.df['date'].max()
            }
        }
        
        self.stats['descriptive'] = stats
        return stats
    
    def analyze_top_publishers(self, top_n: int = 10) -> pd.DataFrame:
        """
        Identify top N most active publishers
        
        Args:
            top_n: Number of top publishers to return
            
        Returns:
            DataFrame with publisher statistics
        """
        publisher_stats = self.df.groupby('publisher').agg({
            'headline': 'count',
            'stock': 'nunique',
            'date': ['min', 'max']
        }).reset_index()
        
        publisher_stats.columns = ['publisher', 'article_count', 'unique_stocks', 'first_article', 'last_article']
        publisher_stats = publisher_stats.sort_values('article_count', ascending=False).head(top_n)
        
        self.stats['top_publishers'] = publisher_stats
        return publisher_stats
    
    def plot_headline_length_distribution(self, save: bool = True):
        """Plot headline length distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Character length distribution
        axes[0, 0].hist(self.df['headline_length'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.df['headline_length'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {self.df["headline_length"].mean():.1f}')
        axes[0, 0].axvline(self.df['headline_length'].median(), color='green', 
                          linestyle='--', label=f'Median: {self.df["headline_length"].median():.1f}')
        axes[0, 0].set_xlabel('Headline Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Headline Length (Characters)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Word count distribution
        axes[0, 1].hist(self.df['headline_word_count'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(self.df['headline_word_count'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {self.df["headline_word_count"].mean():.1f}')
        axes[0, 1].axvline(self.df['headline_word_count'].median(), color='green', 
                          linestyle='--', label=f'Median: {self.df["headline_word_count"].median():.1f}')
        axes[0, 1].set_xlabel('Headline Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Headline Word Count')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot for character length
        axes[1, 0].boxplot(self.df['headline_length'], vert=True)
        axes[1, 0].set_ylabel('Headline Length (characters)')
        axes[1, 0].set_title('Box Plot: Headline Length')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot for word count
        axes[1, 1].boxplot(self.df['headline_word_count'], vert=True)
        axes[1, 1].set_ylabel('Headline Word Count')
        axes[1, 1].set_title('Box Plot: Headline Word Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/headline_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_top_publishers(self, top_n: int = 10, save: bool = True):
        """Plot top publishers by article count"""
        top_publishers = self.analyze_top_publishers(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        axes[0].barh(range(len(top_publishers)), top_publishers['article_count'], 
                     color=sns.color_palette("husl", len(top_publishers)))
        axes[0].set_yticks(range(len(top_publishers)))
        axes[0].set_yticklabels(top_publishers['publisher'], fontsize=9)
        axes[0].set_xlabel('Number of Articles')
        axes[0].set_title(f'Top {top_n} Publishers by Article Count')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(top_publishers['article_count']):
            axes[0].text(v + max(top_publishers['article_count']) * 0.01, i, 
                        str(int(v)), va='center', fontsize=8)
        
        # Pie chart
        axes[1].pie(top_publishers['article_count'], labels=top_publishers['publisher'], 
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
        axes[1].set_title(f'Top {top_n} Publishers Distribution')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/top_publishers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_publication_frequency(self) -> Dict:
        """
        Analyze publication frequency across days, months, years
        
        Returns:
            Dictionary with frequency statistics
        """
        freq_stats = {
            'by_day': self.df.groupby('day_of_week')['headline'].count().to_dict(),
            'by_month': self.df.groupby('month')['headline'].count().to_dict(),
            'by_year': self.df.groupby('year')['headline'].count().to_dict(),
            'by_hour': self.df.groupby('hour')['headline'].count().to_dict(),
            'daily_timeseries': self.df.groupby('date_only')['headline'].count(),
            'monthly_timeseries': self.df.groupby([self.df['date'].dt.to_period('M')])['headline'].count()
        }
        
        self.stats['publication_frequency'] = freq_stats
        return freq_stats
    
    def plot_publication_frequency(self, save: bool = True):
        """Plot publication frequency patterns"""
        freq_stats = self.analyze_publication_frequency()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # By day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = [freq_stats['by_day'].get(day, 0) for day in day_order]
        axes[0, 0].bar(day_order, day_counts, color='steelblue')
        axes[0, 0].set_xlabel('Day of Week')
        axes[0, 0].set_ylabel('Number of Articles')
        axes[0, 0].set_title('Publication Frequency by Day of Week')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # By month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_counts = [freq_stats['by_month'].get(i+1, 0) for i in range(12)]
        axes[0, 1].bar(month_names, month_counts, color='coral')
        axes[0, 1].set_xlabel('Month')
        axes[0, 1].set_ylabel('Number of Articles')
        axes[0, 1].set_title('Publication Frequency by Month')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # By year
        year_data = sorted(freq_stats['by_year'].items())
        years = [str(y[0]) for y in year_data]
        counts = [y[1] for y in year_data]
        axes[0, 2].bar(years, counts, color='green')
        axes[0, 2].set_xlabel('Year')
        axes[0, 2].set_ylabel('Number of Articles')
        axes[0, 2].set_title('Publication Frequency by Year')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # By hour of day
        hour_data = sorted(freq_stats['by_hour'].items())
        hours = [h[0] for h in hour_data]
        hour_counts = [h[1] for h in hour_data]
        axes[1, 0].plot(hours, hour_counts, marker='o', linewidth=2, markersize=6, color='purple')
        axes[1, 0].set_xlabel('Hour of Day (UTC-4)')
        axes[1, 0].set_ylabel('Number of Articles')
        axes[1, 0].set_title('Publication Frequency by Hour of Day')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(range(0, 24, 2))
        
        # Daily timeseries
        daily_ts = freq_stats['daily_timeseries']
        axes[1, 1].plot(daily_ts.index, daily_ts.values, linewidth=1, alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Number of Articles')
        axes[1, 1].set_title('Daily Publication Frequency (Time Series)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Monthly timeseries
        monthly_ts = freq_stats['monthly_timeseries']
        axes[1, 2].plot(range(len(monthly_ts)), monthly_ts.values, 
                       marker='o', linewidth=2, markersize=4, color='darkblue')
        axes[1, 2].set_xlabel('Month Index')
        axes[1, 2].set_ylabel('Number of Articles')
        axes[1, 2].set_title('Monthly Publication Frequency (Time Series)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/publication_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def detect_news_spikes(self, threshold_std: float = 2.0) -> pd.DataFrame:
        """
        Detect spikes in news activity
        
        Args:
            threshold_std: Number of standard deviations above mean to consider a spike
            
        Returns:
            DataFrame with detected spikes
        """
        daily_counts = self.df.groupby('date_only')['headline'].count()
        mean_count = daily_counts.mean()
        std_count = daily_counts.std()
        threshold = mean_count + (threshold_std * std_count)
        
        spikes = daily_counts[daily_counts > threshold].sort_values(ascending=False)
        spike_df = pd.DataFrame({
            'date': spikes.index,
            'article_count': spikes.values,
            'deviation_from_mean': spikes.values - mean_count,
            'std_deviation': (spikes.values - mean_count) / std_count
        })
        
        self.stats['news_spikes'] = spike_df
        return spike_df
    
    def plot_news_spikes(self, threshold_std: float = 2.0, save: bool = True):
        """Plot detected news spikes"""
        spikes = self.detect_news_spikes(threshold_std)
        daily_counts = self.df.groupby('date_only')['headline'].count()
        mean_count = daily_counts.mean()
        threshold = mean_count + (threshold_std * daily_counts.std())
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.plot(daily_counts.index, daily_counts.values, 
               linewidth=1, alpha=0.6, color='blue', label='Daily Article Count')
        ax.axhline(mean_count, color='green', linestyle='--', 
                  linewidth=2, label=f'Mean: {mean_count:.1f}')
        ax.axhline(threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'Spike Threshold: {threshold:.1f}')
        
        # Highlight spikes
        spike_dates = spikes['date'].tolist()
        spike_counts = spikes['article_count'].tolist()
        ax.scatter(spike_dates, spike_counts, color='red', 
                  s=100, zorder=5, label='Detected Spikes', marker='^')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Articles')
        ax.set_title(f'News Activity Spikes (>{threshold_std}σ above mean)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/news_spikes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_publishers(self) -> Dict:
        """
        Comprehensive publisher analysis
        
        Returns:
            Dictionary with publisher statistics
        """
        publisher_analysis = self.df.groupby('publisher').agg({
            'headline': 'count',
            'stock': 'nunique',
            'headline_length': 'mean',
            'date': ['min', 'max']
        }).reset_index()
        
        publisher_analysis.columns = [
            'publisher', 'article_count', 'unique_stocks', 
            'avg_headline_length', 'first_article', 'last_article'
        ]
        
        publisher_analysis = publisher_analysis.sort_values('article_count', ascending=False)
        
        # Publisher domain analysis
        domain_stats = self.df.groupby('publisher_domain')['headline'].count().sort_values(ascending=False)
        
        self.stats['publisher_analysis'] = publisher_analysis
        self.stats['domain_stats'] = domain_stats
        
        return {
            'publisher_stats': publisher_analysis,
            'domain_stats': domain_stats
        }
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report"""
        report = []
        report.append("=" * 80)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("Financial News and Stock Price Integration Dataset (FNSPID)")
        report.append("=" * 80)
        report.append("")
        
        # Descriptive Statistics
        if 'descriptive' in self.stats:
            desc = self.stats['descriptive']
            report.append("1. DESCRIPTIVE STATISTICS")
            report.append("-" * 80)
            report.append(f"Total Articles: {desc['total_articles']:,}")
            report.append(f"Unique Publishers: {desc['unique_publishers']}")
            report.append(f"Unique Stocks: {desc['unique_stocks']}")
            report.append(f"Date Range: {desc['date_range']['start']} to {desc['date_range']['end']}")
            report.append("")
            report.append("Headline Length (Characters):")
            report.append(f"  Min: {desc['headline_length']['min']}")
            report.append(f"  Max: {desc['headline_length']['max']}")
            report.append(f"  Mean: {desc['headline_length']['mean']:.2f}")
            report.append(f"  Median: {desc['headline_length']['median']:.2f}")
            report.append(f"  Std Dev: {desc['headline_length']['std']:.2f}")
            report.append("")
            report.append("Headline Word Count:")
            report.append(f"  Min: {desc['headline_word_count']['min']}")
            report.append(f"  Max: {desc['headline_word_count']['max']}")
            report.append(f"  Mean: {desc['headline_word_count']['mean']:.2f}")
            report.append(f"  Median: {desc['headline_word_count']['median']:.2f}")
            report.append("")
        
        # Top Publishers
        if 'top_publishers' in self.stats:
            report.append("2. TOP 10 PUBLISHERS")
            report.append("-" * 80)
            for idx, row in self.stats['top_publishers'].iterrows():
                report.append(f"{row['publisher']}: {row['article_count']} articles")
            report.append("")
        
        # Publication Frequency
        if 'publication_frequency' in self.stats:
            freq = self.stats['publication_frequency']
            report.append("3. PUBLICATION FREQUENCY INSIGHTS")
            report.append("-" * 80)
            report.append("Most Active Day of Week:")
            max_day = max(freq['by_day'].items(), key=lambda x: x[1])
            report.append(f"  {max_day[0]}: {max_day[1]} articles")
            report.append("")
            report.append("Most Active Month:")
            max_month = max(freq['by_month'].items(), key=lambda x: x[1])
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            report.append(f"  {month_names[max_month[0]-1]}: {max_month[1]} articles")
            report.append("")
            report.append("Peak Hour:")
            max_hour = max(freq['by_hour'].items(), key=lambda x: x[1])
            report.append(f"  Hour {max_hour[0]}: {max_hour[1]} articles")
            report.append("")
        
        # News Spikes
        if 'news_spikes' in self.stats:
            spikes = self.stats['news_spikes']
            report.append("4. DETECTED NEWS ACTIVITY SPIKES")
            report.append("-" * 80)
            report.append(f"Total Spikes Detected: {len(spikes)}")
            if len(spikes) > 0:
                report.append("Top 5 Spikes:")
                for idx, row in spikes.head(5).iterrows():
                    report.append(f"  {row['date']}: {row['article_count']} articles "
                                f"({row['std_deviation']:.2f}σ above mean)")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

