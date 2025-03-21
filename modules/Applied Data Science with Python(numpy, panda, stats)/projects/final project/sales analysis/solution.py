#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AAL Sales Analysis Solution
This script analyzes sales data for AAL clothing brand in Australia.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import os
from datetime import datetime

# Constants
DATA_FILE = "AusApparalSales4thQrt2020.csv"
CHECKPOINTS_DIR = "checkpoints"
RESULTS_DIR = "results"

class SalesAnalysis:
    def __init__(self):
        self.data = None
        self.normalized_data = None
        self.data_wrangling_results = {}
        
    def load_data(self):
        """Load and perform initial data inspection."""
        try:
            # Read the CSV file
            self.data = pd.read_csv(DATA_FILE)
            
            # Convert Date column to datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'], format='%d-%b-%Y')
            
            # Convert Time to categorical
            self.data['Time'] = pd.Categorical(self.data['Time'], 
                                             categories=['Morning', 'Afternoon', 'Evening'],
                                             ordered=True)
            
            # Convert Group to categorical
            self.data['Group'] = pd.Categorical(self.data['Group'])
            
            # Convert State to categorical
            self.data['State'] = pd.Categorical(self.data['State'])
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def data_wrangling(self):
        """Perform data cleaning and preprocessing."""
        if self.data is None:
            print("Please load data first using load_data()")
            return False
            
        # Store all results in a dictionary
        results = {}
        
        # 1. Check for missing values
        results['missing_values'] = {
            'total_missing': self.data.isna().sum().sum(),
            'missing_by_column': self.data.isna().sum().to_dict()
        }
        
        # 2. Check data types
        results['data_types'] = self.data.dtypes.to_dict()
        
        # 3. Basic data info
        buffer = []
        self.data.info(buf=buffer)
        results['data_info'] = ''.join(buffer)
        
        # 4. Check for duplicates
        results['duplicates'] = {
            'total_duplicates': self.data.duplicated().sum(),
            'duplicate_rows': self.data[self.data.duplicated()].to_dict('records') if self.data.duplicated().sum() > 0 else []
        }
        
        # 5. Basic statistics for numerical columns
        results['numerical_statistics'] = self.data.describe().to_dict()
        
        # 6. Value counts for categorical columns
        results['categorical_distributions'] = {
            'State': self.data['State'].value_counts().to_dict(),
            'Group': self.data['Group'].value_counts().to_dict(),
            'Time': self.data['Time'].value_counts().to_dict()
        }
        
        # 7. Date range information
        results['date_range'] = {
            'start_date': self.data['Date'].min().strftime('%Y-%m-%d'),
            'end_date': self.data['Date'].max().strftime('%Y-%m-%d'),
            'total_days': (self.data['Date'].max() - self.data['Date'].min()).days + 1
        }
        
        # Store results
        self.data_wrangling_results = results
        return results
        
    def normalize_data(self):
        """Normalize numerical columns in the dataset."""
        if self.data is None:
            print("Please load data first using load_data()")
            return False
            
        # Create copy of data for normalization
        self.normalized_data = self.data.copy()
        
        # Identify numerical columns (only Unit and Sales should be normalized)
        numerical_cols = ['Unit', 'Sales']
        
        # Apply min-max normalization
        for col in numerical_cols:
            min_val = self.data[col].min()
            max_val = self.data[col].max()
            self.normalized_data[f'{col}_normalized'] = (self.data[col] - min_val) / (max_val - min_val)
            
        return {
            'original_ranges': {
                col: {'min': self.data[col].min(), 'max': self.data[col].max()}
                for col in numerical_cols
            },
            'normalized_ranges': {
                f'{col}_normalized': {'min': self.normalized_data[f'{col}_normalized'].min(), 
                                    'max': self.normalized_data[f'{col}_normalized'].max()}
                for col in numerical_cols
            }
        }
        
    def group_analysis(self):
        """Perform grouping analysis on the data."""
        if self.data is None:
            print("Please load data first using load_data()")
            return False
            
        grouping_results = {}
        
        # 1. State-wise analysis
        state_analysis = self.data.groupby('State').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Unit': ['sum', 'mean']
        }).round(2)
        
        # 2. Group-wise analysis
        group_analysis = self.data.groupby('Group').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Unit': ['sum', 'mean']
        }).round(2)
        
        # 3. Time-wise analysis
        time_analysis = self.data.groupby('Time').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Unit': ['sum', 'mean']
        }).round(2)
        
        # 4. State-Group combined analysis
        state_group_analysis = self.data.groupby(['State', 'Group'])['Sales'].sum().unstack()
        
        grouping_results = {
            'state_analysis': state_analysis.to_dict(),
            'group_analysis': group_analysis.to_dict(),
            'time_analysis': time_analysis.to_dict(),
            'state_group_analysis': state_group_analysis.to_dict()
        }
        
        return grouping_results
        
    def descriptive_statistics(self):
        """Calculate comprehensive descriptive statistics for Sales and Unit columns."""
        if self.data is None:
            print("Please load data first using load_data()")
            return False
            
        stats_dict = {}
        
        # Analyze Sales and Unit columns
        for column in ['Sales', 'Unit']:
            stats_dict[column] = {
                'basic_stats': {
                    'mean': self.data[column].mean(),
                    'median': self.data[column].median(),
                    'mode': self.data[column].mode().values[0],
                    'std': self.data[column].std(),
                    'var': self.data[column].var(),
                    'skew': self.data[column].skew(),
                    'kurtosis': self.data[column].kurtosis()
                },
                'quartiles': {
                    'q1': self.data[column].quantile(0.25),
                    'q2': self.data[column].quantile(0.50),
                    'q3': self.data[column].quantile(0.75),
                    'iqr': self.data[column].quantile(0.75) - self.data[column].quantile(0.25)
                },
                'range': {
                    'min': self.data[column].min(),
                    'max': self.data[column].max(),
                    'range': self.data[column].max() - self.data[column].min()
                }
            }
            
            # Add outlier detection
            q1 = stats_dict[column]['quartiles']['q1']
            q3 = stats_dict[column]['quartiles']['q3']
            iqr = stats_dict[column]['quartiles']['iqr']
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = self.data[
                (self.data[column] < lower_bound) | 
                (self.data[column] > upper_bound)
            ]
            
            stats_dict[column]['outliers'] = {
                'count': len(outliers),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_rows': outliers.to_dict('records') if len(outliers) > 0 else []
            }
        
        return stats_dict
        
    def sales_performance_analysis(self):
        """Perform comprehensive sales performance analysis."""
        if self.data is None:
            print("Please load data first using load_data()")
            return False
            
        performance_results = {}
        
        # 1. Overall Sales Performance
        performance_results['overall'] = {
            'total_sales': self.data['Sales'].sum(),
            'total_units': self.data['Unit'].sum(),
            'average_sale_per_unit': (self.data['Sales'].sum() / self.data['Unit'].sum()).round(2)
        }
        
        # 2. State Performance
        state_performance = self.data.groupby('State').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Unit': ['sum', 'mean']
        }).round(2)
        
        # Add market share calculation
        total_sales = self.data['Sales'].sum()
        state_market_share = (state_performance[('Sales', 'sum')] / total_sales * 100).round(2)
        
        performance_results['state_performance'] = {
            'metrics': state_performance.to_dict(),
            'market_share': state_market_share.to_dict()
        }
        
        # 3. Group Performance
        group_performance = self.data.groupby('Group').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Unit': ['sum', 'mean']
        }).round(2)
        
        # Add market share calculation for groups
        group_market_share = (group_performance[('Sales', 'sum')] / total_sales * 100).round(2)
        
        performance_results['group_performance'] = {
            'metrics': group_performance.to_dict(),
            'market_share': group_market_share.to_dict()
        }
        
        # 4. Time of Day Analysis
        time_performance = self.data.groupby('Time').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Unit': ['sum', 'mean']
        }).round(2)
        
        performance_results['time_performance'] = time_performance.to_dict()
        
        # 5. Daily Performance Trends
        daily_performance = self.data.groupby('Date').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Unit': ['sum', 'mean']
        }).round(2)
        
        performance_results['daily_performance'] = {
            'metrics': daily_performance.to_dict(),
            'trends': {
                'highest_sales_day': daily_performance[('Sales', 'sum')].idxmax().strftime('%Y-%m-%d'),
                'lowest_sales_day': daily_performance[('Sales', 'sum')].idxmin().strftime('%Y-%m-%d'),
                'average_daily_sales': daily_performance[('Sales', 'mean')].mean().round(2)
            }
        }
        
        # 6. Cross Analysis (State x Group)
        cross_analysis = pd.pivot_table(
            self.data,
            values='Sales',
            index='State',
            columns='Group',
            aggfunc='sum'
        ).round(2)
        
        performance_results['cross_analysis'] = cross_analysis.to_dict()
        
        return performance_results
        
    def create_visualizations(self):
        """Create comprehensive visualizations for the analysis."""
        if self.data is None:
            print("Please load data first using load_data()")
            return False
            
        # Create results directory if it doesn't exist
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        # Set style for all plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        
        visualizations = {}
        
        # 1. State-wise Sales Analysis
        def create_state_analysis():
            # Bar plot for state-wise sales
            plt.figure(figsize=(12, 6))
            state_sales = self.data.groupby('State')['Sales'].sum().sort_values(ascending=False)
            sns.barplot(x=state_sales.index, y=state_sales.values)
            plt.title('Total Sales by State')
            plt.xlabel('State')
            plt.ylabel('Total Sales (AUD)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/state_sales.png')
            plt.close()
            
            # Create state-wise sales by group
            plt.figure(figsize=(12, 6))
            state_group_sales = self.data.pivot_table(
                values='Sales',
                index='State',
                columns='Group',
                aggfunc='sum'
            )
            state_group_sales.plot(kind='bar', stacked=True)
            plt.title('State-wise Sales by Group')
            plt.xlabel('State')
            plt.ylabel('Sales (AUD)')
            plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/state_group_sales.png')
            plt.close()
            
            return ['state_sales.png', 'state_group_sales.png']
            
        # 2. Time Analysis
        def create_time_analysis():
            # Daily sales trend
            plt.figure(figsize=(15, 6))
            daily_sales = self.data.groupby('Date')['Sales'].sum()
            plt.plot(daily_sales.index, daily_sales.values, marker='o')
            plt.title('Daily Sales Trend')
            plt.xlabel('Date')
            plt.ylabel('Total Sales (AUD)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/daily_sales_trend.png')
            plt.close()
            
            # Time of day analysis
            plt.figure(figsize=(10, 6))
            time_sales = self.data.groupby('Time')['Sales'].sum()
            sns.barplot(x=time_sales.index, y=time_sales.values)
            plt.title('Sales by Time of Day')
            plt.xlabel('Time of Day')
            plt.ylabel('Total Sales (AUD)')
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/time_of_day_sales.png')
            plt.close()
            
            return ['daily_sales_trend.png', 'time_of_day_sales.png']
            
        # 3. Demographic Analysis
        def create_demographic_analysis():
            # Group performance
            plt.figure(figsize=(10, 6))
            group_sales = self.data.groupby('Group')['Sales'].sum().sort_values(ascending=False)
            sns.barplot(x=group_sales.index, y=group_sales.values)
            plt.title('Total Sales by Demographic Group')
            plt.xlabel('Group')
            plt.ylabel('Total Sales (AUD)')
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/group_sales.png')
            plt.close()
            
            # Group performance by time of day
            plt.figure(figsize=(12, 6))
            time_group_sales = self.data.pivot_table(
                values='Sales',
                index='Time',
                columns='Group',
                aggfunc='sum'
            )
            time_group_sales.plot(kind='bar', stacked=True)
            plt.title('Group Sales by Time of Day')
            plt.xlabel('Time of Day')
            plt.ylabel('Sales (AUD)')
            plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/time_group_sales.png')
            plt.close()
            
            return ['group_sales.png', 'time_group_sales.png']
            
        # 4. Statistical Visualizations
        def create_statistical_visualizations():
            # Sales distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(data=self.data, x='Sales', bins=30, kde=True)
            plt.title('Sales Distribution')
            plt.xlabel('Sales (AUD)')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/sales_distribution.png')
            plt.close()
            
            # Box plots for sales by group and state
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            sns.boxplot(data=self.data, x='Group', y='Sales', ax=ax1)
            ax1.set_title('Sales Distribution by Group')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            
            sns.boxplot(data=self.data, x='State', y='Sales', ax=ax2)
            ax2.set_title('Sales Distribution by State')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/sales_boxplots.png')
            plt.close()
            
            return ['sales_distribution.png', 'sales_boxplots.png']
            
        # Create all visualizations
        visualizations['state_analysis'] = create_state_analysis()
        visualizations['time_analysis'] = create_time_analysis()
        visualizations['demographic_analysis'] = create_demographic_analysis()
        visualizations['statistical_analysis'] = create_statistical_visualizations()
        
        # Create an interactive dashboard using plotly
        def create_interactive_dashboard():
            # State-wise sales
            fig_state = px.bar(
                self.data.groupby('State')['Sales'].sum().reset_index(),
                x='State',
                y='Sales',
                title='Interactive State-wise Sales'
            )
            fig_state.write_html(f'{RESULTS_DIR}/interactive_state_sales.html')
            
            # Daily trend
            fig_daily = px.line(
                self.data.groupby('Date')['Sales'].sum().reset_index(),
                x='Date',
                y='Sales',
                title='Interactive Daily Sales Trend'
            )
            fig_daily.write_html(f'{RESULTS_DIR}/interactive_daily_trend.html')
            
            # Group performance
            fig_group = px.pie(
                self.data.groupby('Group')['Sales'].sum().reset_index(),
                values='Sales',
                names='Group',
                title='Interactive Sales Distribution by Group'
            )
            fig_group.write_html(f'{RESULTS_DIR}/interactive_group_sales.html')
            
            return [
                'interactive_state_sales.html',
                'interactive_daily_trend.html',
                'interactive_group_sales.html'
            ]
            
        visualizations['interactive_dashboard'] = create_interactive_dashboard()
        
        return visualizations
        
    def generate_report(self):
        """Generate comprehensive analysis report with insights and recommendations."""
        if self.data is None:
            print("Please load data first using load_data()")
            return False
            
        # Get all analysis results
        wrangling_results = self.data_wrangling()
        stats_results = self.descriptive_statistics()
        performance_results = self.sales_performance_analysis()
        
        # Create the report
        report = []
        
        # 1. Executive Summary
        report.append("# AAL Sales Analysis Report\n")
        report.append("## Executive Summary\n")
        report.append("This report presents a comprehensive analysis of AAL's sales data for the fourth quarter, examining performance across states, demographic groups, and time periods.\n")
        
        # Add key metrics
        total_sales = performance_results['overall']['total_sales']
        total_units = performance_results['overall']['total_units']
        avg_sale_per_unit = performance_results['overall']['average_sale_per_unit']
        
        report.append("\n### Key Metrics\n")
        report.append(f"- Total Sales: AUD {total_sales:,.2f}")
        report.append(f"- Total Units Sold: {total_units:,}")
        report.append(f"- Average Sale per Unit: AUD {avg_sale_per_unit:,.2f}\n")
        
        # 2. Data Quality Assessment
        report.append("\n## Data Quality Assessment\n")
        report.append("### Data Overview\n")
        report.append(f"- Date Range: {wrangling_results['date_range']['start_date']} to {wrangling_results['date_range']['end_date']}")
        report.append(f"- Total Days: {wrangling_results['date_range']['total_days']}")
        report.append(f"- Missing Values: {wrangling_results['missing_values']['total_missing']}")
        report.append(f"- Duplicate Entries: {wrangling_results['duplicates']['total_duplicates']}\n")
        
        # 3. Statistical Analysis
        report.append("\n## Statistical Analysis\n")
        
        # Sales Statistics
        report.append("### Sales Statistics\n")
        sales_stats = stats_results['Sales']['basic_stats']
        report.append(f"- Mean Sales: AUD {sales_stats['mean']:,.2f}")
        report.append(f"- Median Sales: AUD {sales_stats['median']:,.2f}")
        report.append(f"- Standard Deviation: AUD {sales_stats['std']:,.2f}")
        report.append(f"- Skewness: {sales_stats['skew']:,.2f}")
        report.append(f"- Kurtosis: {sales_stats['kurtosis']:,.2f}\n")
        
        # 4. Performance Analysis
        report.append("\n## Performance Analysis\n")
        
        # State Performance
        report.append("### State-wise Performance\n")
        state_market_share = performance_results['state_performance']['market_share']
        top_state = max(state_market_share.items(), key=lambda x: x[1])
        bottom_state = min(state_market_share.items(), key=lambda x: x[1])
        
        report.append(f"- Top Performing State: {top_state[0]} ({top_state[1]:.2f}% market share)")
        report.append(f"- Bottom Performing State: {bottom_state[0]} ({bottom_state[1]:.2f}% market share)\n")
        
        # Group Performance
        report.append("### Demographic Group Performance\n")
        group_market_share = performance_results['group_performance']['market_share']
        top_group = max(group_market_share.items(), key=lambda x: x[1])
        bottom_group = min(group_market_share.items(), key=lambda x: x[1])
        
        report.append(f"- Top Performing Group: {top_group[0]} ({top_group[1]:.2f}% market share)")
        report.append(f"- Bottom Performing Group: {bottom_group[0]} ({bottom_group[1]:.2f}% market share)\n")
        
        # Time Analysis
        report.append("### Time Analysis\n")
        daily_trends = performance_results['daily_performance']['trends']
        report.append(f"- Highest Sales Day: {daily_trends['highest_sales_day']}")
        report.append(f"- Lowest Sales Day: {daily_trends['lowest_sales_day']}")
        report.append(f"- Average Daily Sales: AUD {daily_trends['average_daily_sales']:,.2f}\n")
        
        # 5. Key Insights
        report.append("\n## Key Insights\n")
        
        # State Insights
        report.append("### State-level Insights\n")
        report.append(f"1. {top_state[0]} leads in sales with {top_state[1]:.2f}% market share")
        report.append(f"2. {bottom_state[0]} shows opportunity for growth with {bottom_state[1]:.2f}% market share")
        report.append("3. Consider implementing successful strategies from top-performing states in lower-performing regions\n")
        
        # Demographic Insights
        report.append("### Demographic Insights\n")
        report.append(f"1. {top_group[0]} segment shows strongest performance with {top_group[1]:.2f}% market share")
        report.append(f"2. {bottom_group[0]} segment may need targeted marketing with {bottom_group[1]:.2f}% market share")
        report.append("3. Analyze successful product lines in top-performing segments for potential expansion\n")
        
        # 6. Recommendations
        report.append("\n## Recommendations\n")
        
        # State-focused Recommendations
        report.append("### Geographic Strategy\n")
        report.append(f"1. Focus expansion efforts in {bottom_state[0]} to capture untapped market potential")
        report.append(f"2. Analyze successful practices in {top_state[0]} for potential replication")
        report.append("3. Consider regional marketing campaigns to boost performance in lower-performing states\n")
        
        # Demographic-focused Recommendations
        report.append("### Demographic Strategy\n")
        report.append(f"1. Develop targeted marketing campaigns for {bottom_group[0]} segment")
        report.append(f"2. Leverage insights from {top_group[0]} segment success")
        report.append("3. Consider product line expansion based on demographic preferences\n")
        
        # Time-based Recommendations
        report.append("### Operational Strategy\n")
        report.append("1. Optimize staffing during peak sales periods")
        report.append("2. Consider special promotions during off-peak hours")
        report.append("3. Align inventory management with daily sales patterns\n")
        
        # Save the report
        report_content = "\n".join(report)
        with open(f"{RESULTS_DIR}/final_report.md", "w") as f:
            f.write(report_content)
            
        return True
        
def main():
    """Main function to run the analysis."""
    analysis = SalesAnalysis()
    
    # Load data
    if not analysis.load_data():
        return
        
    # Perform analysis steps
    wrangling_results = analysis.data_wrangling()
    analysis.normalize_data()
    stats_results = analysis.descriptive_statistics()
    performance_results = analysis.sales_performance_analysis()
    
    # Create visualizations and generate report
    visualizations = analysis.create_visualizations()
    analysis.generate_report()

if __name__ == "__main__":
    main()
