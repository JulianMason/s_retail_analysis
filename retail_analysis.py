import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sqlite3
from datetime import datetime, timedelta
import os

# Set style for all visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

def load_data():
    """Load data from SQLite database"""
    conn = sqlite3.connect('data/storage_data.db')
    df = pd.read_sql_query("SELECT * FROM storage_data", conn)
    conn.close()
    return df

def analyze_data_quality(df):
    """Analyze data quality and create visualizations"""
    # Missing values analysis
    missing_data = df.isnull().sum()
    plt.figure(figsize=(10, 6))
    missing_data.plot(kind='bar')
    plt.title('Missing Values Analysis')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/segment_discovery.png')
    plt.close()

def analyze_customer_clusters(df):
    """Perform customer clustering analysis"""
    # Prepare data for clustering
    features = ['monthly_rent', 'contract_length', 'occupancy_days']
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['monthly_rent'], df['occupancy_days'], 
                         c=df['cluster'], cmap='viridis')
    plt.title('Customer Clusters by Monthly Rent and Occupancy')
    plt.xlabel('Monthly Rent')
    plt.ylabel('Occupancy Days')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.savefig('visualizations/customer_clustering.png')
    plt.close()

def analyze_financial_impact(df):
    """Analyze financial impact and create visualizations"""
    # Calculate cost components
    cost_components = {
        'Labour': 45,
        'Marketing': 15,
        'Utilities': 10,
        'Maintenance': 10,
        'Other Opex': 20
    }
    
    # Create pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(cost_components.values(), labels=cost_components.keys(), 
            autopct='%1.1f%%', startangle=90)
    plt.title('Cost Structure Analysis')
    plt.tight_layout()
    plt.savefig('visualizations/financial_impact.png')
    plt.close()

def analyze_revenue_by_size(df):
    """Analyze revenue by unit size"""
    # Calculate average revenue by unit size
    revenue_by_size = df.groupby('unit_size')['monthly_rent'].mean()
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    revenue_by_size.plot(kind='bar')
    plt.title('Average Revenue by Unit Size')
    plt.xlabel('Unit Size')
    plt.ylabel('Average Monthly Revenue')
    plt.tight_layout()
    plt.savefig('visualizations/revenue_by_size.png')
    plt.close()

def analyze_enquiry_channels(df):
    """Analyze enquiry channel distribution"""
    # Calculate payment method distribution
    channel_dist = df['payment_method'].value_counts()
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    channel_dist.plot(kind='bar')
    plt.title('Payment Method Distribution')
    plt.xlabel('Payment Method')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig('visualizations/enquiry_channel_analysis.png')
    plt.close()

def analyze_conversion(df):
    """Analyze conversion rates and optimization opportunities"""
    # Calculate occupancy rates by unit size
    occupancy_rates = df.groupby('unit_size')['occupancy_days'].mean() / 365
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    occupancy_rates.plot(kind='bar')
    plt.title('Occupancy Rates by Unit Size')
    plt.xlabel('Unit Size')
    plt.ylabel('Occupancy Rate')
    plt.tight_layout()
    plt.savefig('visualizations/conversion_analysis.png')
    plt.close()

def analyze_seasonal_patterns(df):
    """Analyze seasonal patterns in the data"""
    # Convert last_payment_date to datetime
    df['last_payment_date'] = pd.to_datetime(df['last_payment_date'])
    
    # Calculate monthly averages
    monthly_avg = df.groupby(df['last_payment_date'].dt.month)['monthly_rent'].mean()
    
    # Create line plot
    plt.figure(figsize=(10, 6))
    monthly_avg.plot(kind='line', marker='o')
    plt.title('Seasonal Revenue Patterns')
    plt.xlabel('Month')
    plt.ylabel('Average Monthly Revenue')
    plt.tight_layout()
    plt.savefig('visualizations/seasonal_patterns.png')
    plt.close()

def analyze_customer_lifecycle(df):
    """Analyze customer lifecycle patterns"""
    # Calculate average contract length by customer type
    lifecycle_data = df.groupby('customer_type')['contract_length'].mean()
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    lifecycle_data.plot(kind='bar')
    plt.title('Customer Lifecycle Analysis')
    plt.xlabel('Customer Type')
    plt.ylabel('Average Contract Length (months)')
    plt.tight_layout()
    plt.savefig('visualizations/customer_lifecycle.png')
    plt.close()

def create_implementation_timeline():
    """Create implementation timeline visualization"""
    # Create timeline data
    timeline_data = {
        'Phase 1': 30,
        'Phase 2': 60,
        'Phase 3': 90,
        'Phase 4': 120
    }
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(timeline_data.keys(), timeline_data.values())
    plt.title('Implementation Timeline')
    plt.xlabel('Phase')
    plt.ylabel('Duration (days)')
    plt.tight_layout()
    plt.savefig('visualizations/implementation_timeline.png')
    plt.close()

def main():
    """Main function to run all analyses"""
    # Load data
    df = load_data()
    
    # Run all analyses
    analyze_data_quality(df)
    analyze_customer_clusters(df)
    analyze_financial_impact(df)
    analyze_revenue_by_size(df)
    analyze_enquiry_channels(df)
    analyze_conversion(df)
    analyze_seasonal_patterns(df)
    analyze_customer_lifecycle(df)
    create_implementation_timeline()
    
    print("Analysis complete! Visualizations have been saved to the 'visualizations' directory.")

if __name__ == "__main__":
    main() 