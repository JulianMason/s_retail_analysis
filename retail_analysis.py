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

def analyze_recommendations():
    """Create visualizations for recommendations"""
    # Labour hours adjustment impact
    current_hours = 42.5
    new_hours = 40
    current_cost = 510.43
    new_cost = 518.93
    
    # Create bar plot for labour hours
    plt.figure(figsize=(10, 6))
    hours_data = pd.Series({
        'Current Hours': current_hours,
        'Proposed Hours': new_hours
    })
    hours_data.plot(kind='bar')
    plt.title('Labour Hours Before and After Recommendations')
    plt.ylabel('Weekly Hours')
    plt.tight_layout()
    plt.savefig('visualizations/labour_hours_comparison.png')
    plt.close()
    
    # Create bar plot for labour costs
    plt.figure(figsize=(10, 6))
    cost_data = pd.Series({
        'Current Cost': current_cost,
        'New Cost': new_cost
    })
    cost_data.plot(kind='bar')
    plt.title('Weekly Labour Cost Per Staff Member')
    plt.ylabel('Cost (£)')
    plt.tight_layout()
    plt.savefig('visualizations/labour_cost_comparison.png')
    plt.close()

def analyze_unit_size_optimization():
    """Create visualizations for unit size optimization"""
    # Current unit size distribution
    current_dist = {
        'Small (10-15 sqft)': 20,
        'Medium (16-50 sqft)': 40,
        'Large (51-100 sqft)': 25,
        'Extra Large (100+ sqft)': 15
    }
    
    # Projected distribution after optimization
    projected_dist = {
        'Medium (16-50 sqft)': 45,
        'Large (51-100 sqft)': 35,
        'Extra Large (100+ sqft)': 20
    }
    
    # Create stacked bar plot
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame({
        'Current': current_dist,
        'Projected': projected_dist
    })
    df.plot(kind='bar', stacked=True)
    plt.title('Unit Size Distribution Before and After Optimization')
    plt.xlabel('Unit Size')
    plt.ylabel('Percentage')
    plt.legend(title='Time Period')
    plt.tight_layout()
    plt.savefig('visualizations/unit_size_optimization.png')
    plt.close()

def analyze_impact_analysis():
    """Create visualizations for impact analysis"""
    # Cost reduction data
    cost_data = {
        'Reception & Caretaker': {'Current': 22969.35, 'New': 21695.50},
        'Customer Service': {'Current': 6005.00, 'New': 6105.00},
        'Total': {'Current': 28974.35, 'New': 27800.50}
    }
    
    # Create grouped bar plot
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame(cost_data)
    df.plot(kind='bar')
    plt.title('Weekly Labour Costs Before and After Recommendations')
    plt.xlabel('Staff Category')
    plt.ylabel('Weekly Cost (£)')
    plt.legend(title='Time Period')
    plt.tight_layout()
    plt.savefig('visualizations/cost_reduction_impact.png')
    plt.close()
    
    # Revenue enhancement data
    revenue_data = {
        'Small Units': {'Current': 100, 'Projected': 0},
        'Medium Units': {'Current': 100, 'Projected': 125},
        'Large Units': {'Current': 100, 'Projected': 100},
        'Extra Large Units': {'Current': 100, 'Projected': 100}
    }
    
    # Create line plot
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame(revenue_data)
    df.plot(kind='line', marker='o')
    plt.title('Revenue Impact by Unit Size (Indexed)')
    plt.xlabel('Unit Size')
    plt.ylabel('Revenue Index (Current = 100)')
    plt.legend(title='Time Period')
    plt.tight_layout()
    plt.savefig('visualizations/revenue_enhancement.png')
    plt.close()
    
    # Margin improvement data
    margin_data = {
        'Gross Margin': {'Current': 100, 'Projected': 107},
        'Operating Margin': {'Current': 100, 'Projected': 105},
        'Net Margin': {'Current': 100, 'Projected': 104}
    }
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame(margin_data)
    df.plot(kind='bar')
    plt.title('Margin Improvement Projections')
    plt.xlabel('Margin Type')
    plt.ylabel('Margin Index (Current = 100)')
    plt.legend(title='Time Period')
    plt.tight_layout()
    plt.savefig('visualizations/margin_improvement.png')
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
    
    # Run new analyses
    analyze_recommendations()
    analyze_unit_size_optimization()
    analyze_impact_analysis()
    
    print("Analysis complete! Visualizations have been saved to the 'visualizations' directory.")

if __name__ == "__main__":
    main() 