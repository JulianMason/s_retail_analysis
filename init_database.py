"""
Database Initialization Script
----------------------------
Creates and populates the SQLite database with sample data for testing.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data():
    # Create sample data
    n_records = 1000
    
    # Generate random data
    np.random.seed(42)
    
    data = {
        'unit_id': range(1, n_records + 1),
        'unit_size': np.random.choice(['small', 'medium', 'large'], n_records, p=[0.4, 0.4, 0.2]),
        'monthly_rent': np.random.normal(150, 50, n_records).clip(50, 300),
        'occupancy_days': np.random.randint(0, 365, n_records),
        'customer_type': np.random.choice(['residential', 'business', 'student'], n_records, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['direct_debit', 'credit_card', 'bank_transfer'], n_records, p=[0.6, 0.3, 0.1]),
        'contract_length': np.random.choice([1, 3, 6, 12], n_records, p=[0.2, 0.3, 0.2, 0.3]),
        'last_payment_date': pd.date_range(start='2024-01-01', periods=n_records, freq='D').strftime('%Y-%m-%d'),
        'maintenance_required': np.random.choice([0, 1], n_records, p=[0.8, 0.2]),
        'customer_satisfaction': np.random.randint(1, 6, n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Round monthly rent to 2 decimal places
    df['monthly_rent'] = df['monthly_rent'].round(2)
    
    return df

def init_database():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create database connection
    db_path = data_dir / 'storage_data.db'
    conn = sqlite3.connect(db_path)
    
    # Create sample data
    df = create_sample_data()
    
    # Write to database
    df.to_sql('storage_data', conn, if_exists='replace', index=False)
    
    print(f"Database initialized at {db_path}")
    print(f"Created {len(df)} records")
    
    conn.close()

if __name__ == "__main__":
    init_database() 