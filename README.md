# Retail Storage Space Optimization Analysis

This project analyzes the impact of the UK's Minimum Wage Increase (MWI) on storage business operations and identifies optimization opportunities across manned (premium) and unmanned (budget) locations.

## Technologies Used
- Python 3.8+
- Pandas for data manipulation and analysis
- NumPy for numerical computations
- Scikit-learn for data analysis and clustering
- Matplotlib and Seaborn for data visualization
- SQLite for data storage

## Project Structure

```
retail_analysis/
├── data/                   # Data files
├── visualizations/         # Analysis visualizations
├── init_database.py        # Database initialization script
├── retail_analysis.py      # Main analysis script
├── retail_analysis_report.md  # Main analysis report
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Key Findings

1. **Cost Optimization**
   - Labour costs are the primary cost driver at 45%
   - Reception staff and caretakers account for 60% of labour costs
   - Marketing costs are the second-largest expense

2. **Revenue Analysis**
   - Business customers provide highest revenue per square foot
   - Student units show significant seasonal variation
   - Small units (10-15 sqft) have lowest occupancy rates at 38%

3. **Operational Efficiency**
   - Premium locations have 45% higher labour costs
   - Budget locations show better cost efficiency
   - Customer service handles 100% of enquiries outside office hours

## Recommendations

1. **Labour Hours Adjustment**
   - Reduce reception and caretaker hours from 42.5 to 40 hours per week
   - Change closing time from 5:30 PM to 5:00 PM
   - Expected annual savings: 3.2% of total labor costs

2. **Unit Size Optimization**
   - Merge small units (10-15 sqft) into medium units
   - Move smaller units to budget locations
   - Expected revenue increase: 25% per converted unit

## Setup and Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Initialize the database:
   ```bash
   python init_database.py
   ```

3. Run the analysis:
   ```bash
   python retail_analysis.py
   ```

## Analysis Components

The analysis includes:
- Data quality assessment and cleaning
- Business cost analysis
- Revenue analysis
- Customer profile analysis
- Enquiry distribution analysis
- Operational efficiency metrics
- Implementation recommendations
- Risk management strategies

## Visualizations

The project includes comprehensive visualizations for:
- Data quality and cleaning
- Business costs and financial impact
- Revenue performance by unit size
- Enquiry channel analysis
- Customer lifecycle analysis
- Seasonal patterns
- Implementation timeline
- Optimization opportunities 
