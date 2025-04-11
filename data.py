import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 realistic companies
companies = []
industries = ['Technology', 'Banking', 'Automotive', 'Energy', 'Pharmaceuticals', 'Retail', 'Telecom']
countries = ['USA', 'UK', 'Germany', 'Japan', 'France', 'Brazil', 'India']

for i in range(1, 101):
    company_id = 2000 + i
    company_name = f"Company_{i}_Inc" if i <= 50 else f"Global_{i-50}_Ltd"
    industry = np.random.choice(industries)
    country = np.random.choice(countries)
    
    # Financials (scaled by industry)
    revenue = np.random.lognormal(mean=5, sigma=0.8) * 1000  # USD millions
    ebitda = revenue * (0.15 + np.random.normal(0.05, 0.02))  # Margin variation
    total_assets = revenue * (1.2 + np.random.uniform(-0.3, 0.5))
    cash = total_assets * (0.05 + np.random.uniform(0, 0.1))
    total_debt = total_assets * (0.3 + np.random.uniform(-0.1, 0.2))
    
    # Credit ratings (linked to financial health)
    if ebitda/total_debt > 5:
        ratings = ['Aaa', 'AAA', 'AAA'] if np.random.rand() > 0.3 else ['Aa1', 'AA+', 'AA+']
    elif ebitda/total_debt > 2:
        ratings = ['A2', 'A', 'A-'] if np.random.rand() > 0.5 else ['Baa1', 'BBB+', 'BBB+']
    else:
        ratings = ['Ba2', 'BB', 'BB-'] if np.random.rand() > 0.7 else ['B1', 'B+', 'B']
    
    companies.append([
        company_id, f"{company_name}", round(revenue, 2), round(ebitda, 2),
        round(total_debt, 2), round(total_assets, 2), round(cash, 2),
        ratings[0], ratings[1], ratings[2], industry, country
    ])

# Create DataFrame
df = pd.DataFrame(companies, columns=[
    'company_id', 'company_name', 'revenue_2023 (USD M)', 'ebitda (USD M)',
    'total_debt (USD M)', 'total_assets (USD M)', 'cash_equivalents (USD M)',
    'moodys_rating', 'sp_rating', 'fitch_rating', 'industry', 'country'
])

# Save to CSV
df.to_csv("corporate_credit_2024.csv", index=False)
print("File generated: corporate_credit_2024.csv") 