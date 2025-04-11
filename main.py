import pandas as pd

# Load CSV file
df = pd.read_csv('credit_data.csv')

# Print basic info
print("Number of records:", len(df))
print("Average Credit Score:", df['CreditScore'].mean())
print("Total Loan Amount:", df['LoanAmount'].sum())

# Flag Risk Level
def flag_risk(score):
    return "High Risk" if score < 600 else "Low Risk"

df['RiskLevel'] = df['CreditScore'].apply(flag_risk)

# Save summary by Risk Level
summary = df.groupby('RiskLevel').agg({'LoanAmount': 'sum'})
summary.to_csv('summary_report.csv')

print("Summary saved to summary_report.csv")