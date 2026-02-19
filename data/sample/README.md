# Sample Data

This directory contains synthetic sample data for demonstration purposes.

## Generate Sample Data

To create synthetic credit application data:

```bash
python src/data/make_dataset.py --output data/sample/credit_applications.csv --n-samples 10000
```

The generated CSV file (`credit_applications.csv`) is excluded from version control via .gitignore.

## Data Schema

The synthetic dataset includes:
- `application_id`: Unique application identifier
- `application_date`: Date of application
- `age`: Applicant age
- `annual_income`: Annual income
- `employment_length_years`: Years of employment
- `debt_to_income_ratio`: DTI ratio (%)
- `credit_history_length_years`: Years of credit history
- `num_open_credit_lines`: Number of open credit lines
- `num_derogatory_marks`: Number of derogatory marks
- `total_revolving_balance`: Total revolving balance
- `revolving_utilization`: Credit utilization (%)
- `num_recent_inquiries`: Number of recent credit inquiries
- `loan_amount_requested`: Requested loan amount
- `loan_purpose`: Purpose of the loan
- `default`: Target variable (1 = default, 0 = no default)
