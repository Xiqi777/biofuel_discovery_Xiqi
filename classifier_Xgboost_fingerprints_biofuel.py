import pandas as pd

# Read the Excel file
excel_file = 'new dataset.xlsx'  # Replace with your Excel file name
df = pd.read_excel(excel_file)

# Save the DataFrame as a CSV file
csv_file = 'new dataset.csv'
df.to_csv(csv_file, index=False)

print(f'The Excel file has been successfully converted to {csv_file}')








