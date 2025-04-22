import pandas as pd

# Load the data, skipping the first three rows
data = pd.read_csv('2024-03-22_12-45-31.csv', sep=';', skiprows=3)

# Print the columns to verify that 'north_from_observer' is present
print(data.columns)

# Additionally, print the first few rows to ensure that the data is read correctly
print(data.head())
