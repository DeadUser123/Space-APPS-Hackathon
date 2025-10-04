import pandas as pd

# Path to your CSV file
CSV_FILE = 'merged_exoplanets.csv'

# Load CSV into a DataFrame
df = pd.read_csv(CSV_FILE)

# Select numeric columns (ignores disposition and source)
numeric_cols = df.select_dtypes(include='number').columns

# Create a dictionary to store min and max
min_max_dict = {}

for col in numeric_cols:
    min_max_dict[col] = {
        'min': df[col].min(),
        'max': df[col].max()
    }

# Convert to DataFrame for easy CSV export
min_max_df = pd.DataFrame(min_max_dict).T  # Transpose for readability

# Save min/max values to a new CSV
min_max_df.to_csv('min_max_values.csv')

print("Min and max values saved to 'min_max_values.csv'.")
