import pandas as pd

# File paths
input_file = 'creditcard.csv'
output_file = 'creditcard_filtered.csv'

# Load the data
data = pd.read_csv(input_file)

# Separate class 0 and class 1 data
class_1_data = data[data['Class'] == 1]
class_0_data = data[data['Class'] == 0]

# Determine the size of the filtered file
target_size_mb = 15
current_size_mb = 150
filter_ratio = target_size_mb / current_size_mb

# Sample a subset of class 0 data
num_class_0_rows = int(len(class_0_data) * filter_ratio)
sampled_class_0_data = class_0_data.sample(n=num_class_0_rows, random_state=42)

# Combine the data
filtered_data = pd.concat([class_1_data, sampled_class_0_data])

# Save the filtered data to a new file
filtered_data.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}.")
