import pandas as pd
import numpy as np
import glob
from pathlib import Path
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os

# Define data path
data_path = Path('Dataset/')

# Define the columns we are interested in
carbon_intensity_columns = ['kg_CO2/kWh']
pricing_columns = ['Electricity Pricing [$]']
weather_columns = ['Direct Solar Radiation [W/m2]']
building_columns = ['Equipment Electric Power [kWh]', 'Solar Generation [W/kW]']

# Load the data
carbon_intensity = pd.read_csv(data_path / 'carbon_intensity.csv', usecols=carbon_intensity_columns)
pricing = pd.read_csv(data_path / 'pricing.csv', usecols=pricing_columns)
weather = pd.read_csv(data_path / 'weather.csv', usecols=weather_columns)

# Load Building_1.csv for 'Day Type', 'Equipment Electric Power', and 'Solar Generation'
building_1_full = pd.read_csv(data_path / 'Building_1.csv', usecols=['Day Type', 'Equipment Electric Power [kWh]', 'Solar Generation [W/kW]'])

# Define the days to check
chosen_day_types = [1]  
print(f"chosen day type{chosen_day_types}")

# Load the 17 Building_x.csv files (including Building_1.csv)
building_files = sorted(glob.glob(str(data_path / 'Building_*.csv')))

# Check that 17 files were found
expected_buildings = 17
found_buildings = len(building_files)
if found_buildings != expected_buildings:
    print(f"Warning: Expected {expected_buildings} Building_x.csv files, but found {found_buildings} files.")
else:
    print(f"All {expected_buildings} Building_x.csv files were found.")

# Initialize lists to store the data
equipment_power_list = []
solar_generation_list = []

# Read each file and store the columns
for file in building_files:
    df = pd.read_csv(file, usecols=building_columns)
    equipment_power_list.append(df['Equipment Electric Power [kWh]'])
    solar_generation_list.append(df['Solar Generation [W/kW]'])

# Create DataFrames where each column corresponds to a building
equipment_power_df = pd.concat(equipment_power_list, axis=1)
solar_generation_df = pd.concat(solar_generation_list, axis=1)

# Combine all data into a single DataFrame
# 'Day Type' comes from Building_1.csv
combined_df = pd.concat([
    building_1_full['Day Type'],
    carbon_intensity,
    pricing,
    weather,
    equipment_power_df,
    solar_generation_df
], axis=1)

# Check for NaNs and convert to numeric types
combined_df = combined_df.apply(pd.to_numeric, errors='coerce')

# Handle NaNs (e.g., fill with the mean of each column)
combined_df.fillna(combined_df.mean(), inplace=True)

# Create two DataFrames: one for selected days and one for the overall
secondary_df = combined_df[combined_df['Day Type'].isin(chosen_day_types)]
overall_df = combined_df.copy()

# List of variables to analyze
variables = {
    'Carbon Intensity': 'kg_CO2/kWh',
    'Electricity Pricing': 'Electricity Pricing [$]',
    'Direct Solar Radiation': 'Direct Solar Radiation [W/m2]',
    'Equipment Electric Power': 'Equipment Electric Power [kWh]',
    'Solar Generation': 'Solar Generation [W/kW]'
}

# Function to compute KL divergence
def compute_kl_divergence(p, q):
    # Add a small value to avoid zeros
    p += 1e-10
    q += 1e-10
    return entropy(p, q)

# Initialize lists to store the results
kl_results = {
    'Variable': [],
    'KL Divergence': []
}

# Calculate KL divergence for each variable
for var_name, column in variables.items():
    # Create histogram for secondary and overall
    p_counts, p_bins = np.histogram(secondary_df[column], bins=50, density=True)
    q_counts, q_bins = np.histogram(overall_df[column], bins=50, density=True)
    
    # Calculate KL divergence
    kl_div = compute_kl_divergence(p_counts, q_counts)
    
    # Store the results
    kl_results['Variable'].append(var_name)
    kl_results['KL Divergence'].append(kl_div)

# Create DataFrame with KL divergence results
kl_df = pd.DataFrame(kl_results)

# Calculate Means, Standard Deviations, and Mean Deviations for secondary_df
secondary_stats = {
    'Variable': [],
    'Mean': [],
    'Std Dev': [],
    'Mean Deviation': []
}

for var_name, column in variables.items():
    if var_name in ['Equipment Electric Power', 'Solar Generation']:
        # Calculate the mean per timestep for variables with multiple columns
        column_mean = secondary_df[column].mean(axis=1)
        # Calculate statistical columns for the mean
        mean = column_mean.mean()
        std_dev = column_mean.std()
        mean_dev = (column_mean - mean).abs().mean()
    else:
        mean = secondary_df[column].mean()
        std_dev = secondary_df[column].std()
        mean_dev = (secondary_df[column] - mean).abs().mean()
    
    secondary_stats['Variable'].append(var_name)
    secondary_stats['Mean'].append(mean)
    secondary_stats['Std Dev'].append(std_dev)
    secondary_stats['Mean Deviation'].append(mean_dev)

# Create DataFrame with statistics of secondary_df
secondary_stats_df = pd.DataFrame(secondary_stats)

# Print the first rows of the main datasets
print("\nFirst rows of Carbon Intensity:")
print(secondary_df[['kg_CO2/kWh']].head())

print("\nFirst rows of Electricity Pricing:")
print(secondary_df[['Electricity Pricing [$]']].head())

print("\nFirst rows of Direct Solar Radiation:")
print(secondary_df[['Direct Solar Radiation [W/m2]']].head())

# Print the last rows of the main datasets
print("\nLast rows of Carbon Intensity:")
print(secondary_df[['kg_CO2/kWh']].tail())

print("\nLast rows of Electricity Pricing:")
print(secondary_df[['Electricity Pricing [$]']].tail())

print("\nLast rows of Direct Solar Radiation:")
print(secondary_df[['Direct Solar Radiation [W/m2]']].tail())

# Print KL divergence results
print("\nKL Divergence between Selected Days and Overall Dataset:")
print(kl_df)

# Print statistics of secondary_df
print("\nStatistics for Selected Days:")
print(secondary_stats_df)

# Visualize KL Divergence results
plt.figure(figsize=(10, 6))
bars = plt.bar(kl_df['Variable'], kl_df['KL Divergence'], color='skyblue')
plt.xlabel('Variable')
plt.ylabel('KL Divergence')
plt.title('KL Divergence between Selected Days and Overall Dataset')
plt.xticks(rotation=45)
plt.ylim(0, max(kl_df['KL Divergence']) * 1.1)

# Add values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.6f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
