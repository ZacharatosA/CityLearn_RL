import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

# Define the data path
data_path = 'Dataset/'

# Columns to be used
carbon_intensity_columns = ['kg_CO2/kWh']
pricing_columns = ['Electricity Pricing [$]']
weather_columns = ['Direct Solar Radiation [W/m2]']
environmental_columns = carbon_intensity_columns + pricing_columns + weather_columns
building_columns = ['Equipment Electric Power [kWh]', 'Solar Generation [W/kW]']
week = 167

# Env 1
start1 = 1345   
end1 = start1 + week   
Buildings1 = [10,9,13,16]

#Env 2
start2 = 1345
end2 = start2 + week 
Buildings2 = [16,2,8,14]

# Function to load environmental data for a time period
def load_environmental_data(columns, start_timestep, end_timestep):
    nrows = end_timestep - start_timestep
    data_frames = []
    for column in columns:
        file_name = ''
        if column in carbon_intensity_columns:
            file_name = 'carbon_intensity.csv'
        elif column in pricing_columns:
            file_name = 'pricing.csv'
        elif column in weather_columns:
            file_name = 'weather.csv'
        else:
            continue
        file_path = data_path + file_name
        data = pd.read_csv(file_path, usecols=[column], skiprows=range(1, start_timestep + 1), nrows=nrows)
        data_frames.append(data.reset_index(drop=True))
    combined_data = pd.concat(data_frames, axis=1)
    return combined_data

# Function to load building data for a list of buildings and a time period
def load_buildings_data(buildings_list, columns, start_timestep, end_timestep):
    nrows = end_timestep - start_timestep
    data_list = []
    for building_id in buildings_list:
        building_file = f"{data_path}Building_{building_id}.csv"
        try:
            building_data = pd.read_csv(building_file, usecols=columns, skiprows=range(1, start_timestep + 1), nrows=nrows)
            data_list.append(building_data)
        except FileNotFoundError:
            print(f"The file {building_file} could not be found. Please check the path.")
            continue
    if data_list:
        combined_data = pd.concat(data_list, ignore_index=True)
    else:
        combined_data = pd.DataFrame(columns=columns)
    return combined_data

# Function to calculate KL divergence between two datasets for specific columns
def calculate_kl_divergences(data1, data2, columns, bins=50):
    kl_divergences = {}
    for column in columns:
        data1_column = data1[column].dropna()
        data2_column = data2[column].dropna()

        # Create bins based on combined data
        combined_data = np.concatenate([data1_column, data2_column])
        bin_edges = np.histogram_bin_edges(combined_data, bins=bins)

        # Calculate histograms
        counts1, _ = np.histogram(data1_column, bins=bin_edges, density=True)
        counts2, _ = np.histogram(data2_column, bins=bin_edges, density=True)

        # Add a small value to avoid zeros
        epsilon = 1e-10
        counts1 += epsilon
        counts2 += epsilon

        # Normalize to create probability distributions
        dist1 = counts1 / counts1.sum()
        dist2 = counts2 / counts2.sum()

        # Calculate KL Divergence
        kl_div = np.sum(rel_entr(dist1, dist2))
        kl_divergences[column] = kl_div
    return kl_divergences

# Load environmental data for the two time periods
env_data1 = load_environmental_data(environmental_columns, start1, end1)
env_data2 = load_environmental_data(environmental_columns, start2, end2)

# Load building data for the two groups of buildings and time periods
building_data1 = load_buildings_data(Buildings1, building_columns, start1, end1)
building_data2 = load_buildings_data(Buildings2, building_columns, start2, end2)

# Calculate KL divergence for environmental data
kl_divs_env = calculate_kl_divergences(env_data1, env_data2, environmental_columns)

# Calculate KL divergence for building data
kl_divs_buildings = calculate_kl_divergences(building_data1, building_data2, building_columns)

# Calculate averages
average_env_kl = np.mean(list(kl_divs_env.values()))
average_building_kl = np.mean(list(kl_divs_buildings.values()))

# Print the results
print("KL divergences for environmental data:")
for column, kl_value in kl_divs_env.items():
    print(f"KL divergence for {column}: {kl_value:.4f}")
print(f"Average KL divergence for environmental data: {average_env_kl:.4f}")

print("\nKL divergences for building data:")
for column, kl_value in kl_divs_buildings.items():
    print(f"KL divergence for {column}: {kl_value:.4f}")
print(f"Average KL divergence for building data: {average_building_kl:.4f}")
