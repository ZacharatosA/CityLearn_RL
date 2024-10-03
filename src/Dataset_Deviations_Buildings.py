import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

Figures = True  # Set to False to disable plotting

# Define paths and columns
data_path = 'Dataset/'
building_columns = ['Equipment Electric Power [kWh]', 'Solar Generation [W/kW]']

# Buildings to include (exclude 12 and 15)
building_ids = [i for i in range(1, 18) if i not in [12, 15]]

# Initialize lists to hold data from all buildings
equip_power_list = []
solar_gen_list = []

# Load data for each building and append to the lists
for building_id in building_ids:
    building_file = f"{data_path}Building_{building_id}.csv"
    try:
        building_data = pd.read_csv(building_file, usecols=building_columns)
        # Add building ID
        building_data['Building_ID'] = building_id
        # Append data to the respective lists
        equip_power_list.append(building_data[['Equipment Electric Power [kWh]', 'Building_ID']])
        solar_gen_list.append(building_data[['Solar Generation [W/kW]', 'Building_ID']])
    except FileNotFoundError:
        print(f"File {building_file} could not be found. Please check the path.")
        continue

# Concatenate the dataframes once outside the loop
equip_power_df = pd.concat(equip_power_list, ignore_index=True)
solar_gen_df = pd.concat(solar_gen_list, ignore_index=True)

# Function to compute average probability density
def compute_average_density(data, column, bins=50):
    # Compute histograms for each building
    densities = {}
    for building_id in building_ids:
        building_data = data[data['Building_ID'] == building_id][column]
        counts, bin_edges = np.histogram(building_data, bins=bins, density=True)
        densities[building_id] = counts
    # Compute average density
    all_densities = np.array(list(densities.values()))
    average_density = np.mean(all_densities, axis=0)
    return average_density, bin_edges

# Compute average densities
equip_avg_density, equip_bin_edges = compute_average_density(equip_power_df, 'Equipment Electric Power [kWh]')
solar_avg_density, solar_bin_edges = compute_average_density(solar_gen_df, 'Solar Generation [W/kW]')

# Function to compute KL divergence between a building's distribution and the average distribution
def compute_kl_divergences(data, column, average_density, bin_edges):
    kl_divergences = {}
    for building_id in building_ids:
        building_data = data[data['Building_ID'] == building_id][column]
        counts, _ = np.histogram(building_data, bins=bin_edges, density=True)
        # Add epsilon to avoid zeros
        epsilon = 1e-10
        counts += epsilon
        average_density_adjusted = average_density + epsilon
        # Normalize distributions
        building_dist = counts / counts.sum()
        avg_dist = average_density_adjusted / average_density_adjusted.sum()
        # Compute KL divergence
        kl_div = np.sum(rel_entr(building_dist, avg_dist))
        kl_divergences[building_id] = kl_div
    return kl_divergences

# Compute KL divergences
equip_kl_divs = compute_kl_divergences(equip_power_df, 'Equipment Electric Power [kWh]', equip_avg_density, equip_bin_edges)
solar_kl_divs = compute_kl_divergences(solar_gen_df, 'Solar Generation [W/kW]', solar_avg_density, solar_bin_edges)

# Compute average KL divergence for each building
average_kl_divs = {}
for building_id in building_ids:
    avg_kl = (equip_kl_divs[building_id] + solar_kl_divs[building_id]) / 2
    average_kl_divs[building_id] = avg_kl

# **Compute Mean, Std, and Mean Deviation for each building**
# Initialize dictionaries to store statistics
mean_values = {}
std_values = {}
mean_dev_values = {}

for building_id in building_ids:
    # Equipment Electric Power statistics
    equip_data = equip_power_df[equip_power_df['Building_ID'] == building_id]['Equipment Electric Power [kWh]']
    equip_mean = equip_data.mean()
    equip_std = equip_data.std()
    equip_mean_dev = (equip_data - equip_mean).abs().mean()
    
    # Solar Generation statistics
    solar_data = solar_gen_df[solar_gen_df['Building_ID'] == building_id]['Solar Generation [W/kW]']
    solar_mean = solar_data.mean()
    solar_std = solar_data.std()
    solar_mean_dev = (solar_data - solar_mean).abs().mean()
    
    # Average statistics
    avg_mean = (equip_mean + solar_mean) / 2
    avg_std = (equip_std + solar_std) / 2
    avg_mean_dev = (equip_mean_dev + solar_mean_dev) / 2
    
    # Store the average statistics
    mean_values[building_id] = avg_mean
    std_values[building_id] = avg_std
    mean_dev_values[building_id] = avg_mean_dev

# Create a DataFrame with all KL divergences and statistics
kl_df = pd.DataFrame({
    'Building_ID': building_ids,
    'Equip_KL': [equip_kl_divs[bid] for bid in building_ids],
    'Solar_KL': [solar_kl_divs[bid] for bid in building_ids],
    'Average_KL': [average_kl_divs[bid] for bid in building_ids],
    'Mean': [mean_values[bid] for bid in building_ids],
    'Std_Dev': [std_values[bid] for bid in building_ids],
    'Mean_Deviation': [mean_dev_values[bid] for bid in building_ids]
})

# Sort kl_df by Building_ID
kl_df_sorted_by_id = kl_df.sort_values(by='Building_ID').reset_index(drop=True)

# **Change the title of the printed table**
print("\nKL and Deviations for each Building:")
print(kl_df_sorted_by_id)

# Create ranking lists
equip_kl_ranking = kl_df[['Building_ID', 'Equip_KL']].sort_values(by='Equip_KL')['Building_ID'].tolist()
solar_kl_ranking = kl_df[['Building_ID', 'Solar_KL']].sort_values(by='Solar_KL')['Building_ID'].tolist()
average_kl_ranking = kl_df[['Building_ID', 'Average_KL']].sort_values(by='Average_KL')['Building_ID'].tolist()

# Print the rankings in the requested format
print("\nEquip_KL Ranking:")
print(f"Equip_KL: {equip_kl_ranking}")

print("\nSolar_KL Ranking:")
print(f"Solar_KL: {solar_kl_ranking}")

print("\nAverage_KL Ranking:")
print(f"Average_KL: {average_kl_ranking}")

# **Plotting the KL divergences if Figures is True**
if Figures:
    plt.figure(figsize=(12, 6))
    index = np.arange(len(building_ids))
    bar_width = 0.25

    # Sort the DataFrame by Building_ID for consistent plotting
    kl_df_sorted = kl_df.sort_values(by='Building_ID').reset_index(drop=True)

    plt.bar(index, kl_df_sorted['Equip_KL'], bar_width, label='Equip_KL')
    plt.bar(index + bar_width, kl_df_sorted['Solar_KL'], bar_width, label='Solar_KL')
    plt.bar(index + 2 * bar_width, kl_df_sorted['Average_KL'], bar_width, label='Average_KL')

    plt.xlabel('Buildings')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence per Building')
    plt.xticks(index + bar_width, kl_df_sorted['Building_ID'])
    plt.legend()
    plt.tight_layout()
    plt.show()
