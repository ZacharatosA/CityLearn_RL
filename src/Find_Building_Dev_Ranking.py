import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr
import os


Figures = True  # Set to False to disable plotting

# Parameters
start_timestep = 7561

# Define paths and columns
data_path = 'Dataset/'
building_columns = ['Equipment Electric Power [kWh]', 'Solar Generation [W/kW]']



WEEK = 167
end_timestep = start_timestep + WEEK

# Function to calculate standard deviation difference
def calculate_std_diff(train, rest):
    """
    Calculates the absolute difference in standard deviation between training and rest data.
    """
    week_std = train.std()
    rest_std = rest.std()
    std_diff = abs(week_std - rest_std)
    return std_diff

# Function to calculate KL divergence with combined binning
def calculate_kl_divergence(train, rest, bins=50):
    """
    Calculates the Kullback-Leibler (KL) divergence between training and rest data distributions.
    """
    # Create bins based on combined training and rest data
    combined = np.concatenate([train, rest])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)
    
    # Calculate histograms
    train_counts, _ = np.histogram(train, bins=bin_edges, density=True)
    rest_counts, _ = np.histogram(rest, bins=bin_edges, density=True)
    
    # Add a small value to avoid zeros
    epsilon = 1e-10
    train_counts += epsilon
    rest_counts += epsilon
    
    # Normalize to sum to 1
    train_dist = train_counts / train_counts.sum()
    rest_dist = rest_counts / rest_counts.sum()
    
    # Calculate the KL Divergence
    kl_div = np.sum(rel_entr(train_dist, rest_dist))
    return kl_div

# Function to calculate standard deviation differences for buildings with normalization
def calculate_building_std_diffs(building_data, start_timestep, end_timestep):
    """
    Calculates the standard deviation differences for each column of a building after normalization.
    """
    std_diffs = {}
    # Normalize the data
    building_data_normalized = (building_data - building_data.mean()) / building_data.std()
    # Define train and rest data
    train_data = building_data_normalized.iloc[start_timestep:end_timestep+1]
    rest_data = pd.concat([building_data_normalized.iloc[:start_timestep], building_data_normalized.iloc[end_timestep+1:]])
    
    for column in building_columns:
        std_diff = calculate_std_diff(train_data[column], rest_data[column])
        std_diffs[column] = std_diff
    
    # Calculate average std_diff
    std_diffs['average_std_diff'] = np.mean(list(std_diffs.values()))
    return std_diffs

# Function to calculate KL divergence for buildings (using original data)
def calculate_building_kl_divergences(building_data, start_timestep, end_timestep):
    """
    Calculates the KL divergence for each column of a building.
    """
    kl_divs = {}
    train_data = building_data.iloc[start_timestep:end_timestep+1]
    rest_data = pd.concat([building_data.iloc[:start_timestep], building_data.iloc[end_timestep+1:]])
    
    for column in building_columns:
        kl_div = calculate_kl_divergence(train_data[column], rest_data[column], bins=50)
        kl_divs[column] = kl_div
    
    # Calculate average KL divergence
    kl_divs['average_kl_div'] = np.mean(list(kl_divs.values()))
    return kl_divs

# Initialize dictionaries to store results
building_std_diffs = {}
building_kl_divergences = {}

# Iterate over all buildings except 12 and 15
for i in range(1, 18):
    if i in [12, 15]:
        continue
    # Load building data
    building_file = f"{data_path}Building_{i}.csv"
    try:
        building_data = pd.read_csv(building_file, usecols=building_columns)
    except FileNotFoundError:
        print(f"File {building_file} could not be found. Please check the path.")
        continue
    
    # Calculate std_diff for the building (with normalization)
    std_diffs = calculate_building_std_diffs(building_data, start_timestep, end_timestep)
    building_std_diffs[i] = std_diffs
    
    # Calculate KL divergence for the building (using original data)
    kl_divs = calculate_building_kl_divergences(building_data, start_timestep, end_timestep)
    building_kl_divergences[i] = kl_divs

# Convert dictionaries to DataFrames
std_diff_df = pd.DataFrame(building_std_diffs).T  # Buildings as rows
kl_div_df = pd.DataFrame(building_kl_divergences).T  # Buildings as rows

# Rename columns for clarity
std_diff_df.rename(columns={'average_std_diff': 'average_std_diff'}, inplace=True)
kl_div_df.rename(columns={'average_kl_div': 'average_kl_div'}, inplace=True)

# Sorting for std_diff
sorted_buildings_avg_std = std_diff_df.sort_values(by='average_std_diff')
sorted_buildings_equip_std = std_diff_df.sort_values(by='Equipment Electric Power [kWh]', ascending=True)
sorted_buildings_solar_std = std_diff_df.sort_values(by='Solar Generation [W/kW]', ascending=True)

# Sorting for KL divergence
sorted_buildings_avg_kl = kl_div_df.sort_values(by='average_kl_div')
sorted_buildings_equip_kl = kl_div_df.sort_values(by='Equipment Electric Power [kWh]', ascending=True)
sorted_buildings_solar_kl = kl_div_df.sort_values(by='Solar Generation [W/kW]', ascending=True)

# Extract sorted building IDs without spaces
def format_building_ids(building_ids):
    return '[' + ','.join(map(str, building_ids)) + ']'

sorted_building_ids_avg_std = sorted_buildings_avg_std.index.tolist()
sorted_building_ids_equip_std = sorted_buildings_equip_std.index.tolist()
sorted_building_ids_solar_std = sorted_buildings_solar_std.index.tolist()

sorted_building_ids_avg_kl = sorted_buildings_avg_kl.index.tolist()
sorted_building_ids_equip_kl = sorted_buildings_equip_kl.index.tolist()
sorted_building_ids_solar_kl = sorted_buildings_solar_kl.index.tolist()

# Sort building IDs in ascending order for detailed statistics
sorted_building_ids = sorted(building_std_diffs.keys())

# Print detailed information sorted by building ID
print("\nDetailed Statistics for Each Building (sorted by Building ID):")
for building_id in sorted_building_ids:
    std_info = building_std_diffs[building_id]
    kl_info = building_kl_divergences[building_id]
    print(f"\nBuilding {building_id}:")
    print(f"  Equipment Electric Power [kWh]_std_diff:           {std_info['Equipment Electric Power [kWh]']:<10.4f}")
    print(f"  Solar Generation [W/kW]_std_diff:                   {std_info['Solar Generation [W/kW]']:<10.4f}")
    print(f"  Average STD Difference:                             {std_info['average_std_diff']:<10.4f}")
    print(f"  Equipment Electric Power [kWh]_KL_divergence:      {kl_info['Equipment Electric Power [kWh]']:<10.4f}")
    print(f"  Solar Generation [W/kW]_KL_divergence:              {kl_info['Solar Generation [W/kW]']:<10.4f}")
    print(f"  Average KL Divergence:                              {kl_info['average_kl_div']:<10.4f}")

# Print sorted lists with no spaces between commas and spacing between tables
print("\nEquipment Electric Power STD Difference:")
print(format_building_ids(sorted_building_ids_equip_std))

print("\nEquipment Electric Power KL Divergence:")
print(format_building_ids(sorted_building_ids_equip_kl))

print("\nSolar Generation STD Difference:")
print(format_building_ids(sorted_building_ids_solar_std))

print("\nSolar Generation KL Divergence:")
print(format_building_ids(sorted_building_ids_solar_kl))

print("\nAverage STD Difference:")
print(format_building_ids(sorted_building_ids_avg_std))

print("\nAverage KL Divergence:")
print(format_building_ids(sorted_building_ids_avg_kl))

# **Plotting the results if Figures is True**
if Figures:
    # Generate a bar chart for each building showing the metrics
    # Create lists to hold the metrics
    building_ids = sorted_building_ids
    equip_std = [building_std_diffs[b]['Equipment Electric Power [kWh]'] for b in building_ids]
    solar_std = [building_std_diffs[b]['Solar Generation [W/kW]'] for b in building_ids]
    avg_std = [building_std_diffs[b]['average_std_diff'] for b in building_ids]
    equip_kl = [building_kl_divergences[b]['Equipment Electric Power [kWh]'] for b in building_ids]
    solar_kl = [building_kl_divergences[b]['Solar Generation [W/kW]'] for b in building_ids]
    avg_kl = [building_kl_divergences[b]['average_kl_div'] for b in building_ids]
    
    # Calculate the overall average KL divergence and average STD difference
    overall_avg_kl = np.mean(avg_kl)
    overall_avg_std = np.mean(avg_std)
    
    # Set the width of each bar
    bar_width = 0.15
    
    # Set positions of bars on the x-axis
    r1 = np.arange(len(building_ids))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    r6 = [x + bar_width for x in r5]
    
    # Create the plot
    plt.figure(figsize=(20, 10))
    
    # Make the bars
    plt.bar(r1, equip_std, color='b', width=bar_width, edgecolor='grey', label='Equip STD')
    plt.bar(r2, solar_std, color='g', width=bar_width, edgecolor='grey', label='Solar STD')
    plt.bar(r3, avg_std, color='c', width=bar_width, edgecolor='grey', label='Average STD')
    plt.bar(r4, equip_kl, color='r', width=bar_width, edgecolor='grey', label='Equip KL')
    plt.bar(r5, solar_kl, color='y', width=bar_width, edgecolor='grey', label='Solar KL')
    plt.bar(r6, avg_kl, color='m', width=bar_width, edgecolor='grey', label='Average KL')
    
    # Add horizontal lines for overall averages
    plt.axhline(y=overall_avg_std, color='k', linestyle='--', linewidth=2, label=f'Overall Average STD ({overall_avg_std:.4f})')
    plt.axhline(y=overall_avg_kl, color='orange', linestyle='-.', linewidth=2, label=f'Overall Average KL ({overall_avg_kl:.4f})')
    
    # Add labels and title
    plt.xlabel('Building ID', fontweight='bold', fontsize=15)
    plt.ylabel('Value', fontweight='bold', fontsize=15)
    plt.title('Standard Deviation Differences and KL Divergences per Building', fontsize=18)
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + 2.5 * bar_width for r in range(len(building_ids))], building_ids)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
