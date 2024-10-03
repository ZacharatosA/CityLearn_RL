import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define paths and columns
data_path = 'Dataset/'

# Environmental columns
carbon_intensity_columns = ['kg_CO2/kWh']
pricing_columns = ['Electricity Pricing [$]']
weather_columns = ['Direct Solar Radiation [W/m2]']
environmental_columns = carbon_intensity_columns + pricing_columns + weather_columns

# Building columns
building_columns = ['Equipment Electric Power [kWh]', 'Solar Generation [W/kW]']

# Variables to control plotting
Figures = True      # Set to False to disable time series plots
DevFigures = True   # Set to False to disable mean and std plots

start = 1345  
buildings = [8,14,5,4 ] 


end = start + 167

# Function to load environmental data
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
        data = pd.read_csv(
            file_path,
            usecols=[column],
            skiprows=range(1, start_timestep + 1),
            nrows=nrows
        )
        data_frames.append(data.reset_index(drop=True))
    combined_data = pd.concat(data_frames, axis=1)
    return combined_data

# Function to load building data
def load_building_data(building_id, columns, start_timestep, end_timestep):
    nrows = end_timestep - start_timestep
    building_file = f"{data_path}Building_{building_id}.csv"
    try:
        building_data = pd.read_csv(
            building_file,
            usecols=columns,
            skiprows=range(1, start_timestep + 1),
            nrows=nrows
        )
        return building_data.reset_index(drop=True)
    except FileNotFoundError:
        print(f"File {building_file} not found. Please check the path.")
        return pd.DataFrame(columns=columns)

# Load environmental data
env_data = load_environmental_data(environmental_columns, start, end)

# Compute mean and std for environmental data
env_mean = env_data.mean()
env_std = env_data.std()

# Print mean and std for environmental data
print("Environmental Data Statistics:")
for column in environmental_columns:
    print(f"Mean of {column}: {env_mean[column]:.4f}")
    print(f"Std of {column}: {env_std[column]:.4f}")
    print("-" * 40)

# Plotting environmental data if Figures is True
if Figures:
    for column in environmental_columns:
        plt.figure(figsize=(10, 6))
        plt.plot(env_data[column])
        plt.title(f'{column} over Time')
        plt.xlabel('Time Step')
        plt.ylabel(column)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Plotting mean and std for environmental data if DevFigures is True
if DevFigures:
    # Plotting mean values
    plt.figure(figsize=(8, 6))
    plt.bar(environmental_columns, env_mean[environmental_columns], yerr=env_std[environmental_columns], capsize=5)
    plt.title('Mean and Std of Environmental Variables')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Initialize dictionaries to store mean and std for building data
building_stats = {}

# Loop over each building and compute statistics
for building_id in buildings:
    building_data = load_building_data(building_id, building_columns, start, end)
    if not building_data.empty:
        mean_values = building_data.mean()
        std_values = building_data.std()
        building_stats[building_id] = {
            'mean': mean_values,
            'std': std_values
        }
        # Print mean and std for each building
        print(f"Building {building_id} Statistics:")
        for column in building_columns:
            print(f"Mean of {column}: {mean_values[column]:.4f}")
            print(f"Std of {column}: {std_values[column]:.4f}")
        print("-" * 40)
        
        # Plotting building data if Figures is True
        if Figures:
            for column in building_columns:
                plt.figure(figsize=(10, 6))
                plt.plot(building_data[column])
                plt.title(f'Building {building_id} - {column} over Time')
                plt.xlabel('Time Step')
                plt.ylabel(column)
                plt.grid(True)
                plt.tight_layout()
                plt.show()
    else:
        print(f"No data available for Building {building_id}.")

# Plotting mean and std for building data if DevFigures is True
if DevFigures and building_stats:
    # Prepare data for plotting
    for column in building_columns:
        means = [building_stats[b]['mean'][column] for b in buildings if b in building_stats]
        stds = [building_stats[b]['std'][column] for b in buildings if b in building_stats]
        b_ids = [b for b in buildings if b in building_stats]
        
        # 
