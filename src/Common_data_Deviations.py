import os  # Εισαγωγή της βιβλιοθήκης os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define data path
data_path = 'Dataset/'

# Define the columns we are interested in
carbon_intensity_columns = ['kg_CO2/kWh']
pricing_columns = ['Electricity Pricing [$]']
weather_columns = ['Direct Solar Radiation [W/m2]']

# Load the data
carbon_intensity = pd.read_csv(f"{data_path}carbon_intensity.csv", usecols=carbon_intensity_columns)
pricing = pd.read_csv(f"{data_path}pricing.csv", usecols=pricing_columns)
weather = pd.read_csv(f"{data_path}weather.csv", usecols=weather_columns)

# Calculate mean, standard deviation, and mean deviation
mean_carbon = carbon_intensity.mean().values[0]
std_carbon = carbon_intensity.std().values[0]
mean_dev_carbon = (abs(carbon_intensity['kg_CO2/kWh'] - mean_carbon)).mean()

mean_pricing = pricing.mean().values[0]
std_pricing = pricing.std().values[0]
mean_dev_pricing = (abs(pricing['Electricity Pricing [$]'] - mean_pricing)).mean()

mean_weather = weather.mean().values[0]
std_weather = weather.std().values[0]
mean_dev_weather = (abs(weather['Direct Solar Radiation [W/m2]'] - mean_weather)).mean()

# Create X-axis (timesteps)
timesteps = np.arange(1, len(carbon_intensity) + 1)

# List of data and parameters for each variable
variables = [
    {
        'data': carbon_intensity['kg_CO2/kWh'],
        'mean': mean_carbon,
        'std': std_carbon,
        'mean_dev': mean_dev_carbon,
        'ylabel': 'kg_CO2/kWh',
        'title': 'Carbon Intensity over Timesteps',
        'color': 'blue'
    },
    {
        'data': pricing['Electricity Pricing [$]'],
        'mean': mean_pricing,
        'std': std_pricing,
        'mean_dev': mean_dev_pricing,
        'ylabel': 'Electricity Pricing [$]',
        'title': 'Electricity Pricing over Timesteps',
        'color': 'purple'
    },
    {
        'data': weather['Direct Solar Radiation [W/m2]'],
        'mean': mean_weather,
        'std': std_weather,
        'mean_dev': mean_dev_weather,
        'ylabel': 'Direct Solar Radiation [W/m2]',
        'title': 'Direct Solar Radiation over Timesteps',
        'color': 'brown'
    }
]

# Define the path for saving figures
figures_path = 'Figures/'

# Create the "Figures" directory if it doesn't exist
if not os.path.exists(figures_path):
    os.makedirs(figures_path)

# Create and save separate plots for each variable
for var in variables:
    plt.figure(figsize=(15, 8), dpi=150)  # Reduced size and increased resolution
    
    # Plot the data
    plt.plot(timesteps, var['data'], label='Data', color=var['color'])
    
    # Add lines for Mean, Mean + Std, Mean - Std
    plt.axhline(var['mean'], color='red', linestyle='--', label=f'Mean: {var["mean"]:.2f}')
    plt.axhline(var['mean'] + var['std'], color='green', linestyle='--', label=f'Mean + Std: {var["mean"] + var["std"]:.2f}')
    plt.axhline(var['mean'] - var['std'], color='orange', linestyle='--', label=f'Mean - Std: {var["mean"] - var["std"]:.2f}')
    
    # Add lines for Mean Deviation
    plt.axhline(var['mean'] + var['mean_dev'], color='cyan', linestyle=':', label=f'Mean + Mean Dev: {var["mean"] + var["mean_dev"]:.2f}')
    plt.axhline(var['mean'] - var['mean_dev'], color='magenta', linestyle=':', label=f'Mean - Mean Dev: {var["mean"] - var["mean_dev"]:.2f}')
    
    # Graph settings
    plt.title(var['title'], fontsize=16)
    plt.xlabel('Timesteps (Hours)', fontsize=14)
    plt.ylabel(var['ylabel'], fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Create the filename and path
    filename = var['title'].replace(" ", "_").lower() + ".png"
    filepath = os.path.join(figures_path, filename)
    
    # Save the plot as a PNG with high resolution inside the "Figures" folder
    plt.savefig(filepath, dpi=300)
    
    # Show the plot
    plt.show()

# Print the mean, standard deviation, and mean deviation values
print("Mean, Standard Deviation, and Mean Deviation Values:")
print(f"Carbon Intensity - Mean: {mean_carbon:.2f}, Std Dev: {std_carbon:.2f}, Mean Dev: {mean_dev_carbon:.2f}")
print(f"Electricity Pricing - Mean: {mean_pricing:.2f}, Std Dev: {std_pricing:.2f}, Mean Dev: {mean_dev_pricing:.2f}")
print(f"Direct Solar Radiation - Mean: {mean_weather:.2f}, Std Dev: {std_weather:.2f}, Mean Dev: {mean_dev_weather:.2f}")
