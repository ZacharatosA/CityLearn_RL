import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from math import log2


Week = 167
#-----------------
train_start_row = 0
train_end_row =  train_start_row + Week
test_start_row = 0
test_end_row = 8759

# Buildings to plot
selected_buildings = [1]  
data_path = 'Dataset/'

# Columns to read for each dataset
carbon_intensity_columns = ['kg_CO2/kWh']
pricing_columns = ['Electricity Pricing [$]']
weather_columns = ['Direct Solar Radiation [W/m2]']
building_columns = ['Equipment Electric Power [kWh]', 'Solar Generation [W/kW]']

# Function to load specific rows from the datasets
def load_data_with_rows(filepath, columns, start_row, end_row):
    nrows = end_row - start_row
    if nrows < 0:
        raise ValueError("'nrows' must be an integer >=0")
    data = pd.read_csv(filepath, usecols=columns, skiprows=range(1, start_row + 1), nrows=nrows)
    return data

# Load the datasets with specific rows for training and testing separately
carbon_intensity_train = load_data_with_rows(data_path + 'carbon_intensity.csv', carbon_intensity_columns, train_start_row, train_end_row)
carbon_intensity_test = load_data_with_rows(data_path + 'carbon_intensity.csv', carbon_intensity_columns, test_start_row, test_end_row)
pricing_train = load_data_with_rows(data_path + 'pricing.csv', pricing_columns, train_start_row, train_end_row)
pricing_test = load_data_with_rows(data_path + 'pricing.csv', pricing_columns, test_start_row, test_end_row)
weather_train = load_data_with_rows(data_path + 'weather.csv', weather_columns, train_start_row, train_end_row)
weather_test = load_data_with_rows(data_path + 'weather.csv', weather_columns, test_start_row, test_end_row)

# Initialize all_data with combined datasets
all_data = []

# Load building datasets with specific rows for training and testing separately
def load_building_data(building_number, columns, train_start_row, train_end_row, test_start_row, test_end_row):
    building_data_path = data_path + f'Building_{building_number}.csv'
    building_train = load_data_with_rows(building_data_path, columns, train_start_row, train_end_row)
    building_test = load_data_with_rows(building_data_path, columns, test_start_row, test_end_row)
    return building_train, building_test

# Load and split building data
num_buildings = 17  
for building_number in range(1, num_buildings + 1):
    building_train, building_test = load_building_data(building_number, building_columns, train_start_row, train_end_row, test_start_row, test_end_row)
    all_data.extend([building_train, building_test])

# Combine all data into a single DataFrame
All = pd.concat(all_data, sort=True).reset_index(drop=True)

# Functions to calculate probabilities with binning
def prob_distribution(data, column, bins):
    binned = pd.cut(data[column], bins, include_lowest=True, right=False)
    probabilities = binned.value_counts(normalize=True, sort=False)
    return probabilities

# Function to plot distributions side by side
def distribution_plot(train_data, test_data, column, bins, log_scale=False):
    # Combine train and test data to determine bin edges
    combined_data = pd.concat([train_data[column], test_data[column]], ignore_index=True)
    bin_edges = np.histogram_bin_edges(combined_data, bins=bins)
    
    train_probs = prob_distribution(train_data, column, bin_edges)
    test_probs = prob_distribution(test_data, column, bin_edges)
    
    # Ensure that both train_probs and test_probs have the same index
    all_bins = train_probs.index.union(test_probs.index)
    train_probs = train_probs.reindex(all_bins, fill_value=0)
    test_probs = test_probs.reindex(all_bins, fill_value=0)
    
    # Add epsilon to avoid zeros when using log scale
    epsilon = 1e-10
    train_probs += epsilon
    test_probs += epsilon
    
    # Get min and max probabilities to set the same y-axis limits
    min_prob = min(train_probs.min(), test_probs.min())
    max_prob = max(train_probs.max(), test_probs.max())
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    font = {'family': 'serif', 'color': 'blue', 'size': 15}
    
    train_title = f"Train ({train_start_row}-{train_end_row})"
    test_title = f"Test ({test_start_row}-{test_end_row})"

    axes[0].bar(train_probs.index.astype(str), train_probs.values, width=0.8)
    axes[0].set(xlabel=column, ylabel='Probability')
    if log_scale:
        axes[0].set_yscale('log')
    axes[0].set_ylim(min_prob, max_prob)
    axes[0].set_title(train_title, fontdict=font)
    axes[0].tick_params(axis='x', rotation=90)

    axes[1].bar(test_probs.index.astype(str), test_probs.values, width=0.8)
    axes[1].set(xlabel=column, ylabel='Probability')
    if log_scale:
        axes[1].set_yscale('log')
    axes[1].set_ylim(min_prob, max_prob)
    axes[1].set_title(test_title, fontdict=font)
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

# Function to plot distributions side by side for buildings
def building_distribution_plot(building_number, bins):
    building_train, building_test = load_building_data(building_number, building_columns, train_start_row, train_end_row, test_start_row, test_end_row)
    
    # Combine train and test data for each column to determine bin edges
    combined_equipment = pd.concat([building_train['Equipment Electric Power [kWh]'], building_test['Equipment Electric Power [kWh]']], ignore_index=True)
    combined_solar = pd.concat([building_train['Solar Generation [W/kW]'], building_test['Solar Generation [W/kW]']], ignore_index=True)
    
    bin_edges_equipment = np.histogram_bin_edges(combined_equipment, bins=bins)
    bin_edges_solar = np.histogram_bin_edges(combined_solar, bins=bins)
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle(f"Building {building_number}", fontsize=20, fontweight='bold')
    font = {'family': 'serif', 'color': 'blue', 'size': 15}
    
    train_title = f"Train ({train_start_row}-{train_end_row})"
    test_title = f"Test ({test_start_row}-{test_end_row})"

    columns = ['Equipment Electric Power [kWh]', 'Solar Generation [W/kW]']
    bin_edges_list = [bin_edges_equipment, bin_edges_solar]
    for i, (column, bin_edges) in enumerate(zip(columns, bin_edges_list)):
        train_probs = prob_distribution(building_train, column, bin_edges)
        test_probs = prob_distribution(building_test, column, bin_edges)
        
        # Ensure that both train_probs and test_probs have the same index
        all_bins = train_probs.index.union(test_probs.index)
        train_probs = train_probs.reindex(all_bins, fill_value=0)
        test_probs = test_probs.reindex(all_bins, fill_value=0)
        
        # Add epsilon to avoid zeros when using log scale
        epsilon = 1e-10
        train_probs += epsilon
        test_probs += epsilon
        
        # Get min and max probabilities to set the same y-axis limits
        min_prob = min(train_probs.min(), test_probs.min())
        max_prob = max(train_probs.max(), test_probs.max())

        # Ensure min_prob and max_prob are not NaN or Inf
        if np.isnan(min_prob) or np.isinf(min_prob):
            min_prob = 0
        if np.isnan(max_prob) or np.isinf(max_prob):
            max_prob = 1
        
        row = i
        col = 0
        axes[row, col].bar(train_probs.index.astype(str), train_probs.values, width=0.8)
        axes[row, col].set(xlabel=column, ylabel='Probability')
        if column == 'Solar Generation [W/kW]':
            axes[row, col].set_yscale('log')
        axes[row, col].set_ylim(min_prob, max_prob)
        axes[row, col].set_title(train_title, fontdict=font)
        axes[row, col].tick_params(axis='x', rotation=90)

        axes[row, col + 1].bar(test_probs.index.astype(str), test_probs.values, width=0.8)
        axes[row, col + 1].set(xlabel=column, ylabel='Probability')
        if column == 'Solar Generation [W/kW]':
            axes[row, col + 1].set_yscale('log')
        axes[row, col + 1].set_ylim(min_prob, max_prob)
        axes[row, col + 1].set_title(test_title, fontdict=font)
        axes[row, col + 1].tick_params(axis='x', rotation=90)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Plot distributions for Carbon_intensity
for column in carbon_intensity_train.columns:
    distribution_plot(carbon_intensity_train, carbon_intensity_test, column, bins=20)

# Plot distributions for Pricing
for column in pricing_train.columns:
    distribution_plot(pricing_train, pricing_test, column, bins=20)

# Plot distributions for Weather
for column in weather_train.columns:
    log_scale = column == 'Direct Solar Radiation [W/m2]'
    distribution_plot(weather_train, weather_test, column, bins=20, log_scale=log_scale)

# Plot distributions for each selected building
for building_number in selected_buildings:
    building_distribution_plot(building_number, bins=20)
