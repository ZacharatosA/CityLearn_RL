import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

Figures = True  # Set to False to disable plotting

# Define paths and columns
data_path = 'Dataset/'
carbon_intensity_columns = ['kg_CO2/kWh']
pricing_columns = ['Electricity Pricing [$]']
weather_columns = ['Direct Solar Radiation [W/m2]']

# Load datasets
carbon_intensity = pd.read_csv(f"{data_path}carbon_intensity.csv", usecols=carbon_intensity_columns)
pricing = pd.read_csv(f"{data_path}pricing.csv", usecols=pricing_columns)
weather = pd.read_csv(f"{data_path}weather.csv", usecols=weather_columns)

# Combine datasets into one DataFrame
data = pd.concat([carbon_intensity, pricing, weather], axis=1)

# **Normalization of the data**
data_normalized = (data - data.mean()) / data.std()

# Initialize parameters
train_period = 168  # Number of timesteps representing a week
End = len(data_normalized)
statistics = []

# Function to calculate KL divergence with combined binning
def calculate_kl_divergence(train, rest, bins=50):
    combined = np.concatenate([train, rest])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)
    train_counts, _ = np.histogram(train, bins=bin_edges, density=True)
    rest_counts, _ = np.histogram(rest, bins=bin_edges, density=True)
    epsilon = 1e-10
    train_counts += epsilon
    rest_counts += epsilon
    train_dist = train_counts / train_counts.sum()
    rest_dist = rest_counts / rest_counts.sum()
    kl_div = np.sum(rel_entr(train_dist, rest_dist))
    return kl_div

# Function to calculate standard deviation difference
def calculate_std_diff(train, rest):
    week_std = train.std()
    rest_std = rest.std()
    std_diff = abs(week_std - rest_std)
    return std_diff

# Iterate over each week in the dataset starting from timestep 0
for start in range(0, End, train_period):
    end = start + train_period  # Non-inclusive end index for iloc
    if end > End:
        end = End  # Ensure we don't go beyond the data
    week_data = data_normalized.iloc[start:end]
    rest_data = pd.concat([data_normalized.iloc[:start], data_normalized.iloc[end:]])
    
    stats = {}
    stats['start'] = start + 1  # Adjust to 1-based indexing
    stats['end'] = end
    
    # Calculate std_diff and KL_divergence for each column
    for column in data_normalized.columns:
        # Standard Deviation Difference
        std_diff = calculate_std_diff(week_data[column], rest_data[column])
        stats[f'{column}_std_diff'] = std_diff
        
        # KL Divergence (using original data)
        week_data_original = data[column].iloc[start:end].values
        rest_data_original = pd.concat([data[column].iloc[:start], data[column].iloc[end:]]).values
        kl_div = calculate_kl_divergence(week_data_original, rest_data_original, bins=50)
        stats[f'{column}_KL_divergence'] = kl_div
    
    # Calculate average std_diff and average KL_divergence
    std_diffs = [v for k, v in stats.items() if 'std_diff' in k]
    kl_divs = [v for k, v in stats.items() if 'KL_divergence' in k]
    stats['average_std_diff'] = np.mean(std_diffs)
    stats['average_KL_divergence'] = np.mean(kl_divs)
    
    # Calculate the mean and std for the week and add to stats
    week_means = week_data.mean()
    week_stds = week_data.std()
    stats['average_mean'] = week_means.mean()
    stats['average_std'] = week_stds.mean()
    
    statistics.append(stats)

# Convert to DataFrame
stats_df = pd.DataFrame(statistics)

# Create week number for plotting and identification
stats_df['week_number'] = stats_df.index + 1

# Rename columns for easier access
stats_df.rename(columns={
    'kg_CO2/kWh_KL_divergence': 'CO2_KL',
    'Electricity Pricing [$]_KL_divergence': 'Electricity_P_KL',
    'Direct Solar Radiation [W/m2]_KL_divergence': 'Direct_solar_KL',
    'kg_CO2/kWh_std_diff': 'CO2_std_diff',
    'Electricity Pricing [$]_std_diff': 'Electricity_P_std_diff',
    'Direct Solar Radiation [W/m2]_std_diff': 'Direct_solar_std_diff'
}, inplace=True)

# Sort weeks based on different criteria
sorted_weeks_std_diff = stats_df.sort_values(by='average_std_diff')
sorted_weeks_KL = stats_df.sort_values(by='average_KL_divergence')
sorted_weeks_avg_std = stats_df.sort_values(by='average_std')
sorted_weeks_KL_CO2 = stats_df.sort_values(by='CO2_KL')
sorted_weeks_KL_direct_solar = stats_df.sort_values(by='Direct_solar_KL')

# Extract sorted week IDs
std_diff_sorted_ids = sorted_weeks_std_diff['week_number'].astype(int).tolist()
KL_div_sorted_ids = sorted_weeks_KL['week_number'].astype(int).tolist()
avg_std_sorted_ids = sorted_weeks_avg_std['week_number'].astype(int).tolist()
KL_div_CO2_sorted_ids = sorted_weeks_KL_CO2['week_number'].astype(int).tolist()
KL_div_direct_solar_sorted_ids = sorted_weeks_KL_direct_solar['week_number'].astype(int).tolist()

# Print the five sorted lists
print("Sorted Weeks by Average Std Diff:     ", std_diff_sorted_ids)
print("Sorted Weeks by Average KL:           ", KL_div_sorted_ids)
print("Sorted Weeks by Average Std:          ", avg_std_sorted_ids)
print("Sorted Weeks by KL for kg_CO2:        ", KL_div_CO2_sorted_ids)
print("Sorted Weeks by KL for Solar Rad:     ", KL_div_direct_solar_sorted_ids)

# Calculate overall averages and medians
direct_solar_avg_std_diff = stats_df['Direct_solar_std_diff'].mean()
overall_median_std = stats_df['average_std_diff'].median()
overall_avg_std_diff = stats_df['average_std_diff'].mean()
overall_avg_KL_divergence = stats_df['average_KL_divergence'].mean()
overall_avg_mean = stats_df['average_mean'].mean()
overall_avg_std = stats_df['average_std'].mean()

# Print the ranking table based on Week Number without Rank
print("\nWeek     TS         Ave_KL    Ave_STD_Diff   Ave_Mean    Ave_STD     CO2_KL     Elec_P_KL     SolarRad_KL")
print("--------------------------------------------------------------------------------------------------------------")
sorted_by_week = stats_df.sort_values(by='week_number')
for row in sorted_by_week.itertuples():
    week_id = int(row.week_number)
    start = int(row.start)
    end = int(row.end)
    Ave_KL = row.average_KL_divergence
    Ave_STD_Diff = row.average_std_diff
    Ave_Mean = row.average_mean
    Ave_STD = row.average_std
    CO2_KL = row.CO2_KL
    Elec_P_KL = row.Electricity_P_KL
    SolarRad_KL = row.Direct_solar_KL
    
    print(f"Week {week_id:<3} {start}-{end:<7} {Ave_KL:<10.4f} {Ave_STD_Diff:<13.4f} {Ave_Mean:<10.4f} {Ave_STD:<10.4f} {CO2_KL:<9.4f} {Elec_P_KL:<14.4f} {SolarRad_KL:<14.4f}")

if Figures:
    # -------------------- First Figure: Standard Deviation Differences --------------------
    plt.figure(figsize=(20, 10))
    plt.plot(stats_df['week_number'], stats_df['Electricity_P_std_diff'], marker='o', label='Electricity Pricing [$]_std_diff')
    plt.plot(stats_df['week_number'], stats_df['CO2_std_diff'], marker='s', label='kg_CO2/kWh_std_diff')
    plt.plot(stats_df['week_number'], stats_df['Direct_solar_std_diff'], marker='^', label='Direct Solar Radiation [W/m²]_std_diff')
    plt.plot(stats_df['week_number'], stats_df['average_std_diff'], marker='d', label='Average Std Difference', linewidth=3, color='black')
    
    # Identify the best and worst weeks for std_diff
    best_week_std = sorted_weeks_std_diff.iloc[0]
    worst_week_std = sorted_weeks_std_diff.iloc[-1]
    
    # Highlight the best and worst weeks for std_diff
    plt.scatter(best_week_std['week_number'], best_week_std['average_std_diff'], color='green', s=100, label='Best Week (Std)')
    plt.scatter(worst_week_std['week_number'], worst_week_std['average_std_diff'], color='red', s=100, label='Worst Week (Std)')
    
    # Add horizontal line for the average std_diff of Direct Solar Radiation
    plt.axhline(y=direct_solar_avg_std_diff, color='purple', linestyle='--', linewidth=2, label='Avg Std Diff Direct Solar Radiation')
    
    # Add horizontal line for the median of average_std_diff
    plt.axhline(y=overall_median_std, color='orange', linestyle='-.', linewidth=2, label=f'Overall Median STD ({overall_median_std:.4f})')
    
    plt.xlabel('Week Number (Time Steps)', fontsize=15, fontweight='bold')
    plt.ylabel('Standard Deviation Difference', fontsize=15, fontweight='bold')
    plt.title('Standard Deviation Differences for Each Week (Normalized Data)', fontsize=18, fontweight='bold')
    plt.legend()
    plt.xticks(
        stats_df['week_number'], 
        [f"{int(w)} ({int(s)}-{int(e)})" for w, s, e in zip(stats_df['week_number'], stats_df['start'], stats_df['end'])], 
        rotation=90
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # -------------------- Second Figure: KL Divergence --------------------
    plt.figure(figsize=(20, 10))
    plt.plot(stats_df['week_number'], stats_df['Electricity_P_KL'], marker='o', label='Electricity Pricing [$]_KL_divergence')
    plt.plot(stats_df['week_number'], stats_df['CO2_KL'], marker='s', label='kg_CO2/kWh_KL_divergence')
    plt.plot(stats_df['week_number'], stats_df['Direct_solar_KL'], marker='^', label='Direct Solar Radiation [W/m²]_KL_divergence')
    plt.plot(stats_df['week_number'], stats_df['average_KL_divergence'], marker='d', label='Average KL Divergence', linewidth=3, color='black')
    
    # Identify the best and worst weeks for KL_divergence
    best_week_KL = sorted_weeks_KL.iloc[0]
    worst_week_KL = sorted_weeks_KL.iloc[-1]
    
    # Highlight the best and worst weeks for KL_divergence
    plt.scatter(best_week_KL['week_number'], best_week_KL['average_KL_divergence'], color='green', s=100, label='Best Week (KL)')
    plt.scatter(worst_week_KL['week_number'], worst_week_KL['average_KL_divergence'], color='red', s=100, label='Worst Week (KL)')
    
    plt.xlabel('Week Number (Time Steps)', fontsize=15, fontweight='bold')
    plt.ylabel('KL Divergence', fontsize=15, fontweight='bold')
    plt.title('KL Divergence for Each Week', fontsize=18, fontweight='bold')
    plt.legend(fontsize=10)
    plt.xticks(
        stats_df['week_number'], 
        [f"{int(w)} ({int(s)}-{int(e)})" for w, s, e in zip(stats_df['week_number'], stats_df['start'], stats_df['end'])], 
        rotation=90
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # -------------------- Third Figure: Std Differences and KL Divergence (Subplots) --------------------
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    
    # First Subplot: Standard Deviation Differences
    ax[0].plot(stats_df['week_number'], stats_df['Electricity_P_std_diff'], marker='o', label='Electricity Pricing [$]_std_diff')
    ax[0].plot(stats_df['week_number'], stats_df['CO2_std_diff'], marker='s', label='kg_CO2/kWh_std_diff')
    ax[0].plot(stats_df['week_number'], stats_df['Direct_solar_std_diff'], marker='^', label='Direct Solar Radiation [W/m²]_std_diff')
    ax[0].plot(stats_df['week_number'], stats_df['average_std_diff'], marker='d', label='Average Std Difference', linewidth=3, color='black')
    
    # Highlight best and worst weeks for std_diff
    ax[0].scatter(best_week_std['week_number'], best_week_std['average_std_diff'], color='green', s=100, label='Best Week (Std)')
    ax[0].scatter(worst_week_std['week_number'], worst_week_std['average_std_diff'], color='red', s=100, label='Worst Week (Std)')
    
    # Add horizontal lines
    ax[0].axhline(y=overall_avg_std_diff, color='blue', linestyle='--', linewidth=1, alpha=0.7, label=f'Overall Average STD ({overall_avg_std_diff:.4f})')
    ax[0].axhline(y=direct_solar_avg_std_diff, color='purple', linestyle='--', linewidth=2, label='Avg Std Diff Direct Solar Radiation')
    ax[0].axhline(y=overall_median_std, color='orange', linestyle='-.', linewidth=2, label=f'Overall Median STD ({overall_median_std:.4f})')
    
    ax[0].set_ylabel('Standard Deviation Difference', fontsize=15, fontweight='bold')
    ax[0].legend(fontsize=12, loc='upper left')
    ax[0].grid(True, linestyle='--', alpha=0.5)
    ax[0].set_xticks([])
    ax[0].set_xlabel('')
    
    # Second Subplot: KL Divergence
    ax[1].plot(stats_df['week_number'], stats_df['Electricity_P_KL'], marker='o', label='Electricity Pricing [$]_KL_divergence')
    ax[1].plot(stats_df['week_number'], stats_df['CO2_KL'], marker='s', label='kg_CO2/kWh_KL_divergence')
    ax[1].plot(stats_df['week_number'], stats_df['Direct_solar_KL'], marker='^', label='Direct Solar Radiation [W/m²]_KL_divergence')
    ax[1].plot(stats_df['week_number'], stats_df['average_KL_divergence'], marker='d', label='Average KL Divergence', linewidth=3, color='black')
    
    # Highlight best and worst weeks for KL Divergence
    ax[1].scatter(best_week_KL['week_number'], best_week_KL['average_KL_divergence'], color='green', s=100, label='Best Week (KL)')
    ax[1].scatter(worst_week_KL['week_number'], worst_week_KL['average_KL_divergence'], color='red', s=100, label='Worst Week (KL)')
    
    # Add horizontal line for overall average KL divergence
    ax[1].axhline(y=overall_avg_KL_divergence, color='blue', linestyle='--', linewidth=1, alpha=0.7, label=f'Overall Average KL ({overall_avg_KL_divergence:.4f})')
    
    ax[1].set_xlabel('Week Number (Time Steps)', fontsize=15, fontweight='bold')
    ax[1].set_ylabel('KL Divergence', fontsize=15, fontweight='bold')
    ax[1].legend(fontsize=10)
    ax[1].grid(True, linestyle='--', alpha=0.5)
    
    # Set xticks to display all week numbers on the bottom plot
    ax[1].set_xticks(stats_df['week_number'])
    ax[1].set_xticklabels([f"{int(w)}" for w in stats_df['week_number']], rotation=90)
    
    plt.tight_layout()
    plt.show()
    
    # -------------------- Fourth Figure: Average KL and Average STD Diff --------------------
    plt.figure(figsize=(20, 10))
    plt.plot(stats_df['week_number'], stats_df['average_std_diff'], marker='o', label='Average STD Difference', linewidth=2)
    plt.plot(stats_df['week_number'], stats_df['average_KL_divergence'], marker='s', label='Average KL Divergence', linewidth=2)
    
    # Add horizontal lines for overall averages
    plt.axhline(y=overall_avg_std_diff, color='blue', linestyle='--', linewidth=2, label=f'Overall Average STD ({overall_avg_std_diff:.4f})')
    plt.axhline(y=overall_avg_KL_divergence, color='red', linestyle='--', linewidth=2, label=f'Overall Average KL ({overall_avg_KL_divergence:.4f})')
    
    plt.xlabel('Week Number (Time Steps)', fontsize=15, fontweight='bold')
    plt.ylabel('Value', fontsize=15, fontweight='bold')
    plt.title('Average KL Divergence and Average STD Difference per Week', fontsize=18, fontweight='bold')
    plt.legend(fontsize=12)
    plt.xticks(
        stats_df['week_number'], 
        [f"{int(w)} ({int(s)}-{int(e)})" for w, s, e in zip(stats_df['week_number'], stats_df['start'], stats_df['end'])], 
        rotation=90
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    # -------------------- Fifth Figure: Additional Plot with Four Subplots --------------------
    # Create a figure with four subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    
    # Plot Average STD per Week
    axs[0].plot(stats_df['week_number'], stats_df['average_std'], marker='v', label='Average STD', linewidth=2, color='purple')
    axs[0].axhline(y=overall_avg_std, color='purple', linestyle='--', linewidth=2, label=f'Overall Average STD ({overall_avg_std:.4f})')
    axs[0].set_ylabel('Average STD', fontsize=12)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.5)
    
    # Plot Average KL Divergence
    axs[1].plot(stats_df['week_number'], stats_df['average_KL_divergence'], marker='s', label='Average KL Divergence', linewidth=2, color='red')
    axs[1].axhline(y=overall_avg_KL_divergence, color='red', linestyle='--', linewidth=2, label=f'Overall Average KL Div ({overall_avg_KL_divergence:.4f})')
    axs[1].set_ylabel('Average KL Divergence', fontsize=12)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    
    # Plot Average STD Difference
    axs[2].plot(stats_df['week_number'], stats_df['average_std_diff'], marker='o', label='Average STD Difference', linewidth=2, color='blue')
    axs[2].axhline(y=overall_avg_std_diff, color='blue', linestyle='--', linewidth=2, label=f'Overall Average STD Diff ({overall_avg_std_diff:.4f})')
    axs[2].set_ylabel('Average STD Difference', fontsize=12)
    axs[2].legend(fontsize=10)
    axs[2].grid(True, linestyle='--', alpha=0.5)
    
    # Plot Average Mean per Week
    axs[3].plot(stats_df['week_number'], stats_df['average_mean'], marker='^', label='Average Mean', linewidth=2, color='green')
    axs[3].axhline(y=overall_avg_mean, color='green', linestyle='--', linewidth=2, label=f'Overall Average Mean ({overall_avg_mean:.4f})')
    axs[3].set_ylabel('Average Mean', fontsize=12)
    axs[3].legend(fontsize=10)
    axs[3].grid(True, linestyle='--', alpha=0.5)
    axs[3].set_xlabel('Week Number (Time Steps)', fontsize=12)
    
    # Set x-ticks for all subplots
    axs[3].set_xticks(stats_df['week_number'])
    axs[3].set_xticklabels(
        [f"{int(w)} ({int(s)}-{int(e)})" for w, s, e in zip(
            stats_df['week_number'], stats_df['start'], stats_df['end'])], rotation=90)
    
    plt.tight_layout()
    plt.show()
