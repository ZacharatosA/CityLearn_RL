import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import rel_entr

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

# Initialize parameters
train_period = 168 
#train_period = 729 
End = len(data)
kl_divergences = []

# Function to calculate KL divergence with combined binning
def calculate_kl_divergence(train, rest, bins=50):
    combined = np.concatenate([train, rest])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)
    
    # Υπολογισμός των histogram
    train_counts, _ = np.histogram(train, bins=bin_edges, density=True)
    rest_counts, _ = np.histogram(rest, bins=bin_edges, density=True)
    
    epsilon = 1e-10
    train_counts += epsilon
    rest_counts += epsilon
    
    train_dist = train_counts / train_counts.sum()
    rest_dist = rest_counts / rest_counts.sum()
    
    kl_div = np.sum(rel_entr(train_dist, rest_dist))
    return kl_div

# Iterate over each week in the dataset starting from timestep 1
for start in range(1, End - train_period + 1, train_period):
    end = start + train_period - 1
    week_data = data.iloc[start:end+1]
    rest_data = pd.concat([data.iloc[:start], data.iloc[end+1:]])
    
    kl_divs = []
    for column in data.columns:
        kl_div = calculate_kl_divergence(week_data[column], rest_data[column], bins=50)
        kl_divs.append(kl_div)
    
    avg_kl_div = np.mean(kl_divs)
    kl_divergences.append((start, avg_kl_div))

# Convert to numpy array for easier processing
kl_divergences = np.array(kl_divergences)

# Find the week with the minimum, maximum, and average KL Divergence
min_kl_divergence = np.min(kl_divergences[:, 1])
max_kl_divergence = np.max(kl_divergences[:, 1])
average_kl_divergence = np.mean(kl_divergences[:, 1])

best_week_idx = np.argmin(kl_divergences[:, 1])
worst_week_idx = np.argmax(kl_divergences[:, 1])
average_week_idx = np.argmin(np.abs(kl_divergences[:, 1] - average_kl_divergence))

best_week_start = kl_divergences[best_week_idx, 0]
worst_week_start = kl_divergences[worst_week_idx, 0]
average_week_start = kl_divergences[average_week_idx, 0]

best_week_end = best_week_start + train_period - 1
worst_week_end = worst_week_start + train_period - 1
average_week_end = average_week_start + train_period - 1

print(f"The best training week is the one starting at hour {int(best_week_start)} and ends at hour {int(best_week_end)} with KL Divergence: {kl_divergences[best_week_idx, 1]:.4f}")
print(f"The average training week is the one starting at hour {int(average_week_start)} and ends at hour {int(average_week_end)} with KL Divergence: {kl_divergences[average_week_idx, 1]:.4f}")
print(f"The worst training week is the one starting at hour {int(worst_week_start)} and ends at hour {int(worst_week_end)} with KL Divergence: {kl_divergences[worst_week_idx, 1]:.4f}")

# Plot the KL divergences
plt.figure(figsize=(12, 8))

# Use a colormap to color the weeks based on KL divergence
norm = plt.Normalize(min_kl_divergence, max_kl_divergence)
cmap = plt.get_cmap("RdYlGn_r")
colors = [cmap(norm(kl)) for kl in kl_divergences[:, 1]]

for i, (start, kl) in enumerate(kl_divergences):
    lw = 2
    if i == best_week_idx or i == worst_week_idx or i == average_week_idx:
        plt.step([start, start + train_period], [kl, kl], color=colors[i], lw=4, where='post')
        plt.vlines(start, 0, kl, colors=colors[i], linestyles='dotted', lw=2)
    else:
        plt.step([start, start + train_period], [kl, kl], color=colors[i], lw=lw, where='post')
        plt.vlines(start, 0, kl, colors=colors[i], linestyles='dotted')

plt.axhline(y=min_kl_divergence, color='g', linestyle='--', label=f'Min KL Divergence: {min_kl_divergence:.4f}')
plt.axhline(y=max_kl_divergence, color='r', linestyle='--', label=f'Max KL Divergence: {max_kl_divergence:.4f}')
plt.axhline(y=average_kl_divergence, color='b', linestyle='--', label=f'Average KL Divergence: {average_kl_divergence:.4f}')

plt.xlabel('Week')
plt.ylabel('KL Divergence')
plt.legend()
plt.title('KL Divergence for each Week')

# Adjust the x-axis labels to show week intervals with week numbers
week_labels = [f"{i+1}. {int(start)}-{int(start + train_period - 1)}" for i, (start, _) in enumerate(kl_divergences)]
plt.xticks(ticks=kl_divergences[:, 0], labels=week_labels, rotation=90)

# Highlight the x-ticks for the best, worst, and average weeks
ax = plt.gca()
ax.get_xticklabels()[best_week_idx].set_fontweight("bold")
ax.get_xticklabels()[worst_week_idx].set_fontweight("bold")
ax.get_xticklabels()[average_week_idx].set_fontweight("bold")

# Adjust layout to give more space for x-axis labels
plt.tight_layout(rect=[0, 0.3, 1, 1])

plt.show()

# Print the weeks sorted by KL Divergence
sorted_weeks = sorted(enumerate(kl_divergences[:, 1]), key=lambda x: x[1])

print(f"{'Rank':<4} {'Week':<8} {'Time Steps':<20} {'KL Divergence':<10}")
print("-" * 50)

for rank, (week, kl) in enumerate(sorted_weeks, start=1):
    start_hour = int(kl_divergences[week, 0])
    end_hour = int(kl_divergences[week, 0] + train_period - 1)
    week_label = f"Week {week + 1}"  # Adjusting the label to show "Week1", "Week2", etc.
    time_steps = f"{start_hour}-{end_hour}"
    print(f"{rank:<4} {week_label:<8} {time_steps:<20} {kl:.4f}")

