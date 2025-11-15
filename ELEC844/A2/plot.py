import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
csv_file = "./rrtstar_w1a_500.csv"

# === Read CSV (ignore commented lines) ===
df = pd.read_csv(csv_file, comment='#')

# Rename columns for easier access
df.columns = ['run', 'iterations', 'vertices', 'solution', 'goal']

# Define a large number to represent infinite path length.
# This number MUST be larger than any valid path length found.
inf_val = 1e9

# Replace the sentinel value (-1) in the 'goal' column with the 'infinite' value
df.loc[df['goal'] == -1, 'goal'] = inf_val
print(
    f"Goal lengths of -1 have been replaced with {inf_val:,.0f} for statistical analysis.")

# === Compute statistics ===
cols = ['iterations', 'vertices', 'solution', 'goal']
print("=== Summary Statistics ===")
for col in cols:
    # Calculate quartiles and mean
    q1 = df[col].quantile(0.25)
    median_val = df[col].quantile(0.50)  # Same as .median()
    q3 = df[col].quantile(0.75)
    mean_val = df[col].mean()

    # Print results
    print(f"--- {col.capitalize()} ---")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median (Q2): {median_val:.2f}")
    print(f"1st Quartile (Q1): {q1:.2f}")
    print(f"3rd Quartile (Q3): {q3:.2f}")

# === Plot histograms ===
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False, sharey=True)
fig.suptitle("Histogram of RRT Performance Metrics")

# Custom titles for each plot
titles = {
    "iterations": "Number of Iterations",
    "vertices": "Number of Vertices",
    "solution": "Vertices in Solution",
    "goal": "Path Length to Goal"
}

# Determine common x-axis range for iterations and vertices
x_min = min(df["iterations"].min(), df["vertices"].min())
x_max = max(df["iterations"].max(), df["vertices"].max())

for ax, col in zip(axes.flat, cols):
    if col in ["iterations", "vertices"]:
        ax.hist(df[col], bins=15, edgecolor='black', range=(x_min, x_max))
    else:
        # Note: You may need to adjust this range (0, 175) based on your RRT* path lengths
        ax.hist(df[col], bins=15, edgecolor='black', range=(0, 175))
    ax.set_title(titles[col])
    ax.set_xlabel(titles[col])
    ax.set_ylabel("Frequency")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
