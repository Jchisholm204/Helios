import pandas as pd
import matplotlib.pyplot as plt

# === Configuration ===
csv_file = "rrt_w2c.csv"  # <-- Replace with your actual path

# === Read CSV (ignore commented lines) ===
df = pd.read_csv(csv_file, comment='#')

# Rename columns for easier access
df.columns = ['run', 'iterations', 'vertices', 'solution', 'goal']

# === Compute statistics ===
cols = ['iterations', 'vertices', 'solution', 'goal']
print("=== Summary Statistics ===")
for col in cols:
    mean_val = df[col].mean()
    median_val = df[col].median()
    print(f"{col.capitalize():<10} | Mean: {mean_val:.2f} | Median: {median_val:.2f}")

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
        ax.hist(df[col], bins=15, edgecolor='black')
    ax.set_title(titles[col])
    ax.set_xlabel(titles[col])
    ax.set_ylabel("Frequency")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
