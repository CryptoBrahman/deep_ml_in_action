import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
tips = sns.load_dataset("tips")
fmri = sns.load_dataset("fmri")
titanic = sns.load_dataset("titanic")
flights = sns.load_dataset("flights")

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Scatter plot
sns.scatterplot(x="total_bill", y="tip", data=tips, ax=axes[0, 0])
axes[0, 0].set_title('Scatter Plot')

# Line plot
sns.lineplot(x="timepoint", y="signal", data=fmri, ax=axes[0, 1])
axes[0, 1].set_title('Line Plot')

# Bar plot
sns.barplot(x="class", y="survived", data=titanic, ax=axes[1, 0])
axes[1, 0].set_title('Bar Plot')

# Heatmap
flights_pivot = flights.pivot_table(index="month", columns="year", values="passengers")
sns.heatmap(flights_pivot, cmap="Blues", annot=True, fmt="d", ax=axes[1, 1])
axes[1, 1].set_title('Heatmap')

# Adjust layout
plt.tight_layout()

# Save the file
plt.savefig('seaborn_plots.png')
plt.show()
