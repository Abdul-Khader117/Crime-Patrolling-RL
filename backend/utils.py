import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_crime_map(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['longitude'], df['latitude'], s=1, alpha=0.3)
    plt.title("Crime Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def plot_grid(grid):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Crime Density")
    plt.title("Crime Heatmap Grid")
    plt.show()


def plot_crime_levels(level_grid):
    cmap = mcolors.ListedColormap(['green', 'yellow', 'red'])

    plt.figure(figsize=(6, 6))
    plt.imshow(level_grid, cmap=cmap)
    plt.colorbar(ticks=[0, 1, 2], label="Crime Level")
    plt.title("Crime Level Map (Low/Medium/High)")
    plt.show()