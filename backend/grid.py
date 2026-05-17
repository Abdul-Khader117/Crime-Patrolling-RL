import numpy as np

class CrimeGrid:
    def __init__(self, df, grid_size=20):
        self.df = df
        self.grid_size = grid_size

        # Boundaries
        self.min_lat = df['latitude'].min()
        self.max_lat = df['latitude'].max()
        self.min_lon = df['longitude'].min()
        self.max_lon = df['longitude'].max()

        # CREATE BINS
        self.lat_bins = np.linspace(self.min_lat, self.max_lat, grid_size + 1)
        self.lon_bins = np.linspace(self.min_lon, self.max_lon, grid_size + 1)

        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size))

    def _map_to_grid(self, lat, lon):
        """Convert lat/lon to grid indices"""
        lat_idx = np.digitize(lat, self.lat_bins) - 1
        lon_idx = np.digitize(lon, self.lon_bins) - 1

        # Clamp values to avoid index overflow
        lat_idx = min(max(lat_idx, 0), self.grid_size - 1)
        lon_idx = min(max(lon_idx, 0), self.grid_size - 1)

        return lat_idx, lon_idx

    def populate_grid(self):
        """Fill grid with crime counts"""
        for _, row in self.df.iterrows():
            lat, lon = row['latitude'], row['longitude']
            i, j = self._map_to_grid(lat, lon)
            self.grid[i][j] += 1
        return self.grid

    def normalize_grid(self):
        """Normalize values between 0 and 1"""
        max_val = np.max(self.grid)
        if max_val > 0:
            self.grid = self.grid / max_val
        return self.grid

    def get_grid(self):
        return self.grid

    def get_crime_levels(self):
        """Categorize grid into 0 (Low), 1 (Medium), 2 (High), -1 (Water/Invalid)"""
        flat = self.grid.flatten()
        non_zero = flat[flat > 0]

        if len(non_zero) < 3:
            # Not enough data for robust percentiles, use simple thresholds
            level_grid = np.zeros_like(self.grid)
            invalid_mask = np.zeros_like(self.grid, dtype=bool)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    val = self.grid[i][j]
                    # Simple heuristic: Eastern side of Chicago is water if zero crime
                    if val == 0 and j > self.grid_size * 0.7:
                        invalid_mask[i][j] = True
                        level_grid[i][j] = -1
                    elif val > 0:
                        level_grid[i][j] = 2
            return level_grid, invalid_mask

        # Use log scale for categorized thresholds to handle high variance in crime counts
        log_non_zero = np.log1p(non_zero)
        low_th = np.percentile(log_non_zero, 33)
        high_th = np.percentile(log_non_zero, 66)

        level_grid = np.zeros_like(self.grid)
        invalid_mask = np.zeros_like(self.grid, dtype=bool)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val_raw = self.grid[i][j]
                val_log = np.log1p(val_raw)

                # HEURISTIC: Water masking for Chicago (Lake Michigan is to the East)
                # We also check if the cell is completely empty.
                is_water = False
                if val_raw == 0:
                    # The eastern boundary of Chicago is not a straight line.
                    # It curves. j > threshold is a decent approximation.
                    # We can make it slightly dynamic based on latitude (i).
                    # Higher i (North) -> Lake is further East.
                    # Lower i (South) -> Lake is slightly closer West.
                    # Normalized latitude (0 to 1)
                    norm_i = i / self.grid_size
                    # Threshold varies slightly with latitude
                    threshold = 0.72 + (norm_i * 0.05) 
                    if j > self.grid_size * threshold:
                        is_water = True

                if is_water:
                    invalid_mask[i][j] = True
                    level_grid[i][j] = -1
                elif val_raw == 0:
                    level_grid[i][j] = 0
                elif val_log < low_th:
                    level_grid[i][j] = 0
                elif val_log < high_th:
                    level_grid[i][j] = 1
                else:
                    level_grid[i][j] = 2

        return level_grid, invalid_mask