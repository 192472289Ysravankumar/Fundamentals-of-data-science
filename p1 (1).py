import pandas as pd
import numpy as np

# Load CSV file
df = pd.read_csv(r"C:\Users\tthul\Downloads\house_data(1).csv")

# Display the data
print("Dataset:")
print(df)

# Convert to NumPy array
house_data = df.to_numpy()

# Select houses with more than 4 bedrooms
filtered_houses = house_data[house_data[:, 1] > 4] 

# Calculate average sale price
average_price = np.mean(filtered_houses[:, 3])       
print("\nAverage Sale Price of Houses with more than 4 Bedrooms: ₹", average_price)