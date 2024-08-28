import os
from PIL import Image

def get_jpg_dimensions(directory):
    jpg_files = [file for file in os.listdir(directory) if file.endswith('.jpg')]
    dimensions = []

    for file in jpg_files:
        file_path = os.path.join(directory, file)
        with Image.open(file_path) as img:
            width, height = img.size
            dimensions.append((file, width, height))

    return dimensions

# Example usage
directory = 'data/train/train'  # Replace with the actual directory path
jpg_dimensions = get_jpg_dimensions(directory)
import matplotlib.pyplot as plt

# Extract width and height from jpg_dimensions
widths = [dim[1] for dim in jpg_dimensions]
heights = [dim[2] for dim in jpg_dimensions]

# Plot scatter plot
plt.scatter(widths, heights)
plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Scatter Plot of Image Dimensions')
plt.show()