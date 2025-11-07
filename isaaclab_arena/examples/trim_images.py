# %%

import pathlib

from PIL import Image

embodiments = ["franka", "gr1"]
objects = ["banana", "apple", "broccoli", "carrot"]

images_root = pathlib.Path("/datasets/2025_11_07_arena_front_page")
input_root = images_root / "raw"
output_root = images_root / "output"

# Crop parameters
X = 250  # Starting X position (left edge of crop)
size = 720  # Width of the cropped region

# Create output directory if it doesn't exist
output_root.mkdir(parents=True, exist_ok=True)

for embodiment in embodiments:
    for object_name in objects:
        image_path = input_root / f"{embodiment}_{object_name}.png"
        output_path = output_root / f"{embodiment}_{object_name}.png"

        if image_path.exists():
            image = Image.open(image_path)
            width, height = image.size

            # Crop: (left, top, right, bottom)
            # Start at X position and take 'size' pixels width
            cropped_image = image.crop((X, 0, X + size, height))

            cropped_image.save(output_path)
            print(f"Processed: {image_path.name} -> {output_path.name}")
        else:
            print(f"Not found: {image_path.name}")


# %%

# Combine all images into a single image
spacing = 40  # Pixels of space between images

combined_images = []

for embodiment in embodiments:
    row_images = []
    for object_name in objects:
        output_path = output_root / f"{embodiment}_{object_name}.png"
        if output_path.exists():
            img = Image.open(output_path)
            row_images.append(img)
    combined_images.append(row_images)

# Get dimensions from first image
img_width = combined_images[0][0].width
img_height = combined_images[0][0].height

# Create combined image (2 rows x 4 columns)
num_rows = len(embodiments)
num_cols = len(objects)
# Add spacing between images
combined_width = img_width * num_cols + spacing * (num_cols - 1)
combined_height = img_height * num_rows + spacing * (num_rows - 1)

combined_image = Image.new("RGB", (combined_width, combined_height), color="white")

# Paste images into combined image
for row_idx, row_images in enumerate(combined_images):
    for col_idx, img in enumerate(row_images):
        x_pos = col_idx * (img_width + spacing)
        y_pos = row_idx * (img_height + spacing)
        combined_image.paste(img, (x_pos, y_pos))

# Save combined image
combined_output_path = images_root / "combined.png"
combined_image.save(combined_output_path)
print(f"Combined image saved to: {combined_output_path}")

# Display the combined image (optional, useful in Jupyter notebooks)
combined_image.show()

# %%


# %%
