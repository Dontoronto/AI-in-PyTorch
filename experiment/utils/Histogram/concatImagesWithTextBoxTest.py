from PIL import Image, ImageDraw, ImageFont

# Paths to the input images
base_path = "/Volumes/Extreme SSD/etc/Gewichtsverteilung"
image_paths = [
    f"{base_path}/Base Model_density_histogram.png",
    f"{base_path}/Unstruct Adv_density_histogram.png",
    f"{base_path}/Trivial Adv_density_histogram.png",
    f"{base_path}/SCP Adv_density_histogram.png"
]

# Open the images
images = [Image.open(image_path) for image_path in image_paths]

# Get the maximum width and height of the images
width, height = 0, 0
for img in images:
    temp_width, temp_height = img.size
    if temp_width > width:
        width = temp_width
    if temp_height > height:
        height = temp_height

# Create a new image with the appropriate size
combined_image = Image.new("RGBA", (2 * width, 2 * height), color='white')

# Calculate the positions to center each image
positions = [
    (int((width - images[0].width) / 2), int((height - images[0].height) / 2)),
    (width + int((width - images[1].width) / 2), int((height - images[1].height) / 2)),
    (int((width - images[2].width) / 2), height + int((height - images[2].height) / 2)),
    (width + int((width - images[3].width) / 2), height + int((height - images[3].height) / 2))
]

# Paste the images into the combined image at the calculated positions
for i, img in enumerate(images):
    combined_image.paste(img, positions[i])

# Save the combined image temporarily
combined_image_path = "combined_image.png"
combined_image.save(combined_image_path)

# Open the combined image to draw the boxes and text
combined_image = Image.open(combined_image_path)

# Define custom texts for each plot
custom_texts = ["Base Model", "Unstruct Adv", "Trivial Adv", "SCP Adv"]

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define a list of colors
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
# colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
# colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462']
# colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']
# colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
# colors = ['#1f77b4', '#d95f02', '#003f5c', '#e7298a', '#66a61e', '#e6ab02']

# Define positions for the custom texts manually
text_positions = [
    (520-30, 520),   # Position for the first box
    (1920*1.6-30, 520),   # Position for the second box
    (520-30, 1620*1.3),  # Position for the third box
    (1920*1.6-30, 1620*1.3)   # Position for the fourth box
]

# Box size and color
box_width, box_height = 620, 200
box_color = "#fb8072"
text_color = "black"
outline_color = "black"

# Choose a font and size
font_path = "fonts/PalatinoBold.ttf"  # Adjust the path to the font file as needed
font_size = 100
font = ImageFont.truetype(font_path, font_size)

# Open a drawing context
draw = ImageDraw.Draw(combined_image)

# Draw boxes with rounded edges and custom texts
for position, text in zip(text_positions, custom_texts):
    x, y = position
    # Draw the rounded rectangle
    draw.rounded_rectangle(
        [(x, y), (x + box_width, y + box_height)],
        radius=10,
        fill=box_color,
        outline=outline_color,
        width=8
    )
    # Calculate the text position using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = x + (box_width - text_width) // 2
    text_y = y + (box_height - text_height) // 2
    # Draw the text
    draw.text((text_x, text_y), text, fill=text_color, font=font)


# Save the combined image
output_image_path = "combined_image_with_boxes.png"
combined_image.save(output_image_path)

print(f"Combined image saved as {output_image_path}")
