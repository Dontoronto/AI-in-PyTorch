from PIL import Image

# Load the image
image_path = "image_2.png"
# image_path = "ILSVRC2012_val_00000236.JPEG"
image = Image.open(image_path)

# Check the mode of the image
original_mode = image.mode

# Save the image back
output_path = "saved_image.JPEG"
image.save(output_path)

# Verify saved image
saved_image = Image.open(output_path)
saved_mode = saved_image.mode

# Display the original and saved image modes
import pandas as pd

data = pd.DataFrame({
    'Original Mode': [original_mode],
    'Saved Mode': [saved_mode]
})

print(data)


#%%
