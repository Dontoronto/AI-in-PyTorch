import math

from PIL import Image, ImageDraw, ImageFont

def get_matching_patterns(pattern_set, percentage):
    matched_patterns = []

    for i in range(len(percentage)):
        for pattern in pattern_set:
            if set(pattern) == set(percentage[i].keys()):
                matched_patterns.append(pattern)
                break

    return matched_patterns

def create_image_with_probabilities(pattern, probabilities, matrix_size=(3, 3), cell_size=100):
    """
    Create an image with positions labeled by probabilities and color intensity based on the probabilities.

    Parameters:
    - pattern: The pattern to highlight.
    - probabilities: The dictionary of probabilities for each position in the pattern.
    - matrix_size: The size of the matrix (default is 3x3).
    - cell_size: The size of each cell in the matrix (default is 50).
    """
    # Calculate the size of the image
    img_size = (matrix_size[1] * cell_size, matrix_size[0] * cell_size)

    img = Image.new('RGB', img_size, color='white') #f6ffb7
    #img = Image.new('RGBA', img_size, color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Font for text (you might need to adjust the path to the font file)
    font_path = "fonts/PalatinoMediumItalic.otf"
    # Load the Palatino Bold font with size 14
    try:
        font = ImageFont.truetype(font_path, 36)
    except IOError:
        print("Font file not found. Please check the path and try again.")
        font = ImageFont.load_default(36)

    for pos in pattern:
        row, col = divmod(pos, matrix_size[1])
        top_left = (col * cell_size, row * cell_size)
        bottom_right = ((col + 1) * cell_size, (row + 1) * cell_size)

        # Get the probability value for the position
        prob = probabilities.get(pos, 0)

        # # Determine the color intensity based on the probability
        # intensity = int(prob * 255)
        # color = (intensity, 0, 0)


        # Determine the color intensity based on the probability
        if prob > 0.5:
            intensity = int((prob - 0.5) * 3.5 * 255)
            color = (255, 255 - intensity, 255 - intensity)  # Bright red
        else:
            intensity = int((0.5 - prob) * 3.5 * 255)
            color = (255 - intensity, 255, 255 - intensity)  # Bright green

        # Draw the rectangle with the color
        draw.rectangle([top_left, bottom_right], fill=color)

        # Draw the probability text in the cell
        text = f"{int((50-prob*100)*2)}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_position = (top_left[0] + (cell_size - text_size[0]) / 2, top_left[1] + (cell_size - text_size[1]) / 2)
        draw.text(text_position, text, fill="black", font=font)

    # Draw the grid
    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            top_left = (j * cell_size, i * cell_size)
            bottom_right = ((j + 1) * cell_size, (i + 1) * cell_size)
            draw.rectangle([top_left, bottom_right], outline='#dcdcdc', width=3)

    return img


def combine_images_with_positions(patterns, percentage_dict, max_images_per_row=4, spacing=5):
    # Create images from each tuple
    # images = [create_image_with_positions(tpl) for tpl in tuples_set]
    patterns_tuple = get_matching_patterns(patterns, percentage_dict)
    images = []
    # Create images for each pattern in the pattern_set
    for i in range(len(patterns_tuple)):
        img = create_image_with_probabilities(patterns_tuple[i], percentage_dict[i])
        #img.save(f'pattern_{i}.png')
        #img.show()
        images.append(img)

    # Determine the size for the combined image
    width, height = images[0].size
    rows = (len(images) + max_images_per_row - 1) // max_images_per_row
    combined_width = width * max_images_per_row + spacing * (max_images_per_row - 1)
    combined_height = height * rows + spacing * (rows - 1)

    # Create a new blank image with the appropriate size
    combined_image = Image.new('RGB', (combined_width, combined_height), color='white')

    # Paste each image into the combined image with spacing
    for i, img in enumerate(images):
        row = i // max_images_per_row
        col = i % max_images_per_row
        if row == rows - 1:
            remaining_images = len(images) % max_images_per_row
            if remaining_images == 0:
                remaining_images = max_images_per_row
            total_row_width = remaining_images * width + (remaining_images - 1) * spacing
            start_x = (combined_width - total_row_width) // 2
            x = start_x + col * (width + spacing)
        else:
            x = col * (width + spacing)
        y = row * (height + spacing)
        combined_image.paste(img, (x, y))

    return combined_image



#%%
