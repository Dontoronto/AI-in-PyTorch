from PIL import Image, ImageDraw

def create_image_with_positions(positions, matrix_size=(3, 3), cell_size=100):
    # Calculate the size of the image
    img_size = (matrix_size[0] * cell_size, matrix_size[1] * cell_size)

    img = Image.new('RGB', img_size, color='#f6ffb7')
    draw = ImageDraw.Draw(img)

    # Highlight the positions
    for pos in positions:
        row, col = divmod(pos, matrix_size[0])
        top_left = (col * cell_size, row * cell_size)
        bottom_right = ((col + 1) * cell_size, (row + 1) * cell_size)
        draw.rectangle([top_left, bottom_right], fill='#890205')

    for i in range(matrix_size[0]):
        for j in range(matrix_size[1]):
            top_left = (i * cell_size, j * cell_size)
            bottom_right = ((i + 1) * cell_size, (j + 1) * cell_size)
            draw.rectangle([top_left, bottom_right], outline='#dcdcdc', width=3)

    return img

def combine_images_with_positions(tuples_set, max_images_per_row=4, spacing=5):
    # Create images from each tuple
    images = [create_image_with_positions(tpl) for tpl in tuples_set]

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


