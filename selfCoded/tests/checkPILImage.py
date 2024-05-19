import PIL
from pathlib import Path
from PIL import UnidentifiedImageError

# path = Path("/Volumes/Extreme SSD/datasets/imagenet/imagenet1k/train/n01440764/").rglob("*.JPEG")
path = Path("/Users/dominik/Downloads/n01440764/").rglob("*.JPEG")
length = 0
error = 0
for img_p in path:
    length += 1
    try:
        img = PIL.Image.open(img_p)
    except PIL.UnidentifiedImageError:
        error += 1
        print(img_p)

print(f"Error Ratio= {error/length}")
#%%
