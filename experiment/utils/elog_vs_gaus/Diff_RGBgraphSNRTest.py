import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.ndimage import convolve
from PIL import Image
import requests
from io import BytesIO


def generate_array(mode, shape):
    """
    Generates a 2D numpy array of zeros with specified mode.

    Parameters:
    mode (str): Mode of filling the array, "column" or "row".
    shape (tuple): Shape of the array (rows, columns).

    Returns:
    np.ndarray: Generated 2D array.
    """
    if mode not in ["column", "row"]:
        raise ValueError("Mode must be 'column' or 'row'")

    rows, columns = shape
    array = np.zeros(shape, dtype=np.uint8)

    if mode == "column":
        for col in range(columns):
            array[:, col] = col
    elif mode == "row":
        for row in range(rows):
            array[row, :] = row

    return array
def calculate_snr(image, noisy_image):
    signal_power = np.mean(image ** 2)
    noise_power = np.mean((image - noisy_image) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    #snr = signal_power / noise_power
    return snr

# Schärfemessung mit Laplacian-Varianz
def measure_sharpness(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # variance = laplacian.var()
    variance = 10 * np.log10(laplacian.var())
    return variance

# Function to measure sharpness using Laplacian variance
def measure_sharpness_gryy(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return round(variance, 1)

# Funktion zum Laden eines Bildes
def load_image(image_path, as_gray=False):
    img = Image.open(image_path)
    if as_gray:
        img = img.convert('L')  # In Graustufen umwandeln
    else:
        img = img.convert('RGB')
    return np.array(img)

scp1 = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 0]

]

scp2 = [
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 0]
]

scp3 = [
    [0, 0, 0],
    [1, 1, 1],
    [0, 1, 0]
]

scp4 = [
    [0, 1, 0],
    [0, 1, 1],
    [0, 1, 0]
]

scps = np.array((scp1, scp2, scp3, scp4)).astype(np.float64)

pos_neg = np.array((-1,1))

# Benutzerdefinierte 3x3-Filter
# filter_1 = np.array([[0, 1, 0],
#                      [1, -4, 1],
#                      [0, 1, 0]])
# filter_2 = filter_2/np.sum(np.abs(filter_2))





filter_1 = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])

filter_2 = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])

filter_3 = np.array([[0, 1, 0],
                     [1, -8, 1],
                     [0, 1, 0]])

filter_4 = np.array([
    [-1, 2, -1],
    [2, -4, 2],
    [-1, 2, -1]
])

filter_2 = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])

# filter_2 = scipy.signal.convolve2d(filter_2,filter_5,mode='same')
filter_3 = filter_3/np.sum(np.abs(filter_3))
filter_2 = filter_2/np.sum(np.abs(filter_2))

# Funktion zur Anwendung des Filters auf das Bild
def apply_filter(image, filter_kernel):
    if len(image.shape) == 2:  # Graustufenbild
        return convolve(image, filter_kernel)
    else:  # RGB-Bild
        channels = []
        for i in range(3):
            channels.append(convolve(image[:, :, i], filter_kernel))
        return np.stack(channels, axis=-1)

# Bild laden (Pfad zum Bild anpassen)
image_path = 'images/10er.jpg'
image_rgb = load_image(image_path)
image_gray = load_image('images/SW-Kontrast_02.jpg', as_gray=True)
# print(image_gray)
# print(image_gray.shape)
# print(type(image_gray))
# image_gray = generate_array(mode="column", shape=(200,200))

noise_raw = (np.random.random((200, 150)) - 0.5) * 2
noise = noise_raw*255*0.2
# noise = np.random.random((200,150))*255*0.1
noise = noise.round(0).astype(np.int8)
image_gray_noise = np.clip(image_gray + noise, 0, 255).astype(np.uint8)
# image_gray_noise = image_gray + noise

noise_raw = (np.random.random((200, 200, 1)) - 0.5) * 2
noise = noise_raw*255*0.2
# noise = np.random.random((200,200,1))*255*0.1
noise = noise.round(0).astype(np.int8)
print(noise)
image_rgb_noise = np.clip(image_rgb + noise, 0, 255).astype(np.uint8)
iterations = 20
name = 'elog'


#image_gray = image_gray.reshape(1,28,28)

# Filter auf die Bilder anwenden
# filtered_image_rgb_1 = apply_filter(image_rgb_noise, filter_2)
# filtered_image_rgb_1 = apply_filter(filtered_image_rgb_1, filter_2)
# filtered_image_rgb_2 = apply_filter(image_rgb, filter_2)
# filtered_image_rgb_3 = apply_filter(image_rgb, filter_2)
# filtered_image_rgb_4 = apply_filter(image_rgb, filter_2)
# filtered_image_rgb_5 = apply_filter(image_rgb, filter_2)
# filtered_image_gray_1 = apply_filter(image_gray_noise, filter_2)
# filtered_image_gray_1 = apply_filter(filtered_image_gray_1, filter_2)
# filtered_image_gray_2 = apply_filter(image_gray, filter_2)
# filtered_image_gray_3 = apply_filter(image_gray, filter_2)
# filtered_image_gray_4 = apply_filter(image_gray, filter_2)
# filtered_image_gray_5 = apply_filter(image_gray, filter_2)

filtered_image_rgb_1 = image_rgb_noise
filtered_image_rgb_2 = image_rgb_noise
filtered_image_gray_1 = image_gray_noise
filtered_image_gray_2 = image_gray_noise

def gaus_snr_sharp(iteration):
    x1 = []
    x2 = []

    filtered_image_rgb_1 = image_rgb_noise
    sharpness = measure_sharpness(filtered_image_rgb_1)
    snr = calculate_snr(filtered_image_rgb_1, image_rgb)
    x1.append(sharpness)
    x2.append(snr)
    for i in range(iteration-1):
        filtered_image_rgb_1 = apply_filter(filtered_image_rgb_1, filter_2)
        # filtered_image_gray_2 = apply_filter(image_gray_noise, filter_3)
        sharpness = measure_sharpness(filtered_image_rgb_1)
        snr = calculate_snr(filtered_image_rgb_1, image_rgb)
        x1.append(sharpness)
        x2.append(snr)
    return x1, x2, filtered_image_rgb_1

def gaus_snr_sharp_gray(iteration):
    x1 = []
    x2 = []

    filtered_image_gray_1 = image_gray_noise
    sharpness = measure_sharpness(filtered_image_gray_1)
    snr = calculate_snr(filtered_image_gray_1, image_gray)
    x1.append(sharpness)
    x2.append(snr)
    for i in range(iteration-1):
        filtered_image_gray_1 = apply_filter(filtered_image_gray_1, filter_2)
        # filtered_image_gray_2 = apply_filter(image_gray_noise, filter_3)
        sharpness = measure_sharpness(filtered_image_gray_1)
        snr = calculate_snr(filtered_image_gray_1, image_gray)
        x1.append(sharpness)
        x2.append(snr)
    return x1, x2, filtered_image_gray_1


def elog_snr_sharp(iteration):
    x1 = []
    x2 = []

    filtered_image_rgb_2 = image_rgb_noise
    sharpness = measure_sharpness(filtered_image_rgb_2)
    snr = calculate_snr(filtered_image_rgb_2, image_rgb)
    x1.append(sharpness)
    x2.append(snr)
    for i in range(iteration-1):
        filtered_image_rgb_2 = apply_filter(filtered_image_rgb_2, filter_3)
        # filtered_image_gray_2 = apply_filter(image_gray_noise, filter_3)
        sharpness = measure_sharpness(filtered_image_rgb_2)
        snr = calculate_snr(filtered_image_rgb_2, image_rgb)
        x1.append(sharpness)
        x2.append(snr)
    return x1, x2, filtered_image_rgb_2

def elog_snr_sharp_gray(iteration):
    x1 = []
    x2 = []

    filtered_image_gray_2 = image_gray_noise
    sharpness = measure_sharpness(filtered_image_gray_2)
    snr = calculate_snr(filtered_image_gray_2, image_gray)
    x1.append(sharpness)
    x2.append(snr)
    for i in range(iteration-1):
        filtered_image_gray_2 = apply_filter(filtered_image_gray_2, filter_3)
        # filtered_image_gray_2 = apply_filter(image_gray_noise, filter_3)
        sharpness = measure_sharpness(filtered_image_gray_2)
        snr = calculate_snr(filtered_image_gray_2, image_gray)
        x1.append(sharpness)
        x2.append(snr)
    return x1, x2, filtered_image_gray_2


def plot_values(iteration):
    # List of placeholder functions

    # # Prepare plots
    # fig, axs = plt.subplots(1, 2, figsize=(10, 15))  # 3 rows and 2 columns
    # fig.suptitle('Plots of Values from Placeholder Functions')

    x1, x2, elog_image = elog_snr_sharp(iteration)
    # x1, x2, elog_image = elog_snr_sharp_gray(iteration)
    x1_, x2_, gaus_image = gaus_snr_sharp(iteration)
    # x1_, x2_, gaus_image = gaus_snr_sharp_gray(iteration)

    x = np.arange(iteration)


    # Create a figure and two subplots (axes)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    # Set background color for the entire figure
    fig.patch.set_facecolor('#f0f0f0')

    # Plot the first set of data on the first subplot
    ax1.plot(x, x1, 'b-o', label='Sharpness ELoG')  # Blue line with circle markers
    ax1.plot(x, x1_, 'r-s', label='Sharpness Gaussian')  # Red line with square markers
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel(r'$\sigma_{\log}$', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    # Set logarithmic scale for the second subplot
    #ax1.set_yscale('log')
    equation_text = r'$\sigma_{\log}^2 = 10 \log_{10} \left( \frac{1}{N} \sum_{i=1}^{N} (\Delta I_i - \mu_{\Delta I})^2 \right)$'


    ax1.annotate(equation_text, xy=(0.55, 0.6), xycoords='axes fraction', fontsize=10,
                 ha='center', va='center', bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='black'),
                 xytext=(0, 20), textcoords='offset points')



    # Plot the second set of data on the second subplot
    ax2.plot(x, x2, 'g-^', label='SNR Differential-ELoG')  # Green line with triangle markers
    ax2.plot(x, x2_, 'm-+', label='SNR Laplacian')  # Magenta line with plus markers
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel(r'SNR', fontsize=12)
    # ax2.set_title(r'SNR: $\sigma_{\log}^2 = 10 \log_{10} \left( \frac{1}{N} \sum_{i=1}^{N} \left( \Delta I_i - '
    #               r'\frac{1}{N} \sum_{j=1}^{N} \Delta I_j \right)^2 \right)$', fontsize=8)
    ax2.legend()
    ax2.grid(True)


    # # Add the equation as a text annotation in the specific subplot
    # equation_text = r'$\sigma_{\log}^2 = 10 \log_{10} \left( \frac{1}{N} \sum_{i=1}^{N} \left( \Delta I_i - \frac{1}{N} \sum_{j=1}^{N} \Delta I_j \right)^2 \right)$'
    snr_equation = r'$SNR = 10 \log_{10} \left( \frac{P_{\text{image}}}{P_{\text{noise}}} \right)$'

    if iteration <= 50:
        ax2.annotate(snr_equation, xy=(0.27, 0), xycoords='axes fraction', fontsize=10,
                     ha='center', va='center', bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='black'),
                     xytext=(0, 20), textcoords='offset points')
    else:
        ax2.annotate(snr_equation, xy=(0.55, 0.65), xycoords='axes fraction', fontsize=10,
                     ha='center', va='center', bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='black'),
                     xytext=(0, 20), textcoords='offset points')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    #fig = plt.gcf()
    fig.savefig(f"elog_vs_gaus/diff_rgb_plot_iter:{iteration}")

    # Show the plot
    plt.show()


    # plt.clf()
    # plt.cla()
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(8, 4))  # 3 rows and 2 columns
    # fig.suptitle('Plots of Values from Placeholder Functions')
    # Set background color for the entire figure
    fig.patch.set_facecolor('white')
    # Set background color for each individual subplot
    for ax in axs.flatten():
        ax.set_facecolor('white')

    axs[0,0].imshow(image_rgb)
    # axs[0,0].imshow(image_gray, cmap='gray')
    axs[0,0].set_title('Original')
    axs[0,0].axis('off')

    axs[0,1].imshow(image_rgb_noise)
    # axs[0,1].imshow(image_gray_noise, cmap='gray')
    axs[0,1].set_title('Noisy (r = +-0.2)')
    axs[0,1].axis('off')

    axs[1,0].imshow(elog_image)
    # axs[1,0].imshow(elog_image, cmap='gray')
    axs[1,0].set_title(f'Differential-ELoG-Filter {iteration} Iterations')
    axs[1,0].axis('off')

    axs[1,1].imshow(gaus_image)
    # axs[1,1].imshow(gaus_image, cmap='gray')
    axs[1,1].set_title(f'Laplacian-Filter {iteration} Iterations')
    axs[1,1].axis('off')


    # Adjust layout to prevent overlap
    plt.tight_layout()
    #fig = plt.gcf()
    fig.savefig(f"elog_vs_gaus/diff_rgb_images_iter:{iteration}")

    # Show the plot
    plt.show()

    plt.close(fig)
    # plt.clf()
    # plt.cla()

# Call the function to plot the values
# plot_values(10)
plot_values(20)
# plot_values(30)
plot_values(40)
# plot_values(50)
plot_values(60)
# plot_values(70)
# plot_values(80)
plot_values(80)
# plot_values(90)
plot_values(100)

#%%
