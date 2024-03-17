import matplotlib.pyplot as plt

@staticmethod
def plot_original_vs_observation_with_text(img_as_tensor, result, text):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5*2), facecolor='dimgray')
    adapt_axes(ax[0], img_as_tensor=img_as_tensor)
    # if img_as_tensor.shape[0] == 1:
    #     # Note: this is just for single channel images gray with values from 0 to 1
    #     ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    # else:
    #     # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
    #     ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(result, cmap='gray')
    ax[1].axis('off')
    plt.tight_layout()
    Header = 'Annual Report of XXX Factory'
    fig.suptitle(text)
    plt.annotate(Header, (.22,.98), weight='regular', fontsize=20, alpha=.6 )
    plt.show()
@staticmethod
def plot_original_vs_observation(img_as_tensor, result, text):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5*2), facecolor='dimgray')
    adapt_axes(ax[0], img_as_tensor=img_as_tensor)
    # if img_as_tensor.shape[0] == 1:
    #     # Note: this is just for single channel images gray with values from 0 to 1
    #     ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    # else:
    #     # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
    #     ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(result, cmap='gray')
    ax[1].axis('off')
    plt.tight_layout()
    Header = 'Annual Report of XXX Factory'
    fig.suptitle(text)
    plt.annotate(Header, (.22,.98), weight='regular', fontsize=20, alpha=.6 )
    plt.show()

@staticmethod
def plot_model_comparison(input_tensor_images: list, model_results : list):
    '''
    this method is for plotting a model comparison (you can use activation maps or feature maps and
    visualize how the different models behave (currently just one visualizing per model)
    :param input_tensor_images: list of tensor images, list should have shape (x)
    :param model_results: list of model results, list should have shape (len(input_tensor_image), amount of models)
    :return:
    '''
    # Create a figure and axes with a grid of 3 columns (input + output from each model)
    fig, axes = plt.subplots(nrows=len(input_tensor_images), ncols=len(model_results[0]) + 1,
                             figsize=(15, 5*len(input_tensor_images)), facecolor='dimgray')

    # Iterate over the images and plot them with the corresponding model outputs
    for i in range(len(input_tensor_images)):
        if len(input_tensor_images) == 1:
            # Plot the input image
            adapt_axes(axes[i], img_as_tensor=input_tensor_images[i])
            axes[i].axis('off')  # Hide axes

            # Plot each model's output
            for j in range(len(model_results[i])):
                axes[j+1].imshow(model_results[i][j], cmap='gray')
                axes[j+1].axis('off')  # Hide axes
        else:
            # Plot the input image
            adapt_axes(axes[i, 0], img_as_tensor=input_tensor_images[i])
            axes[i, 0].axis('off')  # Hide axes

            # Plot each model's output
            for j in range(len(model_results[i])):
                axes[i, j+1].imshow(model_results[i][j], cmap='gray')
                axes[i, j+1].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show()

@staticmethod
def adapt_axes(axes, img_as_tensor):
    '''
    helping function to identify if model has 1 or 3 channels. if it has only one channel it is a
    black and white image with values in between 0 and 1
    :param axes: object of the specific axes which needs to implement the tensor image
    :param img_as_tensor: the image in tensor shape (Channel, Width?, Height?) format with unknown amount of channels
    '''
    if img_as_tensor.shape[0] == 1:
        # Note: this is just for single channel images gray with values from 0 to 1
        axes.imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    else:
        # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
        axes.imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0))
