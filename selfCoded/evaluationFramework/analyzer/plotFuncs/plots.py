import matplotlib.pyplot as plt


@staticmethod
def plot_original_vs_observation(img_as_tensor, result, text):
    fig, ax = plt.subplots(1, 2, facecolor='dimgray')
    if img_as_tensor.shape[0] == 1:
        # Note: this is just for single channel images gray with values from 0 to 1
        ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0), cmap='gray', vmin=0, vmax=1)
    else:
        # Note: Not tested atm, have to check if image values are from 0 to 255 not 0 to 1 and maybe more
        ax[0].imshow(img_as_tensor.cpu().detach().clone().numpy().transpose(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow(result, cmap='gray')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle(text)
    plt.show()