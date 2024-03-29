from .robustml_utils import provider

class ProviderFactory:
    @staticmethod
    def create_provider(dataset_type, *args, **kwargs):
        """
        Factory method to create dataset provider instances.

        Parameters:
        - dataset_type (str): The type of dataset provider to create (e.g., "MNIST", "FMNIST", "GTS", "CIFAR10", "ImageNet").
        - *args: Positional arguments for the dataset provider constructor.
        - **kwargs: Keyword arguments for the dataset provider constructor.

        Returns:
        - An instance of the specified Provider subclass.
        """
        if dataset_type == 'MNIST':
            return provider.MNIST(*args, **kwargs)
        elif dataset_type == 'FMNIST':
            return provider.FMNIST(*args, **kwargs)
        elif dataset_type == 'GTS':
            return provider.GTS(*args, **kwargs)
        elif dataset_type == 'CIFAR10':
            return provider.CIFAR10(*args, **kwargs)
        elif dataset_type == 'ImageNet':
            return provider.ImageNet(*args, **kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")