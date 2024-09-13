from .robustml_utils import threat_model

class ThreatModelFactory:
    @staticmethod
    def create_threat_model(model_type, *args, **kwargs):
        """
        Factory method to create threat model instances.

        Parameters:
        - model_type (str): The type of threat model to create (e.g., "Or", "And", "L0", "L1", "L2", "Linf").
        - *args: Positional arguments for the threat model constructor.
        - **kwargs: Keyword arguments for the threat model constructor.

        Returns:
        - An instance of the specified ThreatModel subclass.
        """
        if model_type == 'Or':
            return threat_model.Or(*args)
        elif model_type == 'And':
            return threat_model.And(*args)
        elif model_type == 'L0':
            return threat_model.L0(*args, **kwargs)
        elif model_type == 'L1':
            return threat_model.L1(*args, **kwargs)
        elif model_type == 'L2':
            return threat_model.L2(*args, **kwargs)
        elif model_type == 'Linf':
            return threat_model.Linf(*args, **kwargs)
        else:
            raise ValueError(f"Unknown threat model type: {model_type}")