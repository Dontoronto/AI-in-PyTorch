def singleton(class_):
    """
    This function is used as Decorator to modifies a class to be a singleton
    -> Only one instance of the class have to exist
    """
    instances = {}
    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance