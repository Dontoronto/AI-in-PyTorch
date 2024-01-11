def singleton(class_):
    instances = {}
    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance


@singleton
class Pattern:
    def __init__(self, pattern_method):
        self.pattern_method = pattern_method

    def execute_function(self, *args, **kwargs):
        return self.pattern_method(*args, **kwargs)

    def load_method(self, pattern_method):
        self.pattern_method = pattern_method


#%%
