# adapter.py
class DictionaryAdapter:
    def __init__(self, instance, args_dict):
        for key, value in args_dict.items():
            setattr(instance, key, value)

