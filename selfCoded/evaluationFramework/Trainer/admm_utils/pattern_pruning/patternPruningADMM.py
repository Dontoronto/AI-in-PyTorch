# TODO: methode schreiben, welche die Pattern Pruning Methoden verwendet um bei jeder ADMM Iteration
# TODO: eine neue Maske zu erstellen und die verfügbaren Patterns der pattern library zu reduzieren

from patternManager import PatternManager

class PatternPruning:

    def __init__(self):
        self.patternManager = PatternManager

    def get_mask_list(self):
        return self.patternManager.get_pattern_masks()

    def update_pattern_assignments(self, tensor_list, min_amount_indices=12):
        self.patternManager.update_pattern_assignments(tensor_list, min_amount_indices)
