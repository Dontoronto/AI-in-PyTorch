class EvaluationMapsStrategy:
    def analyse(self, model, original_image, single_batch, **kwargs):
        raise NotImplementedError("Jede spezifische Visualisierung muss dieses Strategy Pattern implementieren")