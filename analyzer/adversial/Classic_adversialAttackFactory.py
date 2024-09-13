import torchattacks


class AdversarialAttackerFactory:
    @staticmethod
    def create_attacker(model, typ, **kwargs):
        if typ == "DeepFool":
            return torchattacks.DeepFool(model, **kwargs)
        elif typ == "PGD":
            return torchattacks.PGD(model, **kwargs)
        elif typ == "JSMA":
            return torchattacks.JSMA(model, **kwargs)
        elif typ == "OnePixel":
            return torchattacks.OnePixel(model, **kwargs)
        elif typ == "SparseFool":
            return torchattacks.SparseFool(model, **kwargs)
        else:
            raise ValueError("Unbekannter Typ der Bildverarbeitung")