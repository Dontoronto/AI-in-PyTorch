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
        else:
            raise ValueError("Unbekannter Typ der Bildverarbeitung")