import robustml.threat_model


class ThreatModel(robustml.threat_model.ThreatModel):

    def check(self, original=None, perturbed=None):
        '''
        Returns whether the perturbed image is a valid perturbation of the
        original under the threat model.

        `original` and `perturbed` are numpy arrays of the same dtype and
        shape.
        '''
        return True

    @property
    def targeted(self):
        '''
        Returns whether the threat model only includes targeted attacks
        (requiring the attack to be capable of synthesizing targeted
        adversarial examples).
        '''
        return None