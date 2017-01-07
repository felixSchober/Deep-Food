from classification.model import Model

class neural_net_model(Model):

    def __init__(self):
        super(neural_net_model, self).__init__()
        self.set_network_specific_settings()
        self.net = None

    def set_network_specific_settings(self):
        raise NotImplementedError("abstract class")

    def train(self, samples, labels):
        return self.net.fit(samples, labels)

    def predict(self, samples):
        self.net.net.predict_proba(samples)


