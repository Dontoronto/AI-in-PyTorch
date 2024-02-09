import configParser

# TODO: hier hier für jeden Abschnitt eigenen loader erstellen
class Configurator:
    def __init__(self):

        self.ConfigParser = configParser.ConfigParser()

    def loadDataHandlerConfig(self):
        self.configHandlerData = self.ConfigParser.getDataHandlerConfig()
        return self.configHandlerData