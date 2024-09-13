import torch
from torchvision.models import resnet18

from distributionDensity import calculate_distribution_density, plot_distribution_density
import modelWrapper


def density_evaluation(self, bins=None, density_range=None, log_scale=False, title=None):
    '''
    Berechnet und stellt das Verteilungsdiagramm der Modellgewichte dar

    - bins (int): Die Anzahl der Bins für das Histogramm.
    - density_range (tuple): Ein Tupel (min, max) zur Beschränkung des Wertebereichs.
    - log_scale (bool): Wenn True, wird die y-Achse logarithmisch skaliert.
    '''
    density, bin_edges = calculate_distribution_density(self.model, bins, density_range)
    if title is None:
        title = 'Model'

    fig_density = plot_distribution_density(density, bin_edges, log_scale, title=title)
    return fig_density

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# _weights = ResNet18_Weights.IMAGENET1K_V1
# _model = resnet18(_weights)
_model = resnet18()
Model = modelWrapper.ModelWrapper(_model)
Model.load_state_dict(torch.load("Base Model.pth", map_location=device))

fig = density_evaluation(Model, density_range=(-0.2, 0.2), log_scale=False, title='Base Model')

fig.savefig("base_model_density.png")

Model.load_state_dict(torch.load("Trivial Default.pth", map_location=device))
fig = density_evaluation(Model, density_range=(-0.2, 0.2),log_scale=False, title='Trivial Default')

fig.savefig("trivial_default_density.png")

Model.load_state_dict(torch.load("SCP Adv.pth", map_location=device))
fig = density_evaluation(Model, density_range=(-0.2, 0.2),log_scale=False, title='SCP Adv')

fig.savefig("scp_adv_density.png")

Model.load_state_dict(torch.load("Unstruct Default.pth", map_location=device))
fig = density_evaluation(Model, density_range=(-0.2, 0.2),log_scale=False, title='Unstruct Default')

fig.savefig("unstruct_default_density.png")

#%%
