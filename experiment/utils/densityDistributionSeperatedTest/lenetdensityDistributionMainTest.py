import torch
from torchvision.models import resnet18

from distributionDensity import calculate_distribution_density, plot_distribution_density
import modelWrapper
from lenet import LeNet


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
_model = LeNet()
Model = modelWrapper.ModelWrapper(_model)
Model.load_state_dict(torch.load("lenet/Base Model.pth", map_location=device))

fig = density_evaluation(Model, density_range=(-0.5, 0.5), log_scale=False, title='Base Model')

fig.savefig("lenet_base_model_density.png")

Model.load_state_dict(torch.load("lenet/Trivial Adv.pth", map_location=device))
fig = density_evaluation(Model, density_range=(-0.5, 0.5),log_scale=False, title='Trivial Adv')

fig.savefig("lenet_trivial_adv_density.png")

Model.load_state_dict(torch.load("lenet/SCP Default.pth", map_location=device))
fig = density_evaluation(Model, density_range=(-0.5, 0.5),log_scale=False, title='SCP Default')

fig.savefig("lenet_scp_default_density.png")

Model.load_state_dict(torch.load("lenet/Unstruct Adv.pth", map_location=device))
fig = density_evaluation(Model, density_range=(-0.5, 0.5),log_scale=False, title='Unstruct Adv')

fig.savefig("lenet_unstruct_adv_density.png")

#%%
