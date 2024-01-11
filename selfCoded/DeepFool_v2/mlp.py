import torch.nn as nn
import torch.optim as optim
import pandas
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()


        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2, 3)),
            ('sigmoid1', nn.Sigmoid()),
            ('fc2', nn.Linear(3,3)),
            ('sigmoid2', nn.Sigmoid())])
        )

        #self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = nn.MSELoss()
        #self.loss_function = nn.CrossEntropyLoss()

        self.optimiser = optim.SGD(self.parameters(), lr=0.1)


        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)


    def train_mlp(self,inputs, targets):

        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, targets)

        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward(retain_graph=True)
        self.optimiser.step()

        return self


    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass

    def plot_inference_output(self, xx, yy, grid_outputs):

        z = list()
        row = grid_outputs.size()[0]
        col = grid_outputs.size()[1]

        for i in range(col):
            z.append(torch.select(grid_outputs, 1,i ).detach().numpy())
            plot = plt.axes(projection='3d')
            plot.plot_surface(xx, yy, z[i].reshape(xx.shape), cmap='viridis')
            plot.set_title('Output Klasse {}'.format(i+1))
            plot.set_xlabel('Input 1')
            plot.set_zlabel('OutputProbability_Output{}'.format(i+1))
            plot.set_ylabel('Input 2')
            plot.legend("platzhalter")
            plt.show()

    def plot_inference_gradient(self,xx,yy,grid_inputs, grid_outputs):

        z = list()
        row = grid_outputs.size()[0]
        col = grid_outputs.size()[1]

        inference_plot = list()
        for i in range(col):
            derivative = torch.autograd.grad(torch.select(grid_outputs, 1,i ),
                    grid_inputs,retain_graph=True,
                    grad_outputs=torch.ones_like(torch.select(grid_outputs, 1,i )))[0]
            norms = torch.linalg.vector_norm(derivative, ord=2, dim=1, keepdim=True).detach().numpy()
            z.append(norms)
            plot = plt.axes(projection='3d')
            plot.plot_surface(xx, yy, z[i].reshape(xx.shape), cmap='viridis')
            plot.set_title('Output Klasse {}'.format(i+1))
            plot.set_xlabel('Input 1')
            plot.set_zlabel('NormGradient_Output{}'.format(i+1))
            plot.set_ylabel('Input 2')
            plot.legend("platzhalter")
            plt.show()

    # In dieser Methode wird der Vorzeichenbehaftete unnormierte Gradient zurück gegeben
    def plot_inference_output_gradient(self,xx,yy,grid_inputs, grid_outputs):

        outputs = list()
        gradients = list()
        row = grid_outputs.size()[0]
        col = grid_outputs.size()[1]

        for i in range(col):
            outputs.append(torch.select(grid_outputs, 1,i ).detach().numpy())
            plot = plt.axes(projection='3d')
            plot.plot_surface(xx, yy, outputs[i].reshape(xx.shape), cmap='viridis')
            plot.set_title('Output Klasse {}'.format(i+1))
            plot.set_xlabel('Input 1')
            plot.set_zlabel('OutputProbability_Output{}'.format(i+1))
            plot.set_ylabel('Input 2')
            plot.legend("platzhalter")
            plt.show()


            derivative = torch.autograd.grad(torch.select(grid_outputs, 1,i ),
                                             grid_inputs,retain_graph=True,
                                             grad_outputs=torch.ones_like(torch.select(grid_outputs, 1,i )))[0]
            norms = torch.linalg.vector_norm(derivative, ord=2, dim=1, keepdim=True).detach().numpy()
            gradients.append(derivative.detach().numpy())
            plot = plt.axes(projection='3d')
            plot.plot_surface(xx, yy, norms.reshape(xx.shape), cmap='viridis')
            plot.set_title('Gradient Klasse {}'.format(i+1))
            plot.set_xlabel('Input 1')
            plot.set_zlabel('NormGradient_Output{}'.format(i+1))
            plot.set_ylabel('Input 2')
            plot.legend("platzhalter")
            plt.show()

        return outputs, gradients

    def plot_salience_map(self,xx,yy,gradients,grid_inputs, category=0):

        length_array = len(gradients[category])
        salience_arr = np.zeros((length_array,1), dtype=np.float32)

        for i in range(len(gradients[category])):
            target_class = np.dot(gradients[category][i],grid_inputs[i])
            sum_non_target_class = 0

            for j in range(len(gradients)):
                if j != category:
                    sum_non_target_class += np.dot(gradients[j][i],grid_inputs[i])

            if(target_class < 0 or sum_non_target_class > 0):
                salience_arr[i] = 0
            else:
                salience_arr[i] = target_class * abs(sum_non_target_class)

        return salience_arr








