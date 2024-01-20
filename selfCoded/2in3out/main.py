import sys, os
sys.path.append(os.getcwd())
import fixedPattern
import sampleMethod


from mlp import MLP

import torch
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    # Initialisierung von Sample und Labels (Training)
    test_sample = list()
    output = list()

    # Sample Erstellung (Training)
    for i in range(11):
        for j in range(11):
            test_sample.append([i/10, j/10])

    Pattern = fixedPattern.Pattern(sampleMethod.classify)

    #Label Erstellung (Training)
    for elem in test_sample:
        out = Pattern.execute_function(elem)
        output.append(out)


    # Tensor aus Sample und Labels erstellen
    inputs = torch.tensor(test_sample, dtype=torch.float32)
    targets = torch.tensor(output, dtype=torch.float32)

    # Instantiate the model, loss function, and optimizer
    model = MLP()

    # Training loop
    epochs = 100000
    for epoch in range(epochs):
        model.train(inputs, targets)

    # Evaluation Data generation
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    grid_inputs.requires_grad_(True)
    grid_outputs = model(grid_inputs)

    # Output und Gradient von Model mit Evaluationsdaten darstellen
    split_outputs, split_gradients = model.plot_inference_output_gradient(xx,yy,grid_inputs,grid_outputs)


    ## Salience Map Calculations
    for i in range(3):
        saliencemap = model.plot_salience_map(xx,yy,split_gradients,grid_inputs.detach().numpy(),category=i)
        saliencemap_norm = saliencemap / (np.max(saliencemap) if np.max(saliencemap) > 0 else 1)
        plot = plt.axes(projection='3d')

        plot.plot_surface(xx, yy, saliencemap_norm.reshape(xx.shape), cmap='viridis')
        plot.set_title('Salience Map Klasse {}'.format(i+1))
        plot.set_xlabel('Input 1')
        plot.set_zlabel('Salience{}'.format(i+1))
        plot.set_ylabel('Input 2')
        plot.legend("platzhalter")
        plt.show()



#%%

#%%
