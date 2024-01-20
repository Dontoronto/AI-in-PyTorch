import sys, os
sys.path.append(os.getcwd())
import fixedPattern
import sampleMethod


from mlp import MLP

import torch
import numpy as np
import matplotlib.pyplot as plt

#DeepFoolPretrained_not_working
from torchattacks.attacks.deepfool import DeepFool

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
    epochs = 1000000
    for epoch in range(epochs):
        model.train_mlp(inputs, targets)

    # Evaluation Data generation
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    model.eval()

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_inputs = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    grid_outputs = model(grid_inputs)

    # Output und Gradient von Model mit Evaluationsdaten darstellen
    #split_outputs, split_gradients = model.plot_inference_output_gradient(xx,yy,grid_inputs,grid_outputs)


    ## Salience Map Calculations
    # for i in range(3):
    #     saliencemap = model.plot_salience_map(xx,yy,split_gradients,grid_inputs.detach().numpy(),category=i)
    #     saliencemap_norm = saliencemap / (np.max(saliencemap) if np.max(saliencemap) > 0 else 1)
    #     plot = plt.axes(projection='3d')
    #
    #     plot.plot_surface(xx, yy, saliencemap_norm.reshape(xx.shape), cmap='viridis')
    #     plot.set_title('Salience Map Klasse {}'.format(i+1))
    #     plot.set_xlabel('Input 1')
    #     plot.set_zlabel('Salience{}'.format(i+1))
    #     plot.set_ylabel('Input 2')
    #     plot.legend("platzhalter")
    #     plt.show()
    model.eval()


    ### Here starts DeepFoolPretrained_not_working
    attack = DeepFool(model, steps=50, overshoot=0.02)
    print(attack)

    # für deepfool brauchen wir den Index des max-Values des Model-Outputs
    _, idx = torch.max(grid_outputs, dim=1)

    #####
    ## Targeted Attacks  sind bei DeepFool leider nicht möglich aber dafür bei anderen wie z.B. PGD
    #####
    from torchattacks.attacks.pgd import PGD
    # atk = PGD(model, eps=20/255, alpha=8/225, steps=20, random_start=True)
    # print(atk)
    # atk.set_mode_targeted_by_function(target_map_function=lambda images, idx:(idx-idx+1))
    # adv_images = atk(grid_inputs.clone().detach(), idx)

    attack.set_mode_targeted_by_function(target_map_function=lambda images, idx:(idx+1)%3) #(idx+1)%3
    adv_images = attack(grid_inputs.clone().detach(), idx)
    print("Ende")
    model.plot_inference_output(xx,yy,grid_outputs)
    adv_label = model(adv_images)
    model.plot_inference_output(xx,yy,adv_label)
    print("Ende Ende")






#%%

#%%
