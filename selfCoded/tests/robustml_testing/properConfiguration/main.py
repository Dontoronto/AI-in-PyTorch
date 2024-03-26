import sys, os
sys.path.append(os.getcwd())
import robustml
from torchvision.models import resnet18
import robustModel
import attack_custom
from torch import nn, hub
import torchvision





def main():
    #_model = resnet18(pretrained=True).eval()

    # Load the pre-trained EfficientNet-B0 model
    # _model = torchvision.models.efficientnet_b0(pretrained=True)
    #
    # # Modify the classifier for CIFAR-10 (10 classes)
    # num_features = _model.classifier[1].in_features  # Get the input features of the last layer
    # _model.classifier[1] = nn.Linear(num_features, 10)  # Replace the last layer
    # _model.eval()

    _model = hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    _model.eval()

    model = robustModel.RobustModel(model=_model, dataset_name="CIFAR10", threat_model=robustml.threat_model.L2(epsilon=0.3))
    robust_attack = attack_custom.Cifar10PGD(model=_model,epsilon= 8/255, alpha=2/255, max_steps=40, show_images=False)
    provider = robustml.provider.CIFAR10("/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/dataset/cifar-10/cifar-10-batches-py/test_batch")
    start = 0
    end = 100

    rate = robustml.evaluate.evaluate(
        model,
        robust_attack,
        provider,
        start,
        end,
        deterministic=False,
        debug=True
    )

    print("End:")
    print(rate)



if __name__ == '__main__':
    main()


#%%

#%%
