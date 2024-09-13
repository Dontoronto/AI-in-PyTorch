# main.py
import argparse
import warnings
import time

import torchvision.datasets
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Failed to load image Python extension")
import sys, os
sys.path.append(os.getcwd())
import torch
import matplotlib

import modelWrapper
import dataHandler
import configurator
import analyzer.analyzer as analyzer

import IOComponent.transformators.transformators as transformators

from Trainer.trainerFactory import TrainerFactory
from SharedServices.logging_config import setup_logging
from SharedServices.modelArchitectureExtractor import ModelArchitectureExtractor
from SharedServices.utils import create_missing_folders
# from torchvision.datasets import ImageNet
from torchvision.datasets import ImageFolder


setup_logging()
import logging
logger = logging.getLogger(__name__)
from torchvision.models import resnet18, ResNet18_Weights

# NOTE: currently we are testing with LeNet-model
from models.lenet import LeNet

def check_filepath(path):
    """
    Überprüft, ob eine Datei mit "._" im Dateinamen beginnt.

    Parameter:
    path (str): Der gesamte Pfad der Datei.

    Rückgabe:
    bool: False, wenn der Dateiname mit "._" beginnt, True ansonsten.
    """
    filename = os.path.basename(path)
    return not filename.startswith("._")

def main(args):
    # Variable assignments inside the main method
    target_model = args.model
    model_path = args.model_path
    cuda_enabled = args.cuda_enabled
    disable_analysis_training = args.disable_analysis_training
    adv_training = args.adv_training
    adv_training_ratio = args.adv_training_ratio
    tensorboard_enable = args.tensorboard_enable
    optimization_testing_monitoring_enable = args.optimization_testing_monitoring_enable
    create_architecture_config = args.create_architecture_config
    adv_optimization_without_analysis = args.adv_optimization_without_analysis

    # Now you can use these variables for further logic
    print(f"Model: {target_model}")
    print(f"Model Path: {model_path}")
    print(f"CUDA Enabled: {cuda_enabled}")
    print(f"Disable Training: {disable_analysis_training}")
    print(f"Adversarial Training: {adv_training}")
    print(f"Adversarial Training Ratio: {adv_training_ratio}")
    print(f"Tensorboard Enable: {tensorboard_enable}")
    print(f"Optimization Testing Monitoring Enable: {optimization_testing_monitoring_enable}")
    print(f"Create Architecture Config: {create_architecture_config}")
    print(f"Adv Optimization Without Analysis: {adv_optimization_without_analysis}")

    matplotlib.use('Agg')  # Use a non-interactive backend

    # ====================== Note: This code is just for exporting onnx model for tvm
    # _model = LeNet()
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.load_state_dict(torch.load("LeNet_pat_con_fc_admm_retrain.pth"))
    # input_names = ["input"]
    # output_names = ["output"]
    # torch.onnx.export(Model.model,torch.randn(1,1,28,28),'model_opset.onnx', opset_version=11,
    #                   input_names=input_names, output_names=output_names)

    # ============ Note: ResNet onnx export
    # _model = models.resnet18()
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.load_state_dict(torch.load("experiment/temp/ResNet_tvm_example/ResNet18.pth"))
    # input_names = ["input"]
    # output_names = ["output"]
    # torch.onnx.export(Model.model,torch.randn(1,3,224,224),'experiment/temp/ResNet_tvm_example/resnet_default.onnx',
    #                   opset_version=11, input_names=input_names, output_names=output_names)
    #
    # return

    # ================================



    # ResNet18 settings
    if target_model == "ResNet18":
        _weights = ResNet18_Weights.IMAGENET1K_V1
        _model = resnet18(_weights)
        if cuda_enabled is True:
            _model.to('cuda')
        Model = modelWrapper.ModelWrapper(_model)
        if model_path is not None:
            Model.load_state_dict(torch.load(model_path))
        # Model.load_state_dict(torch.load(os.path.join("experiment\\LeNet\\cic\\ResNet\\"
        #                                               "elog_adv_double_v2_cic", "admm_checkpoint.pth")))
        if cuda_enabled is True:
            Model.to('cuda')
        Model.eval()


        Configurator = configurator.Configurator()
        DataHandler = dataHandler.DataHandler(Configurator)
        DataHandler.setAdversarialTransformer(transformators.adv_imagenet_transformer())


        try:
            device = next(Model.parameters()).device
            if device.type == 'cuda':
                torch.set_default_device('cuda')
            print(f"Device= {device}")
        except Exception:
            device = None
            print("Failed to set device automatically, please try set_device() manually.")



    start_time = time.time()
    warnings.filterwarnings(action='once')


    # ================= Note: LeNet
    # LeNet Test
    if target_model == "LeNet":
        _model = LeNet()
        Model = modelWrapper.ModelWrapper(_model)
        # Model.load_state_dict(torch.load("LeNet_admm_train.pth"))
        if model_path is not None:
            Model.load_state_dict(torch.load(model_path))
        if cuda_enabled is True:
            Model.to('cuda')
        Model.eval()

        Configurator = configurator.Configurator()
        DataHandler = dataHandler.DataHandler(Configurator)

        DataHandler.setTransformer(Configurator.loadTransformerNEW())

        #DataHandler.setAdversarialTransformer(transformators.mnist_transformer())
        #
        #
        try:
            device = next(Model.parameters()).device
            if device.type == 'cuda':
                torch.set_default_device('cuda')
            print(f"Device= {device}")
        except Exception:
            device = None
            print("Failed to set device automatically, please try set_device() manually.")

    # ==========================================

    if create_architecture_config is True:
        ArchitecctureExtractor = ModelArchitectureExtractor()
        ArchitecctureExtractor.extractLayers(Model, "ArchitectureExport")

    Analyzer = analyzer.Analyzer(Model, DataHandler)


    logger.critical(f"Time for algo is: {time.time()-start_time}")

    # =============== Note: Run ResNet18 Tuning
    #
    Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    Trainer.setDataLoaderSettings(Configurator.loadDataloaderConfig())
    Trainer.setSnapshotSettings(Configurator.loadSnapshotConfig())
    Trainer.setADMMArchitectureConfig(Configurator.loadConfigFromRegistry("admm_model_architecture"))
    Trainer.setADMMConfig(Configurator.loadConfigFromRegistry("admm_settings"))
    #Trainer.setTensorBufferConfig(Configurator.loadConfigFromRegistry("tensor_buffer"))
    #Trainer.train(test=False)

    Analyzer.setAnalyzerConfig(Configurator.loadConfigFromRegistry("analyzer"))

    Analyzer.init_adversarial_environment(False)
    Analyzer.set_threat_model_config(Configurator.loadConfigFromRegistry("adversarial_threat_model"))
    Analyzer.set_provider_config(Configurator.loadConfigFromRegistry("adversarial_provider"))
    Analyzer.set_attack_type_config(Configurator.loadConfigFromRegistry("adversarial_attacks"))
    Analyzer.set_adv_dataset_generation_settings()

    Analyzer.setTrainer(Trainer)

    if adv_training is True:
        atk = Analyzer.getSingleAttack(Model, "PGD" , eps=1.0, alpha=2/255, steps=10, random_start=False)
        Trainer.setAdversarialTraining(atk, adv_training_ratio)

    train_kwargs = {
        'test': optimization_testing_monitoring_enable,
        'tensorboard': tensorboard_enable
    }
    # # create_missing_folders("experiment\\adv_data\\ResNet18\\deepfool_only_adv_success\\adv_image_generation\\adv_images",
    # #                        1000)
    # # create_missing_folders("experiment\\adv_data\\ResNet18\\deepfool_only_adv_success\\adv_image_generation\\original_images",
    # #                        1000)

    if disable_analysis_training is False:
        run_training = True
    else:
        run_training = False

    if adv_optimization_without_analysis is True:
        Trainer.train(**train_kwargs)
    else:
        Analyzer.startTestrun(train_kwargs, run_training=run_training)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Which mode should be activated")

    # Required argument
    parser.add_argument('--model', type=str, required=True, help='Model to optimize')

    # Optional arguments with default values
    parser.add_argument('--model_path', type=str, default=None, help='Path to the pretrained model')
    parser.add_argument('--cuda_enabled', action='store_true', help='Enable CUDA')
    parser.add_argument('--disable_analysis_training', action='store_false', help='Disable training (default is on)')
    parser.add_argument('--adv_training', action='store_true', help='Enable adversarial training (default is off)')
    parser.add_argument('--adv_training_ratio', type=float, default=0.2, help='Adversarial training ratio (default is 0.2)')
    parser.add_argument('--tensorboard_enable', action='store_true', help='Enable Tensorboard (default is off)')
    parser.add_argument('--optimization_testing_monitoring_enable', action='store_true', help='Enable optimization testing monitoring (default is off)')
    parser.add_argument('--create_architecture_config', action='store_true', help='Create architecture config inside a folder in working directory (default is off)')
    parser.add_argument('--adv_optimization_without_analysis', action='store_true', help='Adversarial optimization without analysis (default is off)')

    # Parse the arguments
    args = parser.parse_args()

    # call python main.py --model LeNet --model_path LeNet_admm_train.pth --cuda_enabled --disable_analysis_training --adv_training --adv_training_ratio 0.1 --optimization_testing_monitoring_enable
    # call python main.py --model ResNet18 --cuda_enabled --disable_analysis_training --optimization_testing_monitoring_enable

    # mock_args = argparse.Namespace(
    #     model="LeNet",  # Required argument, manually provided
    #     # model="ResNet18",  # Required argument, manually provided
    #     model_path="LeNet_admm_train.pth",  # Default: None (no pretrained model path provided)
    #     # model_path=None,
    #     cuda_enabled=True,  # Default: CUDA disabled
    #     disable_analysis_training=False,  # Default: Training enabled (so disable_training is False)
    #     adv_training=True,  # Default: Adversarial training disabled
    #     adv_training_ratio=0.1,  # Default: 0.2
    #     tensorboard_enable=False,  # Default: Tensorboard disabled
    #     optimization_testing_monitoring_enable=True,  # Default: Monitoring disabled
    #     create_architecture_config=False,  # Default: Do not create architecture config
    #     adv_optimization_without_analysis=False  # Default: Do not optimize without analysis
    # )

    # Pass the args object to main
    # main(mock_args)
    main(args)


#%%
