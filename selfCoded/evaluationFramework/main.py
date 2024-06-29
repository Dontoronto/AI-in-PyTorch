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

#

def main():

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


    # ============== Note: Other Model testrun

    # Resnet settings work
    # _weights = ResNet101_Weights.IMAGENET1K_V1
    # _model = resnet101(weights=_weights)
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.eval()


    # AlexNet settings work
    # _weights = models.AlexNet_Weights.IMAGENET1K_V1
    # _model = models.alexnet(_weights)
    # _model.to('cuda')
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.to('cuda')
    # Model.eval()

    # ResNet18 settings
    _weights = ResNet18_Weights.IMAGENET1K_V1
    _model = resnet18()
    #_model = resnet18(_weights)
    _model.to('cuda')
    Model = modelWrapper.ModelWrapper(_model)
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


    # Note: delete
    # img = DataHandler.loadImage("D:\\imagenet\\val\\n01440764\\ILSVRC2012_val_00000293.JPEG")
    # img_tensor = DataHandler.preprocessBatched(img)
    # img_numpy = img_tensor.numpy().astype("float32")
    # import numpy as np
    # print(img_numpy.shape)
    # np.savez("experiment/temp/ResNet_tvm_example/imagenet_test", input=img_numpy)
    #
    # # output = Model(img_tensor)
    # # out = output.squeeze(0).argmax().item()
    # # print(out)
    #
    # return







    #DataHandler.setTransformer(transformators.imagenet_transformer(image_flag=True))
    # DataHandler.setTransformer(_weights.transforms())


    # ------- rest ist testing only
    # img = DataHandler.loadImage("testImages/desk.jpeg")
    # batch = DataHandler.preprocessBatched(img)
    #
    # prediction = Model(batch)
    #
    # utils.alexnet_prediction_evaluation(prediction, 5)
    #
    # imagenet = ImageFolder("/Volumes/Extreme SSD/datasets/imagenet/imagenet1k/train/",
    #                        transform=DataHandler.getTransformer(),
    #                        is_valid_file=lambda path: not os.path.basename(path).startswith("._"))
    #
    # generator1 = torch.Generator().manual_seed(42)
    # train_set, val_set = torch.utils.data.random_split(imagenet, [0.7, 0.3], generator1)
    #
    # dataset_len = len(train_set)
    #
    #
    # loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=1)
    #
    # # Initialisiere tqdm
    # progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training Progress")
    #
    # for batch_idx, (image, label) in progress_bar:
    #     prediction = Model(image)
    #
    #     #utils.alexnet_prediction_evaluation(prediction, 1)
    #     #if batch_idx == 100:
    #         #logger.critical(f"{batch_idx*32/dataset_len}")
    #
    #
    #
    # return
    #

    # =======================================


    start_time = time.time()
    warnings.filterwarnings(action='once')


    # ================= Note: this is the standard model for this thesis
    # LeNet Test
    # _model = LeNet()
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.load_state_dict(torch.load("LeNet_admm_train.pth"))
    # Model.eval()
    #
    # Configurator = configurator.Configurator()
    # DataHandler = dataHandler.DataHandler(Configurator)
    #
    # DataHandler.setTransformer(Configurator.loadTransformerNEW())

    # ==========================================



    # =================== Note: basic block for working in the past with lenet
    # Configurator = configurator.Configurator()
    # DataHandler = dataHandler.DataHandler(Configurator)
    #
    #
    #   Note: weights without configurator was for pretrained downloaded models
    # #DataHandler.setTransformer(_weights.transforms())
    # DataHandler.setTransformer(Configurator.loadTransformer())
    # =================================


    #img = DataHandler.loadImage("testImages/tisch_v2.jpeg")



    Analyzer = analyzer.Analyzer(Model, DataHandler)


    # ================== Note: this part is for generating adversarial examples
    # ========== Note: MNIST Adv. Generation
    # Analyzer.setAnalyzerConfig(Configurator.loadConfigFromRegistry("analyzer"))
    # DataHandler.setTransformer(transformators.mnist_transformer())
    # Analyzer.init_adversarial_environment(False)
    # Analyzer.set_threat_model_config(Configurator.loadConfigFromRegistry("adversarial_threat_model"))
    # Analyzer.set_provider_config(Configurator.loadConfigFromRegistry("adversarial_provider"))
    # Analyzer.set_attack_type_config(Configurator.loadConfigFromRegistry("adversarial_attacks"))
    # Analyzer.set_adv_dataset_generation_settings()
    #
    # #Analyzer.select_attacks_from_config(0, 1)
    # #Analyzer.enable_adversarial_saving("experiment/adv_data/windows/ResNet18/adv_samples")
    # #Analyzer.enable_original_saving("experiment/adv_data/windows/ResNet18//orig_samples")
    #
    # # test1 = Analyzer.start_adversarial_evaluation(0, 10)
    # test1 = Analyzer.start_adversarial_evaluation_preconfigured()

    # print(f"First evaluation:")
    # print(test1)
    #
    # return

    # ========== Note: Imagenet Adv. Generation
    # Analyzer.setAnalyzerConfig(Configurator.loadConfigFromRegistry("analyzer"))
    # DataHandler.setTransformer(transformators.adv_imagenet_transformer())
    # Analyzer.init_adversarial_environment(False)
    # Analyzer.set_threat_model_config(Configurator.loadConfigFromRegistry("adversarial_threat_model"))
    # Analyzer.set_provider_config(Configurator.loadConfigFromRegistry("adversarial_provider"))
    # Analyzer.set_attack_type_config(Configurator.loadConfigFromRegistry("adversarial_attacks"))
    # Analyzer.set_adv_dataset_generation_settings()
    #
    # #Analyzer.select_attacks_from_config(0, 1)
    # #Analyzer.enable_adversarial_saving("experiment/adv_data/windows/ResNet18/adv_samples")
    # #Analyzer.enable_original_saving("experiment/adv_data/windows/ResNet18//orig_samples")
    #
    # # test1 = Analyzer.start_adversarial_evaluation(0, 10)
    # test1 = Analyzer.start_adversarial_evaluation_preconfigured()
    #
    # print(f"First evaluation:")
    # print(test1)
    #
    # return


    # ========================

    # Analyzer.init_adversarial_environment()
    # Analyzer.set_threat_model_config(Configurator.loadConfigFromRegistry("adversarial_threat_model"))
    # Analyzer.set_provider_config(Configurator.loadConfigFromRegistry("adversarial_provider"))
    # Analyzer.set_attack_type_config(Configurator.loadConfigFromRegistry("adversarial_attacks"))
    # Analyzer.select_attacks_from_config(1, 1)
    # Analyzer.enable_adversarial_saving("experiment/PGD/adversarial_images")
    # Analyzer.enable_original_saving("experiment/PGD/original_images")
    #
    # test1 = Analyzer.start_adversarial_evaluation(0, 100)
    # print(f"First evaluation:")
    # print(test1)

    # Analyzer.init_adversarial_environment()
    # Analyzer.set_threat_model_config(Configurator.loadConfigFromRegistry("adversarial_threat_model"))
    # Analyzer.set_provider_config(Configurator.loadConfigFromRegistry("adversarial_provider"))
    # Analyzer.set_attack_type_config(Configurator.loadConfigFromRegistry("adversarial_attacks"))
    # Analyzer.select_attacks_from_config(2, 1)
    # Analyzer.enable_adversarial_saving("experiment/JSMA/adversarial_images")
    # Analyzer.enable_original_saving("experiment/JSMA/original_images")
    # test1 = Analyzer.start_adversarial_evaluation(0, 100)
    # print(f"First evaluation:")
    # print(test1)
    #


    # =====================================================

    # ------------- Note: test of adv
    # Model.load_state_dict(torch.load("experiment/LeNet/v8_post_cuda/placeholder/LeNet.pth"))
    # adv_dataset = DataHandler.create_imageFolder_dataset("experiment/adversarial_data/LeNet/DeepFool/run_1/adv_image_generation/adv_images")
    # orig_dataset = DataHandler.create_imageFolder_dataset("experiment/adversarial_data/LeNet/DeepFool/run_1/adv_image_generation/original_images")
    # Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    # adv_dataloader = Trainer.createCustomDataloader(adv_dataset, batch_size=32, shuffle=False)
    # orig_dataloader = Trainer.createCustomDataloader(orig_dataset, batch_size=32, shuffle=False)
    # #
    # logger.info(f"Starting Test on default model for Original Dataset")
    # Analyzer.test(Model, test_loader=orig_dataloader, loss_func=Trainer.getLossFunction())
    # logger.info(f"Starting Test on default model for Adversarial Dataset")
    # Analyzer.test(Model, test_loader=adv_dataloader, loss_func=Trainer.getLossFunction())
    # # #Analyzer.density_evaluation()
    # #
    # Model.load_state_dict(torch.load("experiment/LeNet/v8_post_cuda/placeholder/LeNet_admm_retrain.pth"))
    # Model.eval()
    # logger.info(f"Starting Test on retrained model for Original Dataset")
    # Analyzer.test(Model, test_loader=orig_dataloader, loss_func=Trainer.getLossFunction())
    # logger.info(f"Starting Test on retrained model for Adversarial Dataset")
    # Analyzer.test(Model, test_loader=adv_dataloader, loss_func=Trainer.getLossFunction())
    # #Analyzer.density_evaluation()
    #
    # Model.load_state_dict(torch.load("LeNet_all_patt_reduce_to_4_admm_retrain.pth"))
    # Model.eval()
    # Analyzer.test(Model, test_loader=orig_dataloader, loss_func=Trainer.getLossFunction())
    # Analyzer.test(Model, test_loader=adv_dataloader, loss_func=Trainer.getLossFunction())
    # #Analyzer.density_evaluation()



    # ------------- Note: end test of adv

    # ------------- Note: start of ADMM Optimization LeNet

    # Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    # Trainer.setDataLoaderSettings(Configurator.loadDataloaderConfig())
    # Trainer.setSnapshotSettings(Configurator.loadSnapshotConfig())
    # Trainer.setADMMArchitectureConfig(Configurator.loadConfigFromRegistry("admm_model_architecture"))
    # Trainer.setADMMonfig(Configurator.loadConfigFromRegistry("admm_settings"))
    # Trainer.setTensorBufferConfig(Configurator.loadConfigFromRegistry("tensor_buffer"))
    #
    # # Trainer.train(test=False, onnx_enabled=False, tensor_buffering=True)
    #
    #
    # Analyzer.setAnalyzerConfig(Configurator.loadConfigFromRegistry("analyzer"))
    # Analyzer.setTrainer(Trainer)
    # train_kwargs = {
    #     'test': False
    # }
    #
    # Analyzer.startTestrun(train_kwargs)


    logger.critical(f"Time for algo is: {time.time()-start_time}")

    # =============== Note: Run ResNet18 Tuning

    Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    Trainer.setDataLoaderSettings(Configurator.loadDataloaderConfig())
    Trainer.setSnapshotSettings(Configurator.loadSnapshotConfig())
    Trainer.setADMMArchitectureConfig(Configurator.loadConfigFromRegistry("admm_model_architecture"))
    Trainer.setADMMConfig(Configurator.loadConfigFromRegistry("admm_settings"))
    #Trainer.setTensorBufferConfig(Configurator.loadConfigFromRegistry("tensor_buffer"))
    #Trainer.train(test=False)

    Analyzer.setAnalyzerConfig(Configurator.loadConfigFromRegistry("analyzer"))
    Analyzer.setTrainer(Trainer)
    train_kwargs = {
        'test': True
    }
    # create_missing_folders("experiment\\adv_data\\ResNet18\\deepfool_only_adv_success\\adv_image_generation\\adv_images",
    #                        1000)
    # create_missing_folders("experiment\\adv_data\\ResNet18\\deepfool_only_adv_success\\adv_image_generation\\original_images",
    #                        1000)
    Analyzer.startTestrun(train_kwargs)


    # copy_directory("configs",
    #                "experiment/LeNet/v7 elog vs all/clipGradientsTest/clip_grad_1/all_pat/configs")

    # ---------------- Note: this is just for visualization
    # test_loader = Trainer.getTestLoader()
    # loss_func = Trainer.getLossFunction()
    # #Analyzer = analyzer.Analyzer(Model, DataHandler)
    #
    # Analyzer.setDataset(DataHandler.loadDataset(testset=True))
    #
    # Model.load_state_dict(torch.load("LeNet_admm_train.pth"))
    # Model.eval()
    # Analyzer.grad_all(27)
    # Analyzer.add_model(Model, "Default Model")
    # # Analyzer.setModel(Model)
    # #Analyzer.run_single_model_test(101, test_end_index=None, test_loader=test_loader, loss_func=loss_func)
    # #Analyzer.grad_all(0)
    #
    # Model.load_state_dict(torch.load("LeNet_elog_consparse_admm_retrain.pth"))
    # Model.eval()
    # Analyzer.grad_all(27)
    # Analyzer.add_model(Model, "elog")
    #
    #
    # Model.load_state_dict(torch.load("LeNet_all_patt_reduce_to_4_admm_retrain.pth"))
    # Model.eval()
    # Analyzer.grad_all(27)
    # Analyzer.add_model(Model, "all pat")
    #
    #
    # #Analyzer.compare_models(10, 11)
    # Analyzer.runCompareTest(25, test_end_index=29, target_layer='model.conv1',
    #                         loss_func=loss_func, test_loader=test_loader)
    # -------------------------------------


    # TODO: überlegen wie man schön und geordnet Models hochladen kann und sie testen kann
    # TODO: Ordner wird cobenötigt oder irgendwas damit man strukturiert die Models speichert


if __name__ == '__main__':
    main()


#%%
