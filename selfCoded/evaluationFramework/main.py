# main.py
import argparse
import warnings
import time

warnings.filterwarnings("ignore", message="Failed to load image Python extension")
import sys, os
sys.path.append(os.getcwd())
import torch
import matplotlib

import modelWrapper
import dataHandler
import configurator
import analyzer.analyzer as analyzer

from Trainer.trainerFactory import TrainerFactory
from SharedServices.logging_config import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)

# NOTE: currently we are testing with LeNet-model
from models.lenet import LeNet


def main():

    #matplotlib.use('Agg')  # Use a non-interactive backend

    # ====================== Note: This code is just for exporting onnx model for tvm
    # _model = LeNet()
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.load_state_dict(torch.load("LeNet_pat_con_fc_admm_retrain.pth"))
    # input_names = ["input"]
    # output_names = ["output"]
    # torch.onnx.export(Model.model,torch.randn(1,1,28,28),'model_opset.onnx', opset_version=11,
    #                   input_names=input_names, output_names=output_names)

    # ================================


    # ============== Note: Other Model testrun

    # Resnet settings work
    # _weights = ResNet101_Weights.IMAGENET1K_V1
    # _model = resnet101(weights=_weights)
    # Model = modelWrapper.ModelWrapper(_model)
    # Model.eval()

    # =======================================


    start_time = time.time()
    warnings.filterwarnings(action='once')

    # LeNet Test
    _model = LeNet()
    Model = modelWrapper.ModelWrapper(_model)
    Model.load_state_dict(torch.load("LeNet_admm_train.pth"))
    # Model.load_state_dict(torch.load("/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/selfCoded/evaluationFramework/experiment/LeNet/v7 elog vs all/clipGradientsTest/clip_grad_1/elog_800admmiter_no_connectivity/LeNet_all_pat_clipGrad_1_0.pth"))
    Model.eval()


    Configurator = configurator.Configurator()
    DataHandler = dataHandler.DataHandler(Configurator)

    #DataHandler.setTransformer(_weights.transforms())
    DataHandler.setTransformer(Configurator.loadTransformer())

    # img = DataHandler.loadImage("testImages/tisch_v2.jpeg")




    Analyzer = analyzer.Analyzer(Model, DataHandler)

    # ================== Note: this part is for generating adversarial examples

    # Analyzer.init_adversarial_environment()
    # Analyzer.set_threat_model_config(Configurator.loadConfigFromRegistry("adversarial_threat_model"))
    # Analyzer.set_provider_config(Configurator.loadConfigFromRegistry("adversarial_provider"))
    # Analyzer.set_attack_type_config(Configurator.loadConfigFromRegistry("adversarial_attacks"))
    # Analyzer.select_attacks_from_config(0, 1)
    # Analyzer.enable_adversarial_saving("experiment/deepfool/adversarial_images")
    # Analyzer.enable_original_saving("experiment/deepfool/original_images")
    #
    # test1 = Analyzer.start_adversarial_evaluation(0, 100)
    # print(f"First evaluation:")
    # print(test1)

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
    # adv_dataset = DataHandler.create_imageFolder_dataset("experiment/JSMA/adversarial_images")
    # orig_dataset = DataHandler.create_imageFolder_dataset("experiment/JSMA/original_images")
    # Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    # adv_dataloader = Trainer.createCustomDataloader(adv_dataset, batch_size=32, shuffle=False)
    # orig_dataloader = Trainer.createCustomDataloader(orig_dataset, batch_size=32, shuffle=False)
    # #
    # Analyzer.test(Model, test_loader=orig_dataloader, loss_func=Trainer.getLossFunction())
    # Analyzer.test(Model, test_loader=adv_dataloader, loss_func=Trainer.getLossFunction())
    # #Analyzer.density_evaluation()
    #
    # Model.load_state_dict(torch.load("/Users/dominik/Documents/jupyter/Neuronale Netze programmieren Buch/AI in PyTorch/selfCoded/evaluationFramework/experiment/LeNet/v7 elog vs all/clipGradientsTest/clip_grad_1/elog_800admmiter_no_connectivity/LeNet_all_pat_clipGrad_1_0_admm_retrain.pth"))
    # Model.eval()
    # Analyzer.test(Model, test_loader=orig_dataloader, loss_func=Trainer.getLossFunction())
    # Analyzer.test(Model, test_loader=adv_dataloader, loss_func=Trainer.getLossFunction())
    # #Analyzer.density_evaluation()
    #
    # Model.load_state_dict(torch.load("LeNet_all_patt_reduce_to_4_admm_retrain.pth"))
    # Model.eval()
    # Analyzer.test(Model, test_loader=orig_dataloader, loss_func=Trainer.getLossFunction())
    # Analyzer.test(Model, test_loader=adv_dataloader, loss_func=Trainer.getLossFunction())
    # #Analyzer.density_evaluation()



    # ------------- Note: end test of adv

    # ------------- Note: start of ADMM Optimization

    Trainer = TrainerFactory.createTrainer(Model, DataHandler, Configurator.loadTrainingConfig())
    Trainer.setDataLoaderSettings(Configurator.loadDataloaderConfig())
    Trainer.setSnapshotSettings(Configurator.loadSnapshotConfig())
    Trainer.setADMMArchitectureConfig(Configurator.loadConfigFromRegistry("admm_model_architecture"))
    Trainer.setADMMConfig(Configurator.loadConfigFromRegistry("admm_settings"))
    Trainer.setTensorBufferConfig(Configurator.loadConfigFromRegistry("tensor_buffer"))

    #Trainer.train(test=False, onnx_enabled=False, tensor_buffering=True)


    Analyzer.setAnalyzerConfig(Configurator.loadConfigFromRegistry("analyzer"))
    Analyzer.setTrainer(Trainer)

    # test_loader = Analyzer.trainer.getTestLoader()
    # loss_func = Trainer.getLossFunction()


    train_kwargs = {
        'test': False
    }

    Analyzer.startTestrun(train_kwargs)

    # test_loader = Analyzer.trainer.getTestLoader()
    # loss_func = Trainer.getLossFunction()

    # histW = Trainer.getHistoryEpsilonW()
    # histZ = Trainer.getHistoryEpsilonZ()
    # thrshW = Trainer.epsilon_W
    # thrshZ = Trainer.epsilon_Z

    #histW, histZ, thrshW, thrshZ = Trainer.getEpsilonResults()

    #Analyzer.setSavePath("experiment/LeNet/v7 elog vs all/clipGradientsTest/clip_grad_1/all_pat")

    #Analyzer.eval_epsilon_distances(histW, histZ, thrshW, thrshZ)

    logger.critical(f"Time for algo is: {time.time()-start_time}")


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
