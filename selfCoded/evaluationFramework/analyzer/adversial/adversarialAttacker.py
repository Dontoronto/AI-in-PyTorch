import math
import random

import numpy as np
import torch
from PIL import Image
import os
import shutil
import torchvision.transforms as T

from .robustml_utils.attack import Attack
from .robustml_utils.evaluate import evaluate
from .adversialModelWrapper import AdversarialModelWrapper
from .adversialAttackFactory import AdversarialAttackerFactory
from .threadModelFactory import ThreatModelFactory
from .providerFactory import ProviderFactory
from .utils import transformators


# TODO: schauen wie ich die Backwards Transformation Methode von Datahandler nachbauen kann als mock
# TODO: datahandler muss später um die funktion erweitert werden damit das automatisch generiert wird
# TODO: datahandler braucht get Methoden für die Transform function- und Backwards Transform funktionen
# TODO: dataset pfade müssen übergeben werden. schauen ob evtl. Parser zum einsatz kommt oder es anders
#       geladen werden muss
# TODO: auf numpy konvertierungen achten
# TODO: testrun machen ohne save funktion, es muss aber backwards transformieren können
# TODO: save funktion implementieren und schauen wie das läuft
# TODO: dynamisch tests einstellen können. Damit ich beliebige Anzahl von tests laufen lasen kann
# TODO: methode die für gewählten Index einmal das bild anzeigt welches daraus entstanden ist
# TODO: abgeänderte Skripte behalten ohne das pip package zu verwenden.


class AdversarialAttacker(Attack):
    def __init__(self, model, transform_batched_function,
                 backwards_transform_function,
                 save_adversarial_images=False,
                 **kwargs):
        '''

        :param model:
        :param transform_function:
        :param backwards_transform_function: muss in numpy ausgeben können
        :param save_adversarial_images:
        :param kwargs:
        '''

        self._model = model

        try:
            self.device = next(self._model.parameters()).device
            if self.device.type == 'cuda':
                torch.set_default_device('cuda')
                self._model.to('cuda')
                print(f"Device= {self.device}")
        except Exception:
            print("Failed to set device automatically, please try set_device() manually.")


        self.transform = transform_batched_function
        self.no_change_transformer = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float)
        ])

        # TODO: Wird evtl. nicht immer gebraucht und soll extern gesetzt werden
        self.backwards_transform_function = backwards_transform_function

        self.save_adversarial_images_flag = save_adversarial_images
        self.save_adversarial_arrays = list()
        self.save_labels = list()
        self.save_adversarial_path = None
        self.save_original_images_flag = False
        self.save_original_arrays = list()
        self.save_original_path = None
        self.save_threshold_flag = False

        self.adversarialModel = AdversarialModelWrapper(self._model, self.transform)

        self.dataset_provider = None

        self.threat_model = None
        self.indices_list = None

        self.attack_type_config = None
        self.attack_instance = None
        self.attack_instances_list = list()
        self.attack_instance_list_names = list()
        self.adv_only_success_flag = False
        self.adv_shuffle = True

    def createAdversarialEvaluationModel(self):
        self.adversarialModel = AdversarialModelWrapper(self._model, self.transform)


    def setModel(self, model):
        self._model = model

    def setProvider(self, provider_config):
        dataset_type = provider_config["provider"]
        params = provider_config["init_params"]
        self.dataset_provider = ProviderFactory.create_provider(dataset_type, **params)
        if dataset_type == "ImageNet":
            self.adversarialModel.set_transformer(transformators.adv_imagenet_transformer())
            self.transform = transformators.adv_imagenet_transformer()

    def getDatasetProvider(self):
        return self.dataset_provider

    def setThreatModel(self, threat_model_config):
        model_type = threat_model_config["thread_model"]
        params = threat_model_config["init_params"]
        self.threat_model = ThreatModelFactory.create_threat_model(model_type, **params)

    def setAdvShuffle(self, adv_shuffle):
        self.adv_shuffle = adv_shuffle

    def getThreatModel(self):
        return self.threat_model

    def setAttackTypeConfig(self, attack_type_config):
        self.attack_type_config = attack_type_config

    def enableAdversarialSaveMode(self, flag):
        self.save_adversarial_images_flag = flag

    def enableOriginalSaveMode(self, flag):
        self.save_original_images_flag = flag

    def setAdversarialSavePath(self, path):
        self.save_adversarial_path = path

    def setOriginalSavePath(self, path):
        self.save_original_path = path

    def enable_threshold_saving(self):
        self.save_threshold_flag = True

    def disable_threshold_saving(self):
        self.save_threshold_flag = False

    def set_adv_only_success_flag(self, success_falg = False):
        self.adv_only_success_flag = success_falg

    def selectAttacks(self, start_index: int = None, amount_of_attacks: int = None):

        configuration = self.attack_type_config

        if start_index is None:
            start_index = 0
            end = len(configuration)
        else:
            if amount_of_attacks is None:
                end = start_index+1
            else:
                if len(configuration)-start_index < amount_of_attacks:
                    print(f"Range exceeds max amount of elements. Amount of Attacks from start="
                          f"{len(configuration)-start_index}")
                    return
                end = start_index + amount_of_attacks

        for i in range(start_index, end):
            self.attack_instances_list.append(AdversarialAttackerFactory.create_attacker(
                self._model, configuration[i]["class"],
                **configuration[i]["init_params"]))
            self.attack_instance_list_names.append(configuration[i]["class"])
        return self.attack_instances_list

    def getSingleAttack(self, model, attack_name, **kwargs):
        atk = AdversarialAttackerFactory.create_attacker(model, attack_name, **kwargs)
        #atk = self.selectAttacks(start_index=attack_index)
        return atk


    def evaluate(self, start, end):
        dataset_size = len(self.dataset_provider)
        if start is not None and not (0 <= start < len(self.dataset_provider)):
            raise ValueError('start value out of range')
        if end is not None and not (0 <= end <= len(self.dataset_provider)):
            raise ValueError('end value out of range')

        result_ratio = dict()
        result_total = dict()
        result_success = dict()
        result_above_threshold = dict()
        result_no_perturbation = dict()

        if self.save_original_images_flag or self.save_adversarial_images_flag:
            if len(self.attack_instances_list) > 1:
                print(f"Image saving Mode is not possible while using multiple attacks.")
                print(f"Please configure AttacksConfig.json or Arguments of selectAttacks-method")
                return
            if self.save_original_images_flag is True:
                delete_folders_with_only_png(self.save_original_path)
            if self.save_adversarial_images_flag is True:
                delete_folders_with_only_png(self.save_adversarial_path)

        if self.adv_shuffle is True and self.indices_list is None:
            self.indices_list = generate_indices(start, end, dataset_size=dataset_size, shuffle=self.adv_shuffle)
        elif self.indices_list is None:
            self.indices_list = generate_indices(start, end, shuffle=False)

        for i in range(len(self.attack_instances_list)):

            self.attack_instance = self.attack_instances_list[i]
            print("========================================")
            print("========================================")
            print(self.attack_instance)
            print("========================================")
            print("========================================")
            chunk_size = 100
            chunk_start = start
            res_total = 0
            res_success = 0
            res_above_thresh = 0
            res_no_perturb = 0
            chunks = math.ceil((end - start)/chunk_size)

            while chunk_start < end:
                chunk_end = min(chunk_start + chunk_size, end)
                success, total, above_thresh, no_perturb = evaluate(
                    self.adversarialModel,
                    self,
                    self.getDatasetProvider(),
                    chunk_start,
                    chunk_end,
                    deterministic=False,
                    debug=True,
                    only_success=self.adv_only_success_flag,
                    index_list=self.indices_list[chunk_start:chunk_end]
                )
                res_total += total
                res_success += success
                res_above_thresh += above_thresh
                res_no_perturb += no_perturb
                chunk_start = chunk_end

                if self.save_adversarial_images_flag is True:
                    save_dataset(self.save_adversarial_arrays, self.save_labels, self.save_adversarial_path)
                if self.save_original_images_flag is True:
                    save_dataset(self.save_original_arrays, self.save_labels, self.save_original_path)
                self.clear_image_buffers()

            print(f"Adversarial succeeded in l-norm with rate: {res_success/res_total}")
            print(f"Adversarial succeeded in l-norm with: {res_success} adversarial samples")
            print(f"Adversarial processed total of: {res_total} samples")
            result_ratio[self.attack_instance_list_names[i]] = res_success/res_total
            result_success[self.attack_instance_list_names[i]] = res_success
            result_total[self.attack_instance_list_names[i]] = res_total
            result_above_threshold[self.attack_instance_list_names[i]] = res_above_thresh
            result_no_perturbation[self.attack_instance_list_names[i]] = res_no_perturb

        # if self.save_adversarial_images_flag is True:
        #     save_dataset(self.save_adversarial_arrays, self.save_labels, self.save_adversarial_path)
        # if self.save_original_images_flag is True:
        #     save_dataset(self.save_original_arrays, self.save_labels, self.save_original_path)

        return result_ratio, result_success, result_total, result_above_threshold, result_no_perturbation



    def run(self, x, y, target=None):


        # #x with shape (32, 32, 3) to tensor with shape (1,3,32,32)
        # x_in = self.transform(x)

        x_in = self.no_change_transformer(x).unsqueeze(0)


        #y as int to tensor with shape (1),  1 -> tensor([1])
        # with dtype=torch.uint8 war die itemsize 1 und DeepFool hat nicht funktioniert weil fs komisch geschnitten
        # wurde, mit torch.int64 funktioniert es
        y_label = torch.tensor([y], dtype=torch.int64)

        # y auch noch von numpy in torch umwandeln
        #attack = torchattacks.PGD(self._model, eps=self._epsilon, alpha=self._alpha, steps=self._max_steps, random_start=True)
        #attack = torchattacks.DeepFool(self._model, steps=60, overshoot=1)
        attack = self.attack_instance
        if self.adv_only_success_flag is True:
            x_test = self.transform(x)
            if len(x_test.shape) < 4:
                x_test = x_test.unsqueeze(0)
            valid_adv_input = self._model(x_test.to(self.device)).squeeze(0).argmax().item()
            if valid_adv_input != y:
                return x

        adv_images = attack(x_in, y_label)

        # if self.adv_only_success_flag is True:
        #     valid_adv_input = self._model(adv_images).squeeze(0).argmax().item()
        #     if valid_adv_input == y:
        #         return x


        #adv_numpy_img = self.backwards_transform_function(adv_images, numpy_original_shape_flag=True)
        adv_numpy_img = np.transpose(adv_images.squeeze(0).cpu().numpy(),(1,2,0))

        if self.save_adversarial_images_flag or self.save_original_images_flag:
            self.save_labels.append(y)
            if self.save_adversarial_images_flag:
                self.save_adversarial_arrays.append(adv_numpy_img)
            if self.save_original_images_flag:
                self.save_original_arrays.append(x)

        # plotten_real = np.copy(x)
        # plot_real_img = Image.fromarray((plotten_real[:,:,0] * 255).astype('uint8'))
        # plot_real_img.show("real")
        #
        # plot_img = Image.fromarray((np.copy(adv_numpy_img[:,:,0]) * 255).astype('uint8'))
        # plot_img.show("adversarial")


        return adv_numpy_img

    def remove_adv_image_over_threshold(self):
        if self.save_threshold_flag is True:
            if self.save_adversarial_images_flag or self.save_original_images_flag:
                self.save_labels.pop()
                if self.save_adversarial_images_flag:
                    self.save_adversarial_arrays.pop()
                if self.save_original_images_flag:
                    self.save_original_arrays.pop()
        else:
            pass

    def clear_image_buffers(self):
        self.save_adversarial_arrays.clear()
        self.save_original_arrays.clear()
        self.save_labels.clear()


def array_to_image(array, save_path):
    """
    Konvertiert ein NumPy-Array in ein Bild und speichert es.

    :param array: NumPy-Array, das die Bilddaten enthält.
    :param save_path: Pfad, unter dem das Bild gespeichert werden soll.
    """
    # Konvertieren des NumPy-Arrays zu einem PIL-Bild
    # Überprüfen, ob das Array für ein Graustufenbild ist (2D-Array).
    if array.ndim == 2:
        # Direkte Konvertierung in ein PIL-Bild und Speichern.
        image = Image.fromarray(array.astype('uint8'), 'L')
    elif array.ndim == 3:
        # Überprüfen, ob das Array einen einzigen Farbkanal hat (z.B. bei einigen Graustufenbildern).
        if array.shape[2] == 1:
            # Behandeln wie ein 2D-Graustufenbild.
            image = Image.fromarray((array[:,:,0] * 255).astype('uint8'), 'L')
        else:
            # Behandeln als Farbbild.
            image = Image.fromarray((array*255).astype('uint8'), 'RGB')
    else:
        raise ValueError("Array hat eine ungültige Form für ein Bild.")
    # Speichern des Bildes
    image.save(save_path)


def save_dataset(arrays, labels, root_dir):
    """
    Speichert eine Serie von NumPy-Arrays als Bilder in einer Ordnerstruktur.

    :param arrays: Liste von NumPy-Arrays, die die Bilddaten enthalten.
    :param labels: Liste von Labels, die den Arrays entsprechen.
    :param root_dir: Wurzelverzeichnis, unter dem die Bilder gespeichert werden sollen.
    """
    # delete_folders_with_only_png(root_dir)
    for i, (array, label) in enumerate(zip(arrays, labels)):
        # Erstellen des Ordners für das Label, wenn er nicht existiert
        label_dir = os.path.join(root_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        existing_files = os.listdir(label_dir)
        file_count = len(existing_files)

        # Pfad für das Bild
        save_path = os.path.join(label_dir, f"image_{file_count}.png")

        # Konvertieren und Speichern
        array_to_image(array, save_path)

def delete_folders_with_only_png(root_dir):
    """
    Durchläuft das angegebene Wurzelverzeichnis und löscht alle Unterverzeichnisse,
    die ausschließlich PNG-Dateien enthalten.

    :param root_dir: Das Wurzelverzeichnis, in dem die Unterverzeichnisse geprüft werden sollen.
    """
    for subdir_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir_name)

        # Stellen Sie sicher, dass es ein Verzeichnis ist
        if os.path.isdir(subdir_path):
            all_files = os.listdir(subdir_path)

            # Prüfen, ob alle Dateien im Verzeichnis PNG-Dateien sind
            if all_files and all(file.endswith(".png") for file in all_files):
                # Löschen des Verzeichnisses und seines Inhalts
                shutil.rmtree(subdir_path)
                print(f"{subdir_path} wurde gelöscht, da es ausschließlich PNG-Dateien enthielt.")
            else:
                print(f"{subdir_path} wurde nicht gelöscht, da es auch Daten enthält die nicht vom Typ PNG sind.")


def has_transform(transform, transform_type):
    for t in transform.transforms:
        if isinstance(t, transform_type):
            return True
    return False

def generate_indices(start, end, dataset_size=None, shuffle=False):
    # Generate the list of indices
    indices = list(range(start, end))

    # Shuffle the list of indices if shuffle is True
    if shuffle:
        if dataset_size is not None:
            _ = list(range(0, dataset_size))
            random.shuffle(_)
            indices = _[start:end]


    # Return the list as an iterator
    return indices