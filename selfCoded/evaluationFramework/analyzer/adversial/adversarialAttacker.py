import math

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

        self.attack_type_config = None
        self.attack_instance = None
        self.attack_instances_list = list()
        self.attack_instance_list_names = list()
        self.adv_only_success_flag = False

    def createAdversarialEvaluationModel(self):
        self.adversarialModel = AdversarialModelWrapper(self._model, self.transform)


    def setModel(self, model):
        self._model = model

    def setProvider(self, provider_config):
        dataset_type = provider_config["provider"]
        params = provider_config["init_params"]
        self.dataset_provider = ProviderFactory.create_provider(dataset_type, **params)

    def getDatasetProvider(self):
        return self.dataset_provider

    def setThreatModel(self, threat_model_config):
        model_type = threat_model_config["thread_model"]
        params = threat_model_config["init_params"]
        self.threat_model = ThreatModelFactory.create_threat_model(model_type, **params)

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


    def evaluate(self, start, end):
        if start is not None and not (0 <= start < len(self.dataset_provider)):
            raise ValueError('start value out of range')
        if end is not None and not (0 <= end <= len(self.dataset_provider)):
            raise ValueError('end value out of range')

        results = dict()

        if self.save_original_images_flag or self.save_adversarial_images_flag:
            if len(self.attack_instances_list) > 1:
                print(f"Image saving Mode is not possible while using multiple attacks.")
                print(f"Please configure AttacksConfig.json or Arguments of selectAttacks-method")
                return
            if self.save_original_images_flag is True:
                delete_folders_with_only_png(self.save_original_path)
            if self.save_adversarial_images_flag is True:
                delete_folders_with_only_png(self.save_adversarial_path)



        for i in range(len(self.attack_instances_list)):
            self.attack_instance = self.attack_instances_list[i]
            chunk_size = 100
            chunk_start = start
            rate = 0
            chunks = math.ceil((end - start)/chunk_size)

            while chunk_start < end:
                chunk_end = min(chunk_start + chunk_size, end)
                rate += evaluate(
                    self.adversarialModel,
                    self,
                    self.getDatasetProvider(),
                    chunk_start,
                    chunk_end,
                    deterministic=False,
                    debug=True,
                    only_success=self.adv_only_success_flag
                )
                chunk_start = chunk_end

                if self.save_adversarial_images_flag is True:
                    save_dataset(self.save_adversarial_arrays, self.save_labels, self.save_adversarial_path)
                if self.save_original_images_flag is True:
                    save_dataset(self.save_original_arrays, self.save_labels, self.save_original_path)
                self.clear_image_buffers()

            print(f"Adversarial succeeded in l-norm with rate: {rate/chunks}")
            results[self.attack_instance_list_names[i]] = rate/chunks

        # if self.save_adversarial_images_flag is True:
        #     save_dataset(self.save_adversarial_arrays, self.save_labels, self.save_adversarial_path)
        # if self.save_original_images_flag is True:
        #     save_dataset(self.save_original_arrays, self.save_labels, self.save_original_path)

        return results



    def run(self, x, y, target=None):


        # #x with shape (32, 32, 3) to tensor with shape (1,3,32,32)
        # x_in = self.transform(x)
        transformator = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float)
        ])

        x_in = transformator(x).unsqueeze(0)


        #y as int to tensor with shape (1),  1 -> tensor([1])
        # with dtype=torch.uint8 war die itemsize 1 und DeepFool hat nicht funktioniert weil fs komisch geschnitten
        # wurde, mit torch.int64 funktioniert es
        y_label = torch.tensor([y], dtype=torch.int64)

        # y auch noch von numpy in torch umwandeln
        #attack = torchattacks.PGD(self._model, eps=self._epsilon, alpha=self._alpha, steps=self._max_steps, random_start=True)
        #attack = torchattacks.DeepFool(self._model, steps=60, overshoot=1)
        attack = self.attack_instance
        if self.adv_only_success_flag is True:
            valid_adv_input = self._model(x_in.to(self.device)).squeeze(0).argmax().item()
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