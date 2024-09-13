import numpy as np
import sys
import random
import torch

#based on https://github.com/robust-ml/robustml

def evaluate(model, attack, provider, start=None, end=None, deterministic=False,
             debug=False, only_success=False, index_list=None):
    '''
    Evaluate an attack on a particular model and return attack success rate.

    An attack is allowed to be adaptive, so it's fine to design the attack
    based on the specific model it's supposed to break.

    `start` (inclusive) and `end` (exclusive) are indices to evaluate on. If
    unspecified, evaluates on the entire dataset.

    `deterministic` specifies whether to seed the RNG with a constant value for
    a more deterministic test (so randomly selected target classes are chosen
    in a pseudorandom way).
    '''

    threat_model = attack.getThreatModel()
    targeted = threat_model.targeted

    success = 0
    total = 0
    above_threshold = 0
    no_perturbation_failure = 0
    topk_correct = [0, 0, 0, 0]  # top-1, top-2, top-3, top-5 correct counters
    ks = [1, 2, 3, 5]  # Define the k values
    if index_list is None:
        indices = generate_indices(start, end)
    else:
        indices = index_list

    for i in indices:
        print('evaluating %d of [%d, %d)' % (i, start, end))
        total += 1
        x, y = provider[i]
        target = None
        if targeted:
            target = choose_target(i, y, provider.labels, deterministic)
        x_adv = attack.run(np.copy(x), y, target)
        prediction = model.predict(np.copy(x_adv))
        for i, k in enumerate(ks):
            _, topk_indices = torch.topk(prediction, k)
            if y in topk_indices.tolist():
                topk_correct[i] += 1
        if not threat_model.check(np.copy(x), np.copy(x_adv)):
            if debug:
                print('check failed')
            above_threshold += 1
            #attack.remove_adv_image_over_threshold()
            continue
        else:
            attack.add_images_label_to_buffer(x, x_adv, y)
        y_adv = model.classify(np.copy(x_adv))
        if debug:
            print('true = %d, adv = %d' % (y, y_adv))
        if targeted:
            if y_adv == target:
                success += 1
        else:
            if only_success is True:
                if threat_model.adv_success(np.copy(x), np.copy(x_adv)) is False:
                    if y_adv != y:
                        no_perturbation_failure += 1
                    attack.remove_adv_image_over_threshold()
                    continue
            if y_adv != y:
                success += 1
            else:
                if only_success is True:
                    attack.remove_adv_image_over_threshold()

    success_rate = success / total


    return success, total, above_threshold, no_perturbation_failure, topk_correct


def choose_target(index, true_label, num_labels, deterministic=False):
    if deterministic:
        rng = np.random.RandomState(index)
    else:
        rng = np.random.RandomState()

    target = true_label
    while target == true_label:
        target = rng.randint(0, num_labels)

    return target


def generate_indices(start, end, shuffle=False):
    # Generate the list of indices
    indices = list(range(start, end))

    # Shuffle the list of indices if shuffle is True
    if shuffle:
        random.shuffle(indices)

    # Return the list as an iterator
    return indices