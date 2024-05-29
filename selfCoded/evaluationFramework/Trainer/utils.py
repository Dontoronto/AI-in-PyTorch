from tqdm import tqdm



def tqdm_progressbar(epoch, iter_per_epo, phase, admm_iter=-1):
    if phase == 'admm':
        total_iteration_tqdm = admm_iter
    elif phase == 'retrain':

        total_iteration_tqdm = epoch * iter_per_epo
        if total_iteration_tqdm > admm_iter:
            total_iteration_tqdm = admm_iter
        else:
            pass

    else:
        total_iteration_tqdm = epoch * iter_per_epo

    pbar = tqdm(total=total_iteration_tqdm, desc=f"{phase}-Iteration", position=0, leave=True)
    return pbar



