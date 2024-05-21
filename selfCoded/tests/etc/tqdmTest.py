from time import time
from tqdm import tqdm

LOOP = 50000

def task(update_cycle):
    pbar = tqdm(total=LOOP)
    for i, _ in enumerate(range(LOOP)):
        for _ in range(LOOP):
            pass

        if i % update_cycle == 0:
            pbar.update(update_cycle)

    pbar.close()

tic = time()
print(f'Update cycle: {1}')
task(1)
toc = time()
print(f'Time elapsed: {toc-tic:0.2f} seconds\n')

tic = time()
print(f'Update cycle: {500}')
task(500)
toc = time()
print(f'Time elapsed: {toc-tic:0.2f} seconds')