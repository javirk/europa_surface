import os
import numpy as np
from itertools import product


def main(folder, base_file):
    # Delete all the files in folder except base_file
    for file in os.listdir(folder):
        if 'base' not in file:
            os.remove(os.path.join(folder, file))
    with open(os.path.join(folder, base_file), 'r') as f:
        base_text = f.read()

    config = {
        'lrs': np.linspace(0.001, 5e-5, 5),
        'loss_weights': ["1.,20.", "1.,100.", "1.,1.", "1.,2."],
        'loss_fn': ['DiceLoss,FocalLoss', 'DiceLoss,CrossEntropyLoss'],
        'batch_size': [8, 16, 24, 32, 64, 128]
    }

    # combinations = product(config['lrs'], config['loss_weights'], config['loss_fn'], config['batch_size'])
    combinations = product(*config.values())
    i = 0
    for combo in combinations:
        text = base_text
        for key, val in zip(config.keys(), combo):
            text = text.replace(f'${key}$', f'{val}')
        with open(os.path.join(folder, f'point_{i}.sh'), 'w') as f:
            f.write(text)
        i += 1

    # Replace the max number of jobs in the arr file
    with open(os.path.join(folder, 'arr_base.sh'), 'r') as f:
        arr_text = f.read()
    arr_text = arr_text.replace('$array_num$', str(i))
    with open(os.path.join(folder, 'arr.sh'), 'w') as f:
        f.write(arr_text)


if __name__ == '__main__':
    base_folder = 'scripts/hyperparam_search/'
    base_file = 'base.sh'
    main(base_folder, base_file)
