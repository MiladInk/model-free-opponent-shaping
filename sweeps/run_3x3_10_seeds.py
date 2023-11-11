import os
import random


def gen_script(run_command):
    script = "#!/bin/bash\n"
    script += "source ./load.sh\n"
    script += run_command
    return script


def run_job(command, fake_submit: bool = True, partition: str = 'long'):
    script = gen_script(command)
    print('script:\n', script)
    # write this to 'sweep_temp_job.sh'
    with open('sweep_temp_job.sh', 'w') as f:
        f.write(script)

    # submit this job using slurm
    if fake_submit:
        print('fake submit')
    else:
        command = f'sbatch --time=6:00:0 --partition={partition} --gres=gpu:a100l:1 --mem=20Gb -c 4 sweep_temp_job.sh'
        os.system(command)

def main(seeds, sizes, fake_submit: bool = True, partition='long'):
    command = "python src/coin_game/main_mfos_coin_game.py --exp-name milad_3x3_{seed} --grid-size {size}"
    for size in sizes:
        for seed in seeds:
            run_job(command=command.format(seed=seed, size=size), fake_submit=fake_submit, partition=partition)


if __name__ == '__main__':
    # use fire to turn this into a command line tool
    import fire
    fire.Fire()









