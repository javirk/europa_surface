import wandb
import subprocess

# Gather nodes allocated to current slurm job
result = subprocess.run(['scontrol', 'show', 'hostnames'], stdout=subprocess.PIPE)
node_list = result.stdout.decode('utf-8').split('\n')[:-1]

def run():
    wandb.init(project="SAM_EUROPA")

    sp = []
    for node in node_list:
        sp.append(subprocess.Popen(['srun',
                                    '--nodes=1',
                                    '--ntasks=1',
                                    '-w',
                                    node,
                                    'scripts/wandb_agent.sh']))
    exit_codes = [p.wait() for p in sp]  # wait for processes to finish
    return exit_codes

if __name__ == '__main__':
    run()
