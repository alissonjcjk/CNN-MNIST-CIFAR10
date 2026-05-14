import os
import subprocess
import sys

ROOT = os.getcwd()

def run(cmd):
    print(f'>>> {" ".join(cmd)}')
    subprocess.run(cmd, check=True, cwd=ROOT)

def convert_and_run(nb_path, args):
    script_path = nb_path.replace('.ipynb', '.py')
    print(f'Convertendo {nb_path} para {script_path}...')
    subprocess.run(['jupyter', 'nbconvert', '--to', 'script', nb_path], check=True)
    run(['python', script_path] + args)

print('Dispositivo:', 'GPU detectada' if subprocess.run(['nvidia-smi'], capture_output=True).returncode == 0 else 'CPU')

# Caminho para o notebook de treino
train_nb = 'src/train.ipynb'

if os.path.exists(train_nb):
    convert_and_run(train_nb, ['--dataset', 'mnist', '--epochs', '20'])
else:
    print(f'Erro: {train_nb} não encontrado!')