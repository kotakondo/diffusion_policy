#!/usr/bin/env python3

import subprocess
import argparse
import os

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--machine", help="machine to run on", type=str, default="kota2")
args = parser.parse_args()

# current directory
cwd = os.getcwd()

# run python script
machine = args.machine
encoders = ["mlp", "lstm", "transformer", "gnn"]
decoders = ["mlp", "diffusion"]

for encoder in encoders:
    for decoder in decoders:
        cmd = f"python test_gnn_diffusion_training.py -en {encoder} -de {decoder} -m {machine}"
        print(cmd.split())
        subprocess.call(cmd.split(), cwd=cwd)


# done: mlp-mlp, mlp-diffusion, lstm-mlp, lstm-diffusion, 