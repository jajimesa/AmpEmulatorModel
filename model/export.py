"""
=============================================================================
Modificación del fichero export.py originario de PedalNet, para poder testear
mi modelo con el plugin de WaveNetVA.

Autor original: GuitarML
Fuente: https://github.com/GuitarML/PedalNetRT
=============================================================================
"""

import argparse
import numpy as np
import json
#import torch
#import os

#from model import PedalNet
from model import AmpEmulatorModel

def convert(args):
    """
    Converts a *.ckpt model from PedalNet into a .json format used in WaveNetVA.

    Current changes to the original PedalNet model to match WaveNetVA include:
      1. Added CausalConv1d() to use causal padding
      2. Added an input layer, which is a Conv1d(in_channls=1, out_channels=num_channels, kernel_size=1)
      3. Instead of two conv_stacks for tanh and sigm, used a single hidden layer with input_channels=16,
         output_channels=32, then split the matrix for tanh and sigm calculation.

      Note: The original PedalNet model was intended for use on PCM Int16 format wave files. The WaveNetVA is
            intended as a plugin, which processes float32 audio data. The PedalNet model must be trained on wave files
            saved as Float32 data, which has sample data in the range -1 to 1.

      Note: The WaveNetVA plugin doesn't perform the standardization step as in predict.py. With the standardization step
            omitted, the signals match between the plugin with converted model, and the predict.py output.

      The model parameters used for conversion testing match the Wavenetva1 model (limited testing using other parameters):
            --num_channels=16, --dilation_depth=10, --num_repeat=1, --kernel_size=3
    """

    # Permute tensors to match Tensorflow format with .permute(a,b,c):
    a, b, c = (
        2,
        1,
        0,
    )  # Pytorch uses (out_channels, in_channels, kernel_size), TensorFlow uses (kernel_size, in_channels, out_channels)
    #model = PedalNet.load_from_checkpoint(checkpoint_path=args.model)
    model = AmpEmulatorModel.load_from_checkpoint(checkpoint_path=args.model)
    #model = AmpEmulatorModel.load_from_checkpoint("results/model.ckpt", num_channels=4, dilation_depth=10, dilation_repeat=1, kernel_size=3, lr=3e-3)

    sd = model.state_dict()

    # Get hparams from model
    hparams = model.hparams
    #residual_channels = hparams.num_channels
    residual_channels = 4
    #filter_width = hparams.kernel_size
    filter_width = 3
    #dilations = [2 ** d for d in range(hparams.dilation_depth)] * hparams.num_repeat
    dilation_depth = 9
    dilation_repeat = 2
    dilations = [2 ** d for d in range(dilation_depth)] * dilation_repeat

    data_out = {
        "activation": "gated",
        "output_channels": 1,
        "input_channels": 1,
        "residual_channels": residual_channels,
        "filter_width": filter_width,
        "dilations": dilations,
        "variables": [],
    }

    # Use pytorch model data to populate the json data for each layer
    for i in range(-1, len(dilations) + 1):
        # Input Layer
        if i == -1:
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(w) for w in (sd["wavenet.input_layer.weight"]).permute(a, b, c).flatten().numpy().tolist()
                    ],
                    "name": "W",
                }
            )
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [str(b) for b in (sd["wavenet.input_layer.bias"]).flatten().numpy().tolist()],
                    "name": "b",
                }
            )
        # Linear Mix Layer
        elif i == len(dilations):
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        #str(w) for w in (sd["wavenet.linear_mix.weight"]).permute(a, b, c).flatten().numpy().tolist()
                        str(w) for w in (sd["wavenet.linear_mixer.weight"]).permute(a, b, c).flatten().numpy().tolist()
                    ],
                    "name": "W",
                }
            )

            data_out["variables"].append(
                {
                    "layer_idx": i,
                    #"data": [str(b) for b in (sd["wavenet.linear_mix.bias"]).numpy().tolist()],
                    "data": [str(b) for b in (sd["wavenet.linear_mixer.bias"]).numpy().tolist()],
                    "name": "b",
                }
            )
        # Hidden Layers
        else:
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(w)
                        #for w in sd["wavenet.hidden." + str(i) + ".weight"].permute(a, b, c).flatten().numpy().tolist()
                        for w in sd["wavenet.hidden_stack." + str(i) + ".weight"].permute(a, b, c).flatten().numpy().tolist()

                    ],
                    "name": "W_conv",
                }
            )
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    #"data": [str(b) for b in sd["wavenet.hidden." + str(i) + ".bias"].flatten().numpy().tolist()],
                    "data": [str(b) for b in sd["wavenet.hidden_stack." + str(i) + ".bias"].flatten().numpy().tolist()],                    
                    "name": "b_conv",
                }
            )
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    "data": [
                        str(w2)
                        #for w2 in sd["wavenet.residuals." + str(i) + ".weight"]
                        for w2 in sd["wavenet.residual_stack." + str(i) + ".weight"]
                        .permute(a, b, c)
                        .flatten()
                        .numpy()
                        .tolist()
                    ],
                    "name": "W_out",
                }
            )
            data_out["variables"].append(
                {
                    "layer_idx": i,
                    #"data": [str(b2) for b2 in sd["wavenet.residuals." + str(i) + ".bias"].flatten().numpy().tolist()],
                    "data": [str(b2) for b2 in sd["wavenet.residual_stack." + str(i) + ".bias"].flatten().numpy().tolist()],
                    "name": "b_out",
                }
            )

    # output final dictionary to json file
    with open(args.model.replace(".ckpt", ".json"), "w") as outfile:
        json.dump(data_out, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="results/model.ckpt")
    parser.add_argument("--format", default="json")
    args = parser.parse_args()
    convert(args)