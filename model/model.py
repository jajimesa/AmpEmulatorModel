"""
Implementación de la red neuronal propuesta en 

    "Real-Time Guitar Amplifier Emulation with Deep Learning" de Wright, et al., 2020
    https://www.mdpi.com/2076-3417/10/3/766

Red neuronal basada en la arquitectura de red neuronal propuesta en 
    "WaveNet: A Generative Model for Raw Audio" de van den Oord, et al., 2016
    https://arxiv.org/abs/1609.03499
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dCausal(torch.nn.Conv1d):
    """
    Clase que implementa una capa de convolución causal.
    """
    
def _pila_convolucional(in_channels, out_channels, kernel_size, dilation, **kwargs):
    """
    Función auxiliar que construye una pila de capas de convolución.
    """
    
class WaveNet(nn.Module):
    """
    Clase que implementa la red neuronal propuesta en "Real-Time Guitar Amplifier Emulation with Deep Learning" de Wright, et al., 2020.
    """
    
    def __init__(self, in_channels, out_channels, n_layers, n_residual_channels, n_skip_channels, n_dilation_cycles, kernel_size, **kwargs):
        """
        Constructor de la clase.
        """
        
    def forward(self, x):
        """
        Método que implementa el paso hacia adelante de la red neuronal.
        """
        
        return x


