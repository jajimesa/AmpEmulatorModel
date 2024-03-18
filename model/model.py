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

class CausalConv1d(torch.nn.Conv1d):
    """
    Clase que implementa una capa de convolución causal.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, dilation=1, groups=1, bias=True):
        """
        Constructor de una convolución causal 1 x 1.

        Ars:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
            kernel_size (int): Tamaño del kernel de la convolución. Por defecto 2.
            stride (int): Tamaño del paso de la convolución.
            dilation (int): Tasa de dilatación de la convolución. Por defecto 1, pero permite implementar convoluciones dilatadas.
            groups (int): Número de grupos en los que se dividen las entradas y salidas. Por defecto 1.
            bias (bool): Indica si se incluye un término de sesgo en la convolución. Por defecto True.
        """
        self.__causal_padding = (kernel_size - 1) * dilation   # Padding que garantiza que la convolución sea causal.

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=self.__causal_padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

        def forward(self, input):
            """
            Método que implementa el paso hacia adelante de la capa de convolución.
            La salida es causal: solo depende de las entradas en pasos de tiempo anteriores o iguales al instante actual.
            """
            output = super(CausalConv1d, self).forward(input)

            if self.__dilated_padding != 0:
                output = output[:, :, :-self.__dilated_padding]

            return output
            
            
        
    
def _pila_convoluciones(in_channels, out_channels, kernel_size, dilation, **kwargs):
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


