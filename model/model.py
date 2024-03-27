"""
Implementación en Pytorch Lightning de la red neuronal propuesta en 

    "Real-Time Guitar Amplifier Emulation with Deep Learning" de Wright, et al., 2020
    https://www.mdpi.com/2076-3417/10/3/766

Red neuronal basada en la arquitectura de red neuronal propuesta en 
    "WaveNet: A Generative Model for Raw Audio" de van den Oord, et al., 2016
    https://arxiv.org/abs/1609.03499
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class DilatedCausalConv1d(torch.nn.Conv1d):
    """
    Clase que implementa una capa de convolución causal dilatada.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, dilation=1, groups=1, bias=True):
        """
        Constructor de una convolución causal dilatada 1 x 1.

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

        super(DilatedCausalConv1d, self).__init__(
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
        output = super(DilatedCausalConv1d, self).forward(input)

        if self.__dilated_padding != 0:
            output = output[:, :, :-self.__dilated_padding]     # Garantiza que la salida sea causal.

        return output
    
class WaveNet(nn.Module):
    """
    Clase que implementa la red neuronal propuesta en "Real-Time Guitar Amplifier Emulation with Deep Learning" de Wright, et al., 2020.
    """
    
    def __init__(self, num_channels, dilation_depth, dilation_repeat, kernel_size=2):
        """
        Constructor de la clase.

        Args:
            num_channels (int): Número de canales de salida de la capa anterior al mixer lineal.
            dilation_depth (int): Máxima potencia de 2 que tendrá la tasa de dilatación.
            dilation_repeat (int): Número de veces que se repite el rango de potencias de 2.
            kernel_size (int): Tamaño del kernel de la convolución. Por defecto 2.
        """
        super(WaveNet, self).__init__()

        self.num_channels = num_channels
        self.dilation_depth = dilation_depth
        self.dilation_repeat = dilation_repeat
        self.kernel_size = kernel_size
        
        # La capa de entrada es una convolución causal 1 x 1 no dilatada.
        self.input_layer = DilatedCausalConv1d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=1
        )

        # El output de la capa de entrada se pasa al bloque residual
        dilations = self.__build_dilations(dilation_depth, dilation_repeat)
        self.hidden_stack = self.__convolution_stack(
            in_channels=num_channels,
            out_channels= 2*num_channels,  # Se duplica la salida para usar la gated activation unit
            kernel_size=kernel_size,
            dilations=dilations
        )
        self.residual_stack = self.__convolution_stack(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=1,
            dilations=dilations
        )

        # Las salidas "skip-connections" pasan al mixer lineal para generar la salida final
        self.linear_mixer = nn.Conv1d(
            in_channels=num_channels*dilation_depth*dilation_repeat,   # Número de salidas de las skip-connections
            out_channels=1,
            kernel_size=1
        )

    def __build_dilations(self, dilation_depth, dilations_repeat):
        """
        Método que construye la lista de tasas de dilatación.

        Ejemplos: 
            depth=3, repeat=2 -> [1, 2, 4, 1, 2, 4]
            depth=2, repeat=3 -> [1, 2, 1, 2, 1, 2]
            depth=9, repeat=2 -> [1, 2, ... , 128, 256, 1, 2, ... , 128, 256]

        Args:
            dilation_depth (int): Máxima potencia de 2 que tendrá la tasa de dilatación.
            dilations_repeat (int): Número de veces que se repite el rango de potencias de 2.
        """
        dilations = []
        for i in range(dilations_repeat):
            for j in range(dilation_depth):
                dilations.append(2**j)
        return dilations
    
    def __convolution_stack(in_channels, out_channels, kernel_size, dilations):
        """
        Función auxiliar que construye una pila de capas de convolución. Devuelve una pila
        de capas de convolución de tamaño igual a la longitud de la lista dilations.

        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
            kernel_size (int): Tamaño del kernel de la convolución.
            dilation_array (list): Lista de tasas de dilatación de las convoluciones.
        """
        pila = nn.ModuleList()
        for d in enumerate(dilations):
            pila.append(
                DilatedCausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                )
            )     
        return pila
    
    def receptive_field(self):
        """
        Método que calcula el campo receptivo de la red neuronal.
        """
        return self.dilation_repeat * (2 ** self.dilation_depth - 1) * (self.kernel_size - 1) + 1   

    def forward(self, x):
        """
        Método que implementa el paso hacia adelante de la red neuronal. Los datos de entrada "x"
        atraviesan la capa de entrada, el bloque residual (tantas veces como capas convolucionales)
        y el mixer lineal para producir el output.
        """

        out = self.input_layer(x)
        skip_connections = []

        # Bloque residual
        for hidden, residual in zip(self.hidden_stack, self.residual_stack):
            # in (num_channels) --> hidden --> out (2*num_channels)
            x = out
            out_hidden = torch.split(hidden(x), self.num_channels, dim=1)

            # in (2*num_channels) --> gated_activation_unit --> out (num_channels)
            in_residual = torch.tanh(out_hidden[0]) * torch.sigmoid(out_hidden[1])
            skip_connections.append(in_residual)

            # in (num_channels) --> residual --> out (num_channels) = x
            out_residual = residual(in_residual)
            out = out_residual + x[:, :, -out_residual.size(2):]

        # Mixer lineal recibe las skips-connections
        out = torch.cat([s[:, :, -out.size(2):] for s in skip_connections], dim=1)  # Concatena las salidas de las skip-connections
        return self.linear_mixer(out)

class AmpEmulatorModel(pl.LightningModule):
    """
    Clase que implementa el modelo basado en WaveNet propuesto en el paper
    "Real-Time Guitar Amplifier Emulation with Deep Learning" de Wright, et al., 2020.
    """
    
    def __init__(self, num_channels=4, dilation_depth=9, dilation_repeat=2, kernel_size=3, learning_rate=3e-3):
        """
        Constructor de la clase.
        """
        super(AmpEmulatorModel, self).__init__()
        
        self.wavenet = WaveNet(
            num_channels=num_channels,
            dilation_depth=dilation_depth,
            dilation_repeat=dilation_repeat,
            kernel_size=kernel_size
        )
        
        self.learning_rate = 3e-3
        
    def __pre_emphasis_filter(x, alpha=0.95):
        """
        Método que implementa el filtro de pre-énfasis de paso alto de primer orden.

        Args:
            x (torch.Tensor): Tensor con las señales de audio.
            alpha (float): Coeficiente de pre-énfasis. Por defecto 0.95.
        """
        return torch.cat([x[:, 0:1], x[:, 1:] - alpha * x[:, :-1]], dim=1)

    def __ESR(self, y, y_hat):
        """
        Método que implementa la función de pérdida ESR (Error-to-Signal Ratio) propuesta en el paper.

        Args:
            y (torch.Tensor): Tensor con las señales de audio originales.
            y_hat (torch.Tensor): Tensor con las señales de audio generadas por el modelo.
        """
        y, y_hat = self.__pre_emphasis_filter(y), self.__pre_emphasis_filter(y_hat)
        return torch.sum(torch.pow(y - y_hat, 2), dim=2) / torch.sum(torch.pow(y, 2), dim=2)

    def forward(self, x):
        """
        Método que implementa el paso hacia adelante de la red neuronal.
        """
        return self.wavenet(x)
    
    # Métodos overriden de Lightning.LightningModule

    def training_step(self, batch):
        """
        Método del bucle de entrenamiento. Implementa el paso hacia adelante de la red neuronal y el cálculo de la pérdida.

        Args:
            batch: lote de datos de entrada y salida.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.__ESR(y[:, :, -y_hat.size(2) :], y_hat).mean()
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch):
        """
        Método del bucle de validación. Implementa el paso hacia adelante de la red neuronal y el cálculo de la pérdida.

        Args:
            batch: lote de datos de entrada y salida.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.__ESR(y[:, :, -y_hat.size(2) :], y_hat).mean()
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """ 
        Método overriden que devuelve el optimizador del modelo, en el caso del paper, Adam.
        """
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.learning_rate)