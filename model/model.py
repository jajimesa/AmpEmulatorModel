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

import pytorch_lightning as pl

import numpy as np
from tqdm import tqdm

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
            dilation (int): Tasa de dilatación de la convolución. Por defecto 1 (sin dilatación).
            groups (int): Número de grupos en los que se dividen las entradas y salidas. Por defecto 1.
            bias (bool): Indica si se incluye un término de sesgo en la convolución. Por defecto True.
        """
        self.__dilated_padding = (kernel_size - 1) * dilation   # Padding que garantiza que la convolución sea causal.

        super(DilatedCausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            padding = self.__dilated_padding,
            dilation = dilation,
            groups = groups,
            bias = bias
        )

    def forward(self, input):
        """
        Método que implementa el paso hacia adelante de la capa de convolución.            
        
        Args:
            input (torch.Tensor): Tensor de entrada a la capa de convolución.
        """
        output = super(DilatedCausalConv1d, self).forward(input)

        if self.__dilated_padding != 0:
            output = output[:, :, :-self.__dilated_padding]     # Garantiza que la salida sea causal.

        return output
    
class WaveNet(nn.Module):
    """
    Clase que implementa la red neuronal propuesta en 
    "Real-Time Guitar Amplifier Emulation with Deep Learning" de Wright, et al., 2020.
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
            in_channels = 1,
            out_channels = num_channels,
            kernel_size = 1
        )

        # El output de la capa de entrada se pasa al bloque residual
        dilations = self.__build_dilations(dilation_depth, dilation_repeat)
        self.hidden_stack = self.__convolution_stack(
            in_channels = num_channels,
            out_channels = 2 * num_channels,  # Se duplica la salida para usar la gated activation unit
            kernel_size = kernel_size,
            dilations = dilations
        )
        self.residual_stack = self.__convolution_stack(
            in_channels = num_channels,
            out_channels = num_channels,
            kernel_size = 1,
            dilations = dilations
        )

        # Las salidas "skip-connections" pasan al mixer lineal para generar la salida final
        self.linear_mixer = nn.Conv1d(
            in_channels = num_channels * dilation_depth * dilation_repeat,   # Número de salidas de las skip-connections
            out_channels = 1,
            kernel_size = 1
        )

    def __build_dilations(self, dilation_depth, dilations_repeat):
        """
        Método que construye la lista de tasas de dilatación.

        Ejemplos: 
            depth=3, repeat=2 -> [[1, 2, 4], [1, 2, 4]]
            depth=2, repeat=3 -> [[1, 2], [1, 2], [1, 2]]
            depth=9, repeat=2 -> [[1, 2, ... , 128, 256], [1, 2, ... , 128, 256]]

        Args:
            dilation_depth (int): Máxima potencia de 2 que tendrá la tasa de dilatación.
            dilations_repeat (int): Número de veces que se repite el rango de potencias de 2.

        Returns:
            dilations (list): Lista de listas de tasas de dilatación.
        """
        dilations = []
        for i in range(dilations_repeat):
            for j in range(dilation_depth):
                dilations.append(2**j)
        return dilations
    
    def __convolution_stack(self, in_channels, out_channels, kernel_size, dilations):
        """
        Función auxiliar que construye una pila de capas de convolución. Devuelve una pila
        de capas de convolución de tamaño igual a la longitud de la lista dilations.

        Args:
            in_channels (int): Número de canales de entrada.
            out_channels (int): Número de canales de salida.
            kernel_size (int): Tamaño del kernel de la convolución.
            dilation_array (list): Lista de tasas de dilatación de las convoluciones.
        
        Returns:
            pila (nn.ModuleList): Pila de capas de convolución.
        """
        pila = nn.ModuleList()
        for v, d in enumerate(dilations):   # dilations es una lista de listas v, con dilataciones d
            pila.append(
                DilatedCausalConv1d(
                    in_channels,
                    out_channels,
                    kernel_size = kernel_size,
                    dilation = d,
                )
            )     
        return pila
    
    def receptive_field(self):
        """
        Método que calcula el campo receptivo de la red neuronal.

        Returns:
            receptive_field (int): Tamaño del receptive field de la red neuronal.
        """
        return self.dilation_repeat * (2 ** self.dilation_depth - 1) * (self.kernel_size - 1) + 1   

    def forward(self, x):
        """
        Método que implementa el paso hacia adelante de la red neuronal. Los datos de entrada "x"
        atraviesan la capa de entrada, el bloque residual (tantas veces como capas convolucionales)
        y el mixer lineal para producir el output.

        Args:
            x (torch.Tensor): Tensor de entrada a la red neuronal.

        Returns:
            out (torch.Tensor): Tensor de salida de la red neuronal.
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
        Constructor de la clase. Los argumentos por defecto se corresponden con WaveNet2:

        ====================================================
        | Model           | WaveNet1 | WaveNet2 | WaveNet3 |
        |-----------------|----------|----------|----------|
        | Layers          | 10       | 18       | 18       |
        | Channels        | 16       | 8        | 16       |
        |-----------------|----------|----------|----------|
        | num_channels    | 8        | 4        | 8        |
        | dilation_depth  | 10       | 9        | 9        |
        | dilation_repeat | 1        | 2        | 2        |
        ====================================================
        """
        super(AmpEmulatorModel, self).__init__()
        
        self.wavenet = WaveNet(
            num_channels = num_channels,
            dilation_depth = dilation_depth,
            dilation_repeat = dilation_repeat,
            kernel_size = kernel_size
        )
        
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Método que implementa el paso hacia adelante de la red neuronal.
        """
        return self.wavenet(x)
    
    def __pre_emphasis_filter(self, x, alpha=0.95):
        """
        Método que implementa el filtro de pre-énfasis de paso alto de primer orden.

        Args:
            x (torch.Tensor): Tensor con las señales de audio.
            alpha (float): Coeficiente de pre-énfasis. Por defecto 0.95.

        Returns:
            out (torch.Tensor): Tensor con las señales de audio enfatizadas en el rango de frecuencias medias y altas.
        """
        return torch.cat([x[:, :, 0:1], x[:, :, 1:] - alpha * x[:, :, :-1]], dim=2)

    def __error_to_signal_ratio(self, y, y_hat):
        """
        Método que implementa la función de pérdida ESR (Error-to-Signal Ratio) sobre tensores.
        
        Args:
            y (torch.Tensor): Tensor con las señales de audio originales.
            y_hat (torch.Tensor): Tensor con las señales de audio generadas por el modelo.

        Returns:
            ESR (torch.Tensor): Tensor con el valor de la función de pérdida ESR.
        """
        y, y_hat = self.__pre_emphasis_filter(y), self.__pre_emphasis_filter(y_hat)

        # ¡Añadimos un pequeño valor para evitar la división por cero!
        return torch.sum(torch.pow(y - y_hat, 2), dim=2) / (torch.sum(torch.pow(y, 2), dim=2) + 1e-10) 



    # Métodos overriden de Lightning.LightningModule

    def training_step(self, batch):
        """
        Método del bucle de entrenamiento. Implementa el paso hacia adelante de la red neuronal y el cálculo de la pérdida.

        Args:
            batch: lote de datos de entrada y salida.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.__error_to_signal_ratio(y[:, :, -y_hat.size(2) :], y_hat).mean()
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
        loss = self.__error_to_signal_ratio(y[:, :, -y_hat.size(2) :], y_hat).mean()
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """ 
        Método overriden que devuelve el optimizador del modelo, en el caso del paper, Adam.

        Returns:
            optimizer (torch.optim.Adam): Optimizador Adam.
        """
        return torch.optim.Adam(self.wavenet.parameters(), lr=self.learning_rate)
    
    def inference(self, x, batch_size, sample_size):
        """
        Método que implementa la inferencia del modelo.

        Args:
            x (np.ndarray): Datos preparados para inferir por dataset.prepare_for_inference().
            batch_size (int): Tamaño del lote de datos.
            shape (int): Tamaño de las señales de audio. Por defecto 4410.
        Returns:
            y_hat (np.ndarray): Datos inferidos por el modelo.
        """
        with torch.no_grad():
            y_hat = []
            batches = x.shape[0] // batch_size

            # Inferencia por lotes con barras de progreso
            for batch in tqdm(np.array_split(x, batches)):
                inference = self(torch.from_numpy(batch)).numpy()
                y_hat.append(inference)

        y_hat = np.concatenate(y_hat)
        y_hat = y_hat[:, :, -sample_size :]

        return y_hat