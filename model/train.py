from model import AmpEmulatorModel
from data import AmpEmulatorDataModule
import pytorch_lightning as pl

import torch

def train():
    """
    Función principal para entrenar el modelo.
    """

    print(torch.cuda.is_available())

    # Cargamos el dataset
    dataset = AmpEmulatorDataModule()
    dataset.setup()
    dataset.prepare_data()

    # Cargamos el modelo
    model = AmpEmulatorModel()

    # Entrenamos el modelo
    model.train()           # Ponemos el modelo en modo de entrenamiento (REDUNDANTE, ya lo hace el Trainer)
    trainer = pl.Trainer(
        accelerator="cuda", # Usamos la GPU para entrenar el modelo
        log_every_n_steps=100,
        max_epochs=1000
    )
    
    # Para empezar desde cero el entrenamiento
    #trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())

    # Para cargar el checkpoint
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader(), ckpt_path="models/model.ckpt")
    trainer.save_checkpoint("models/model.ckpt")

if __name__ == "__main__":
    train()