import os

import matplotlib.pyplot as plt

# Anomalib imports
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from PIL import Image


if __name__ == "__main__":
    print("Step 1: Configuring Dataset...")

    datamodule = Folder(
        name="chest_xray",
        root="dataset",
        normal_dir="train/NORMAL",
        abnormal_dir=["test/DEFECT", "test/TB", "train/PNEUMONIA"],
        normal_test_dir="test/NORMAL",
        val_split_mode="from_test",
        val_split_ratio=0.5,
        train_batch_size=8,
        num_workers=0,  
    )
    datamodule.setup()

    print("Step 2: Initializing DeepCXR Model...")
    model = Patchcore(backbone="resnet18", coreset_sampling_ratio=0.005)

    print("Step 3: Initializing Engine...")
    engine = Engine()

    print("Step 4: Running Evaluation...")
    engine.test(model=model, datamodule=datamodule, ckpt_path="weights/model.ckpt")
