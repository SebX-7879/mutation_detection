"""
End-to-end pipeline for training and evaluating the model.
Most of the code is taken from main.ipynb and converted into a script.
We propose here a robust (limit the impact of weights initialiazation) and efficient (use of GPU) and reproducible (fix the seed) pipeline.


The pipeline is divided into 3 main steps:
1. Data preparation
2. Model training
3. Model evaluation

"""


## 0. LIBRARIES

from pathlib import Path 
import sys
import warnings
from copy import deepcopy
import multiprocessing

import numpy as np
import pandas as pd 
import torch
import torch.nn as nn

from datetime import datetime
from IPython.display import clear_output

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## In-project libraries
from datasets.core import SlideFeaturesDataset
from models.chowder import Chowder
from utils.features import pad_collate_fn
from utils.functional import sigmoid, softmax
from trainer import TorchTrainer
from trainer.utils import slide_level_train_step, slide_level_val_step

working_directory = Path(".").resolve() 
sys.path.append(str(working_directory))


def main():
    ## I- DATA PREPARATION

    # 1. Load the precomputed features

    ## Train and test directories
    train_features_dir = working_directory / "data" / "train_input" / "moco_features"
    test_features_dir = working_directory / "data" / "test_input" / "moco_features"

    ## List of all the files in each directory
    train_features_path_all = list(train_features_dir.glob("*.npy"))
    test_features_path_all = list(test_features_dir.glob("*.npy"))

    # 2. Load the metadata
    train_metadata_df = pd.read_csv(working_directory / "data" / "supplementary_data" / "train_metadata.csv")
    test_metadata_df = pd.read_csv(working_directory / "data" / "supplementary_data" / "test_metadata.csv")

    # 3. Load the traning labels
    y_train = pd.read_csv(working_directory / "data" / "train_output.csv")
    ## And concatenate the labels to the metadata
    train_metadata_df = pd.merge(train_metadata_df, y_train, on="Sample ID")

    y_train = train_metadata_df["Target"].values.astype(np.float32) ## float32 required for BCE loss

    # 4. Train dataset
    train_dataset = SlideFeaturesDataset(
    features = train_features_path_all,
    labels = y_train,
    n_tiles=1000,
    shuffle=True,
    transform=None
    )
    # 5. Test dataset
    test_dataset = SlideFeaturesDataset(
    features = test_features_path_all,
    labels = np.zeros(len(test_features_path_all), dtype=np.float32), ## Dummy labels, won't be used
    n_tiles=1000,
    shuffle=False,
    transform=None
    )

    train_indices = np.arange(len(train_dataset))
    train_labels = train_dataset.labels


    ## II- MODEL TRAINING

    # 1. Hyperparameters
    in_features = 2048
    out_features = 1
    n_top = 5
    n_bottom = 5
    tiles_mlp_hidden = None
    mlp_hidden = [128,64]
    mlp_activation = torch.nn.LeakyReLU()
    mlp_dropout = [0.1,0.1]
    bias = True

    # 2. Model initialization
    chowder = Chowder(
        in_features=in_features,
        out_features=out_features,
        n_top=n_top,
        n_bottom=n_bottom,
        tiles_mlp_hidden=tiles_mlp_hidden,
        mlp_hidden=mlp_hidden,
        mlp_activation=mlp_activation,
        mlp_dropout=mlp_dropout,
        bias=bias
    )
    print_trainable_parameters(chowder)

    ## We define the loss function, optimizer and metrics for the training
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam              # Adam optimizer
    metrics = {"auc": roc_auc_score}          # AUC will be the tracking metric

    # 3. Instantiate the trainer
    trainer = TorchTrainer(
        model=chowder,
        criterion=criterion,
        metrics=metrics,
        batch_size=8,                           # you can tweak this
        num_epochs=20,                           # you can tweak this
        learning_rate=1e-3,                      # you can tweak this
        weight_decay=0,                        # you can tweak this
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        optimizer=deepcopy(optimizer),
        train_step=slide_level_train_step,
        val_step=slide_level_val_step,
        collator=pad_collate_fn,
        use_tqdm=True,
    )
    
    # 4. Train-validation split

    train_indices_, val_indices_ = train_test_split(train_indices, test_size=0.2, stratify=train_labels, random_state=42)
    train_dataset_ = torch.utils.data.Subset(train_dataset, train_indices_)
    val_dataset_ = torch.utils.data.Subset(train_dataset, val_indices_)

    # 5. Training 
    ## Logging of the hyperparameters
    print("-"*50)
    print("Training the {} model".format(chowder.__class__.__name__))
    print("On {} samples, validating on {} samples\n".format(len(train_dataset_), len(val_dataset_)))
    print("-"*50)
    print("Hyperparameters:")
    print(f"num_epochs: {trainer.num_epochs}")
    print(f"learning_rate: {trainer.learning_rate}")
    print(f"weight_decay: {trainer.weight_decay}")
    print(f"mlp_hidden: {mlp_hidden}")
    print(f"mlp_activation: {mlp_activation}")
    print(f"mlp_dropout: {mlp_dropout}")
    print(f"batch_size: {trainer.batch_size}")
    print(f"device: {trainer.device}")
    print("-"*50)
    
    ## training loop
    start = datetime.now()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Training step for the given number of epochs
        train_metrics, val_metrics = trainer.train(
            train_dataset_, val_dataset_
        )
        # Predictions on test (logits, sigmoid(logits) = probability)
        test_logits = trainer.predict(test_dataset)[1]

    
    ## plot the training and validation loss and metrics and save them in a txt file
    plot_loss_auc(trainer, train_metrics, val_metrics)
    end = datetime.now()
    print("Time taken to train the model: {}".format(end-start))


    ## III- MODEL EVALUATION
    
    test_probas = np.mean([sigmoid(logits) for logits in test_logits], axis=0).squeeze()

    ## Prediction to dataframe
    submission = pd.DataFrame(
    {"Sample ID": test_metadata_df["Sample ID"].values, "Target": test_probas}
    ).sort_values(
    "Sample ID"
    )  # extra step to sort the sample IDs

    # sanity checks
    assert all(submission["Target"].between(0, 1)), "`Target` values must be in [0, 1]"
    assert submission.shape == (149, 2), "Your submission file must be of shape (149, 2)"
    assert list(submission.columns) == [
    "Sample ID",
    "Target",
    ], "Your submission file must have columns `Sample ID` and `Target`"


    # save the submission as a csv file
    output_dir = working_directory / "test_output"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    saving_path = output_dir / f"submission_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    submission.to_csv(saving_path, index=None)
    #submission.head()
    ## Save all the hyperparameters in a txt file
    log_dir = working_directory / "logs"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    with open(log_dir / f"hyperparameters_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w") as f:
        f.write(f"num_epochs: {trainer.num_epochs}\n")
        f.write(f"learning_rate: {trainer.learning_rate}\n")
        f.write(f"weight_decay: {trainer.weight_decay}\n")
        f.write(f"mlp_hidden: {mlp_hidden}\n")
        f.write(f"mlp_activation: {mlp_activation}\n")
        f.write(f"mlp_dropout: {mlp_dropout}\n")
        f.write(f"batch_size: {trainer.batch_size}\n")
        f.write(f"device: {trainer.device}\n")
        f.write(f"Time taken to train the model: {end-start}\n")
    print("Submission file saved in \"test_output\" folder.")
    print("Hyperparameters saved in \"logs\" folder.")

    print("-"*50)


def print_trainable_parameters(model: torch.nn) -> None:
    """Print number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param}"
        f" || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def plot_loss_auc(trainer, train_metrics, val_metrics):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].plot(trainer.train_losses, label="train")
    axs[0].plot(trainer.val_losses, label="val")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[1].plot(train_metrics["auc"], label="train")
    axs[1].plot(val_metrics["auc"], label="val")
    axs[1].set_title("AUC")
    axs[1].legend()
    plt.savefig(f"figures/loss_auc_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
    plt.show()
    print("Loss and AUC plots saved in figures folder.")
    
## Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(317)
    print("Start of the pipeline")
    main()
    print("End of the pipeline")