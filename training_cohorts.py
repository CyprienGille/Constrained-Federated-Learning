import os
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch.nn import CrossEntropyLoss, HuberLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional

from models.FCNN import FCNN
from models.AE import netBio
from functions_cohorts import (
    ArtificialDataset,
    bilevel_proj_l1Inftyball,
    compute_mask,
    get_feature_weights,
    get_features_from_df,
    mask_gradient,
    prog,
)
from cohort_splitting import split_into_chunks

## Params
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4
PROJECTION = bilevel_proj_l1Inftyball
ETA = 0.05
N_COHORTS = 2
LOSS_LAMBDA = 0.0005
N_FOLDS = 4  # For cross-validation
VERBOSE = True  # To get print statements indicating the steps
PROGRESS_BARS = False
LABELS_COLUMN_NAME = "Label"
MODELS_DIR = "saved_models/"
RESULTS_DIR = "results/"

MODEL_TYPE = FCNN
# MODEL_TYPE = netBio

## !Note: if you modify the input data, make sure to check that N_FEATURES and feature_axis are correct!
# data_path = "data/Synth_300f_20inf_10000s.csv"
# data_path = "data/GC_Breast_D_MB.csv"
data_path = "data/LUNG.csv"

# 0 if the features are along the columns (e.g. for synthetic data)
# 1 if the features are along the rows (e.g. for Breast or LUNG)
feature_axis = 1

DO_LOG = True  # Log-transform the data
DO_SCALE = True  # Normalize the data (zero mean, unit variance)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## Functions
def double_descent(
    train_dl,
    test_dl,
    verbose=False,
    prog_per_epoch=False,
    save_models: Optional[str] = None,
):
    """Perform the entire double descent routine, with evalutation

    Parameters
    ----------
    train_dl : DataLoader
        The training set DataLoader
    test_dl : DataLoader
        The testing set DataLoader
    verbose : bool, optional
        Whether to print the advancement and accuracies, by default False
    prog_per_epoch : bool, optional
        Whether to display a progress bar for each epoch, by default False
    save_models : Optional[str], optional
        If not None, the name by which to save the model (with no extension), by default None

    Returns
    -------
    nn.Module
        The last model after the second descent
    """
    ## Model
    model = MODEL_TYPE(n_inputs=N_FEATURES)
    model = model.to(DEVICE)

    ## First Training
    if verbose:
        print(f"==First Descent==")
    full_training(model, train_dl, verbose=prog_per_epoch)
    first_acc = compute_metrics(model, test_dl)
    if save_models is not None:
        torch.save(model.state_dict(), MODELS_DIR + save_models + ".pth")
    if verbose:
        # Note: we evaluate the last model, which may not necessarily be the best
        # in the case of overfitting
        print(f"Test Accuracy of last model : {first_acc:.5f}")

    ## Second Training
    masks = compute_mask(model, PROJECTION, ETA, device=DEVICE)
    model_proj = MODEL_TYPE(n_inputs=N_FEATURES)
    # Add the masks as attributes of the model
    for index, mask in masks.items():
        model_proj.register_buffer(name=f"mask_{index}", tensor=mask, persistent=False)
    # Add the masking as an operation to be done before forward operations
    # Note : this simulates pruning
    model_proj.register_forward_pre_hook(mask_gradient)
    model_proj = model_proj.to(DEVICE)
    if verbose:
        # !! Note: this density formula assumes that we're only projecting the first layer
        # and that this layer has 300 hidden neurons
        # (which is true for the default FCNN)
        print(
            f"==Second Descent==\n(Density : {torch.sum(torch.flatten(masks[0]))/(N_FEATURES*300):.4f})"
        )
    full_training(model_proj, train_dl, verbose=prog_per_epoch)
    second_acc = compute_metrics(model_proj, test_dl)
    if save_models is not None:
        torch.save(model_proj.state_dict(), MODELS_DIR + save_models + "_proj.pth")
    if verbose:
        print(f"Test Accuracy of last model : {second_acc:.5f}")
    return model, model_proj, first_acc, second_acc


def full_training(model, train_dl, verbose=False):
    model.train()
    classif_loss = CrossEntropyLoss()
    recon_loss = HuberLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for n in range(N_EPOCHS):
        avg_loss = 0.0
        for x, labels in prog(train_dl, verbose):
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(x)
            if MODEL_TYPE == FCNN:
                loss = classif_loss(out, labels.long())
            elif MODEL_TYPE == netBio:
                loss = classif_loss(out[0], labels.long()) + LOSS_LAMBDA * recon_loss(
                    out[1], x
                )
            else:
                raise NotImplementedError("This model is not implemented.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.detach().cpu().item()
        avg_loss /= len(train_dl)
        if verbose:
            print(f"Epoch {n} : Avg training loss {avg_loss:.6f}")


@torch.no_grad()
def compute_metrics(model: torch.nn.Module, test_dl: DataLoader, verbose=False):
    """Note: This assumes that the test batch size is 1 and will fail otherwise"""
    assert (
        test_dl.batch_size == 1
    ), f"The DataLoader's batch size must be 1, got {test_dl.batch_size}"

    model.eval()
    acc = 0.0
    for x, label in prog(test_dl, verbose):
        x = x.to(DEVICE)
        label = label.item()
        out = model(x)
        if MODEL_TYPE == FCNN:
            pred = out.argmax().item()
        elif MODEL_TYPE == netBio:
            pred = out[0].argmax().item()
        else:
            raise NotImplementedError("This model is not implemented.")
        if pred == label:
            # If we correctly classified this sample
            acc += 1.0
    acc /= len(test_dl)
    return acc


if __name__ == "__main__":
    # Make the subdirectories to save the models and results
    MODELS_DIR += data_path[5:-4] + "/"
    RESULTS_DIR += data_path[5:-4] + "/"
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if feature_axis == 0:
        df = pd.read_csv(data_path, sep=";", header=0)
    elif feature_axis == 1:
        df = pd.read_csv(data_path, sep=";", header=0).transpose()
        df.columns = df.loc["Name"]
        df.drop("Name", axis=0, inplace=True)
        if df["Label"].max() == 2:
            # If the labels are in [1, 2], map them to [0, 1]
            df["Label"] -= 1
    else:
        raise ValueError(f"feature_axis needs to be 0 or 1, got {feature_axis}")

    N_FEATURES = len(df.columns)

    for cohort_i, df_cohort in enumerate(split_into_chunks(df, N_COHORTS)):
        if VERBOSE:
            print(f"\n====Cohort {cohort_i+1}/{N_COHORTS}====")

        # Prepare the result and feature rank dataframes
        results_df = pd.DataFrame(
            index=[f"Fold {i}" for i in range(N_FOLDS)] + ["Mean", "Std"],
            columns=["Test Accuracy"],
        )
        results_df_proj = pd.DataFrame(
            index=[f"Fold {i}" for i in range(N_FOLDS)] + ["Mean", "Std"],
            columns=["Test Accuracy"],
        )
        features_df = pd.DataFrame(
            columns=[f"Fold {i}" for i in range(N_FOLDS)] + ["Mean", "Std"],
            index=get_features_from_df(df),
        )
        features_df_proj = pd.DataFrame(
            columns=[f"Fold {i}" for i in range(N_FOLDS)] + ["Mean", "Std"],
            index=get_features_from_df(df),
        )

        kf = KFold(n_splits=N_FOLDS)  # Cross validation
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(df_cohort.index)):
            if VERBOSE:
                print(f"===Fold {fold_i+1}/{N_FOLDS}===")
            train_df = df_cohort.iloc[train_idx].copy()
            test_df = df_cohort.iloc[test_idx].copy()

            try:
                train_ds = ArtificialDataset(train_df, DO_LOG, DO_SCALE)
                test_ds = ArtificialDataset(test_df, DO_LOG, DO_SCALE)
            except KeyError:
                raise KeyError(
                    "'Name' and/or 'Label' column(s) not found. Make sure that your data contains these rows/columns and that the value of feature_axis is correct."
                )

            train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            test_dl = DataLoader(test_ds, batch_size=1, shuffle=True)

            last_model, last_model_proj, acc, acc_proj = double_descent(
                train_dl,
                test_dl,
                verbose=VERBOSE,
                prog_per_epoch=PROGRESS_BARS,
                save_models=f"{MODEL_TYPE.__name__}_cohort{cohort_i}_fold{fold_i}",
            )

            f_weigths, ranked_f_names = get_feature_weights(
                last_model, test_dl, device=DEVICE
            )
            f_weigths_proj, ranked_f_names_proj = get_feature_weights(
                last_model_proj, test_dl, device=DEVICE
            )

            # Fill the DataFrames
            results_df.at[f"Fold {fold_i}", "Test Accuracy"] = acc
            results_df_proj.at[f"Fold {fold_i}", "Test Accuracy"] = acc_proj

            for weight, name in zip(f_weigths, ranked_f_names):
                features_df.at[name, f"Fold {fold_i}"] = weight
            for weight, name in zip(f_weigths_proj, ranked_f_names_proj):
                features_df_proj.at[name, f"Fold {fold_i}"] = weight

        mean = results_df["Test Accuracy"].mean()
        std = results_df["Test Accuracy"].std()
        mean_proj = results_df_proj["Test Accuracy"].mean()
        std_proj = results_df_proj["Test Accuracy"].std()

        results_df.at["Mean", "Test Accuracy"] = mean
        results_df.at["Std", "Test Accuracy"] = std
        results_df_proj.at["Mean", "Test Accuracy"] = mean_proj
        results_df_proj.at["Std", "Test Accuracy"] = std_proj

        features_df["Mean"] = features_df[[f"Fold {i}" for i in range(N_FOLDS)]].mean(
            axis=1
        )
        features_df_proj["Mean"] = features_df_proj[
            [f"Fold {i}" for i in range(N_FOLDS)]
        ].mean(axis=1)
        features_df["Std"] = features_df[[f"Fold {i}" for i in range(N_FOLDS)]].std(
            axis=1
        )
        features_df_proj["Std"] = features_df_proj[
            [f"Fold {i}" for i in range(N_FOLDS)]
        ].std(axis=1)

        # Sort the feature rankings
        features_df.sort_values(by="Mean", ascending=False, inplace=True)
        features_df_proj.sort_values(by="Mean", ascending=False, inplace=True)

        if VERBOSE:
            print("Results (First descent):")
            print(results_df)
            print("Results (Second descent):")
            print(results_df_proj)
            print("Top 5 Features (First descent):")
            print(features_df.head(5))
            print("Top 5 Features (Second descent):")
            print(features_df_proj.head(5))

        results_df.to_csv(
            RESULTS_DIR + f"results_{MODEL_TYPE.__name__}_cohort{cohort_i}.csv"
        )
        results_df_proj.to_csv(
            RESULTS_DIR + f"results_{MODEL_TYPE.__name__}_cohort{cohort_i}_proj.csv"
        )
        features_df.to_csv(
            RESULTS_DIR + f"features_{MODEL_TYPE.__name__}_cohort{cohort_i}.csv"
        )
        features_df_proj.to_csv(
            RESULTS_DIR + f"features_{MODEL_TYPE.__name__}_cohort{cohort_i}_proj.csv"
        )
