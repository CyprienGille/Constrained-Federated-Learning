import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from dataset import get_train_test_data, ArtificialDataset
from models.FCNN import FCNN
from projections import proj_l1infball

## Params
TRAIN_PROP = 0.8
N_SAMPLES = 10_000
N_FEATURES = 200
N_INFORMATIVE = 10
N_REDUNDANT = 0
N_REPEATED = 0
N_CLUSTERS_PER_CLASS = 1
WEIGHTS = None  # [0.9, 0.1] for example to make unbalanced classes
SEPARABILITY = 1.0  # Increase this to facilitate classification
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 1e-4
PROJECTION = proj_l1infball
ETA = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


## Functions
def compute_mask(net, projection, projection_param):
    tol = 1.0e-4  # threshold under which a weight is considered zero
    full_mask = []
    for index, param in enumerate(net.parameters()):
        if index < len(list(net.parameters())) - 2 and index % 2 == 0:
            # If this layer has to be projected
            # (we don't project bias layers or final layers)
            mask = torch.where(
                condition=(
                    torch.abs(
                        projection(param.detach().clone(), projection_param, DEVICE)
                    )
                    < tol
                ).to(DEVICE),
                input=torch.zeros_like(param),
                other=torch.ones_like(param),
            )
        else:
            # Otherwise we do not care about the mask
            mask = torch.ones_like(mask)
        full_mask.append(mask)
    # turn list of masks into full mask tensor
    return torch.stack(full_mask)


def mask_gradient(module, _):
    for index, param in enumerate(module.parameters()):
        if index < len(list(module.parameters())) - 2 and index % 2 == 0:
            param.data = module.mask[index] * param.data


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
    if verbose:
        print("\nInstantiating model...")
    model = FCNN(n_inputs=N_FEATURES)
    model = model.to(DEVICE)

    ## First Training
    if verbose:
        print(f"===First Descent===")
    full_training(model, train_dl, verbose=prog_per_epoch)
    if save_models is not None:
        torch.save(model.state_dict(), save_models + ".pth")
    if verbose:
        # Note: we evaluate the last model, which may not necessarily be the best
        # in the case of overfitting
        print(f"Test Accuracy of last model : {compute_metrics(model, test_dl)}")

    ## Second Training
    mask = compute_mask(model, PROJECTION, ETA)
    model = FCNN(n_inputs=N_FEATURES)
    # Add the mask as an attribute of the model
    model.register_buffer(name="mask", tensor=mask)
    # Add the masking as an operation to be done before forward operations
    # Note : this simulates pruning
    model.register_forward_pre_hook(mask_gradient)
    model = model.to(DEVICE)
    if verbose:
        # Note: this density formula assumes that we're only projecting the first layer
        # and that this layer has 300 hidden neurons
        print(
            f"===Second Descent===\n(Density : {torch.sum(torch.flatten(mask[0]))/(N_FEATURES*300):.4f})"
        )
    full_training(model, train_dl, verbose=prog_per_epoch)
    if save_models is not None:
        torch.save(model.state_dict(), save_models + "_proj.pth")
    if verbose:
        print(f"Test Accuracy of last model : {compute_metrics(model, test_dl)}")
    return model


def full_training(model, train_dl, verbose=False):
    model.train()
    classif_loss = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for n in range(N_EPOCHS):
        avg_loss = 0.0
        for x, labels in prog(train_dl, verbose):
            x = x.to(DEVICE)
            labels = labels.to(DEVICE)

            out = model(x)
            loss = classif_loss(out, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.detach().cpu().item()
        avg_loss /= len(train_dl)
        if verbose:
            print(f"Epoch {n} : Avg training loss {avg_loss:.6f}")


@torch.no_grad()
def compute_metrics(model, test_dl, verbose=False):
    """Note: This assumes that the test batch size is 1 and will fail otherwise"""
    model.eval()
    acc = 0.0
    for x, label in prog(test_dl, verbose):
        x = x.to(DEVICE)
        label = label.item()
        out = model(x)
        pred = out.argmax().item()
        if pred == label:
            # If we correctly classified this sample
            acc += 1.0
    acc /= len(test_dl)
    return acc


def prog(iterable, verbose=False):
    """Wrap an iterable in a progress bar if verbose is True"""
    if verbose:
        return tqdm(iterable)
    return iterable


if __name__ == "__main__":
    ## Dataset and DataLoader
    print("Generating Dataset...")
    (
        X_train_1,
        y_train_1,
        X_test_1,
        y_test_1,
        X_train_2,
        y_train_2,
        X_test_2,
        y_test_2,
    ) = get_train_test_data(
        TRAIN_PROP,
        N_SAMPLES,
        N_FEATURES,
        N_INFORMATIVE,
        N_REDUNDANT,
        N_REPEATED,
        N_CLUSTERS_PER_CLASS,
        WEIGHTS,
        SEPARABILITY,
    )
    train_ds_1 = ArtificialDataset(X_train_1, y_train_1)
    test_ds_1 = ArtificialDataset(X_test_1, y_test_1)
    train_dl_1 = DataLoader(train_ds_1, batch_size=BATCH_SIZE, shuffle=True)
    test_dl_1 = DataLoader(test_ds_1, batch_size=1, shuffle=False)

    train_ds_2 = ArtificialDataset(X_train_2, y_train_2)
    test_ds_2 = ArtificialDataset(X_test_2, y_test_2)
    train_dl_2 = DataLoader(train_ds_2, batch_size=BATCH_SIZE, shuffle=True)
    test_dl_2 = DataLoader(test_ds_2, batch_size=1, shuffle=False)

    ## Training on both cohorts
    double_descent(train_dl_1, test_dl_1, verbose=True, save_models="model_1")
    double_descent(train_dl_2, test_dl_2, verbose=True, save_models="model_2")
