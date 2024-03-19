from captum.attr import DeepLift
from models.FCNN import FCNN
from models.AE import netBio
from pandas import DataFrame
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Iterable, Dict
import warnings


class ArtificialDataset(Dataset):
    def __init__(self, df: DataFrame):
        super().__init__()
        self.X = df.drop(labels=["Label"], axis=1)
        try:
            self.X = self.X.drop(labels=["Name"], axis=1)
        except KeyError:
            # If there is no Name column, no need to do anything
            pass

        # Make sure that the index is akin to list(range(self.__len__()))
        self.X.reset_index(inplace=True, drop=True)

        self.y = df["Label"]
        self.y.reset_index(inplace=True, drop=True)

    def __getitem__(self, index: int):
        return torch.tensor(self.X.iloc[index], dtype=torch.float32), torch.tensor(
            self.y.iloc[index]
        )

    def __len__(self) -> int:
        return len(self.y)


def bilevel_proj_l1Inftyball(w2, eta: float, device="cpu"):

    w = torch.as_tensor(w2, dtype=torch.get_default_dtype(), device=device)

    if w.dim() == 1:
        Q = proj_l1ball(w, eta, device=device)
    else:

        init_shape = w.shape
        Res = torch.empty(init_shape, device=device)
        nrow, ncol = init_shape[0:2]

        W = torch.tensor(
            [torch.max(torch.abs(w[:, i])).data.item() for i in range(ncol)]
        )

        PW = proj_l1ball(W, eta, device=device)

        for i in range(ncol):
            Res[:, i] = torch.clamp(torch.abs(w[:, i]), max=PW[i].data.item())
            Res[:, i] = Res[:, i] * torch.sign(w[:, i])

        Q = Res.clone().detach().requires_grad_(True).to(device)

    if not torch.is_tensor(w2):
        Q = Q.data.numpy(force=True)

    return Q


def proj_l1ball(w0, eta: float, device="cpu"):
    # To help you understand, this function will perform as follows:
    #    a1 = torch.cumsum(torch.sort(torch.abs(y),dim = 0,descending=True)[0],dim=0)
    #    a2 = (a1 - eta)/(torch.arange(start=1,end=y.shape[0]+1))
    #    a3 = torch.abs(y)- torch.max(torch.cat((a2,torch.tensor([0.0]))))
    #    a4 = torch.max(a3,torch.zeros_like(y))
    #    a5 = a4*torch.sign(y)
    #    return a5

    w = torch.as_tensor(w0, dtype=torch.get_default_dtype(), device=device)

    init_shape = w.size()

    if w.dim() > 1:
        init_shape = w.size()
        w = w.reshape(-1)

    Res = torch.sign(w) * torch.max(
        torch.abs(w)
        - torch.max(
            torch.cat(
                (
                    (
                        torch.cumsum(
                            torch.sort(torch.abs(w), dim=0, descending=True)[0],
                            dim=0,
                            dtype=torch.get_default_dtype(),
                        )
                        - eta
                    )
                    / torch.arange(
                        start=1,
                        end=w.numel() + 1,
                        device=device,
                        dtype=torch.get_default_dtype(),
                    ),
                    torch.tensor([0.0], dtype=torch.get_default_dtype(), device=device),
                )
            )
        ),
        torch.zeros_like(w),
    )

    Q = Res.reshape(init_shape).clone().detach()

    if not torch.is_tensor(w0):
        Q = Q.data.numpy()
    return Q


def compute_mask(
    net: torch.nn.Module, projection, projection_param: float, device="cpu"
) -> Dict[int, torch.Tensor]:
    tol = 1.0e-4  # threshold under which a weight is considered zero
    all_masks = {}
    for index, param in enumerate(net.parameters()):
        if index < len(list(net.parameters())) - 2 and index % 2 == 0:
            # If this layer has to be projected
            # (we don't project bias layers or final layers)
            mask = torch.where(
                condition=(
                    torch.abs(
                        projection(param.detach().clone(), projection_param, device)
                    )
                    < tol
                ).to(device),
                input=torch.zeros_like(param),
                other=torch.ones_like(param),
            )
            all_masks[index] = mask
    return all_masks


def mask_gradient(module, _):
    for index, param in enumerate(module.parameters()):
        if index < len(list(module.parameters())) - 2 and index % 2 == 0:
            param.data = getattr(module, f"mask_{index}") * param.data


def prog(iterable: Iterable, verbose=False):
    """Wrap an iterable in a progress bar if verbose is True"""
    if verbose:
        return tqdm(iterable)
    return iterable


def get_feature_weights(model: torch.nn.Module, test_dl: DataLoader, device="cpu"):
    # Suppress captum UserWarning
    warnings.filterwarnings(action="ignore", category=UserWarning)

    # Create dataloader to get all of the test data in one chunk
    dl = DataLoader(test_dl.dataset, batch_size=len(test_dl.dataset), shuffle=False)

    for x, _ in dl:
        inputs = x.to(device).requires_grad_()

    if isinstance(model, FCNN):
        deepl = DeepLift(model)
    elif isinstance(model, netBio):
        # We only care about which features are used for classification
        deepl = DeepLift(model.encoder)
    else:
        raise NotImplementedError("This model class is not implemented.")

    attributions = deepl.attribute(inputs, target=1).abs().sum(dim=0)
    weights, indexes = attributions.sort(descending=True)
    weights /= weights.max()  # Normalize to have weights descending from 1

    return (
        weights.numpy(force=True),  # Feature weights, ranked
        dl.dataset.X.columns[
            indexes.numpy(force=True)
        ].to_list(),  # feature names, in order
    )


def get_features_from_df(df: DataFrame):
    df = df.drop("Label", axis=1)
    try:
        df = df.drop(labels=["Name"], axis=1)
    except KeyError:
        # If there is no Name column, no need to do anything
        pass
    return df.columns
