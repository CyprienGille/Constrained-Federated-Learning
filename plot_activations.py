import torch
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from models.FCNN import FCNN


def load_model(path, n_inputs=200):
    model = FCNN(n_inputs=n_inputs)
    try:
        model.load_state_dict(torch.load(path))
    except RuntimeError:
        model.register_buffer(name="mask", tensor=torch.zeros((4, 300, n_inputs)))
        model.load_state_dict(torch.load(path))
    return model


def show_matrix(models_list, fig_name):
    fig, axs = plt.subplots(nrows=1, ncols=len(models_list), sharex=True, sharey=True)
    fig.suptitle("First Layer Activations")

    images = []
    for i, ax in enumerate(axs.flat):
        model = models_list[i]
        for param in model.parameters():
            matrix = param
            break
        images.append(ax.imshow(np.abs(matrix.numpy(force=True))))

    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs)

    plt.xlabel("Features")
    plt.ylabel("Neurons")
    plt.savefig(fig_name, dpi=400, facecolor="white", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    model_1 = load_model("model_1.pth")
    model_1_proj = load_model("model_1_proj.pth")
    model_2 = load_model("model_2.pth")
    model_2_proj = load_model("model_2_proj.pth")

    show_matrix([model_1, model_2], fig_name="plots/2cohorts.png")
    show_matrix([model_1_proj, model_2_proj], fig_name="plots/2cohorts_proj.png")
