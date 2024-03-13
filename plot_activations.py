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
        # If the model was a projected model, it has a mask buffer
        # Note: the tensor here assumes a default FCNN
        model.register_buffer(name="mask", tensor=torch.zeros((4, 300, n_inputs)))
        model.load_state_dict(torch.load(path))
    return model


def show_matrix(models_list, fig_path):
    fig, axs = plt.subplots(nrows=1, ncols=len(models_list), sharex=True, sharey=True)
    fig.suptitle("First Layer Activations")

    images = []
    for i, ax in enumerate(axs.flat):
        # for each subplot, get the first layer of the model
        model = models_list[i]
        for param in model.parameters():
            matrix = param
            break
        # Plot the absolute value of the weight matrix
        images.append(ax.imshow(np.abs(matrix.numpy(force=True))))

    # Colorbar unification and normalization
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs)

    plt.xlabel("Features")
    plt.ylabel("Neurons")
    plt.savefig(fig_path, dpi=400, facecolor="white", bbox_inches="tight")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    model_1 = load_model("model_1.pth")
    model_1_proj = load_model("model_1_proj.pth")
    model_2 = load_model("model_2.pth")
    model_2_proj = load_model("model_2_proj.pth")

    show_matrix([model_1, model_2], fig_path="plots/2cohorts.png")
    show_matrix([model_1_proj, model_2_proj], fig_path="plots/2cohorts_proj.png")
