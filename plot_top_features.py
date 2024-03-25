import os
import pandas as pd
import matplotlib.pyplot as plt

N_COHORTS = 2
model_name = "FCNN"
# model_name = "netBio"

results_dir = "results/"

plots_dir = "plots/"  # where the plot will be saved

data_path = "data/Synth_300f_20inf_10000s.csv"
# data_path = "data/GC_Breast_D_MB.csv"
# data_path = "data/LUNG.csv"

N_FEATURES_TO_PLOT = 50  # Where to cut off the plot
SORT_BY_NO_PROJ = False  # Whether to sort the features with the order from cohort 0, no proj or from cohort 0, proj


if __name__ == "__main__":
    # Create subdirectory for this data file
    plots_dir += data_path[5:-4] + "/"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    X = []
    plt.figure(figsize=(12, 8))
    for cohort_i in range(N_COHORTS):
        # Read the feature ranking dataframes
        features_df = pd.read_csv(
            results_dir
            + data_path[5:-4]
            + f"/features_{model_name}_cohort{cohort_i}.csv",
            index_col=0,
        )
        features_df_proj = pd.read_csv(
            results_dir
            + data_path[5:-4]
            + f"/features_{model_name}_cohort{cohort_i}_proj.csv",
            index_col=0,
        )

        if cohort_i == 0:
            # Choose a run to sort the features by
            # Either Cohort 0 first descent or
            # Cohort 0 second descent
            if SORT_BY_NO_PROJ:
                X = features_df.index.to_list()
            else:
                X = features_df_proj.index.to_list()

        # Note: the .reindex(X) ensures consistent order of the features plotted
        plt.plot(
            X, features_df["Mean"].reindex(X), "x-", label=f"Cohort {cohort_i}, No proj"
        )
        plt.plot(
            X,
            features_df_proj["Mean"].reindex(X),
            "x-",
            label=f"Cohort {cohort_i}, Proj",
        )

    # add the figure names on the X axis
    plt.xticks(ticks=[i for i in range(len(X))], labels=X, rotation=90)
    plt.xlim(left=-0.5, right=N_FEATURES_TO_PLOT)
    plt.legend()
    plt.ylabel("Feature Importance")
    plt.title(
        f"[{data_path[5:-4]}] Feature Importances ranked in the descending order for Cohort 0, {'No '*SORT_BY_NO_PROJ}proj"
    )
    plt.savefig(
        plots_dir + f"{model_name}_{N_COHORTS}c.png",
        facecolor="white",
        dpi=500,
        bbox_inches="tight",
    )
    plt.show()
