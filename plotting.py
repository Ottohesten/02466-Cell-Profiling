import matplotlib.pyplot as plt


def loss_plots(plot_data: dict, invidual_plots: bool = False):
    """
    Plots the losses from the dictionary
    """
    if invidual_plots:
        for key, value in plot_data.items():
            plt.plot(value, label=key)
            plt.title(key)
            plt.grid()
            plt.show()

    else:
        for key, value in plot_data.items():
            plt.plot(value, label=key)
        plt.legend()
        plt.show()