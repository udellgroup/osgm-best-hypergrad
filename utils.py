# Plotting utilities for visualizing the descent curves
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import itertools

def plot_dummy_lgd(algorithm_curves, 
                   xlabel="Iteration",
                   ylabel="Function Value",
                   title="Descent Curves",
                   use_log_scale=False,
                   figsize=(11, 6),
                   style="whitegrid",
                   legend_loc="lower right",
                   fname=None):
    
    # Create a SMALL figure that contains ONLY a legend
    fig = plt.figure(figsize=figsize)
    ax_legend = fig.add_subplot(111)
    ax_legend.axis("off")

    sns.set_style(style)
    
    # -- There's no need for a second "plt.figure(figsize=figsize)" if we just want one figure. --
    # plt.figure(figsize=figsize)  # <-- Typically, you'd remove or comment this out
    
    colors = sns.color_palette("Paired", n_colors=12)
    markers = itertools.cycle(["o", "s", "^", "D", "v", ">", "<", "p", "h", "*", "1", "2"])
    
    # Hack into the color of BFGS
    colors[6] = (0.0, 0.0, 0.0) # BFGS
    colors[7] = (0.75, 0.75, 0.75) # L-BFGS-M1
    colors[8] = (0.6, 0.6, 0.6) # L-BFGS-M3
    colors[9] = (0.45, 0.45, 0.45) # L-BFGS-M5
    colors[10] = (0.3, 0.3, 0.3) # L-BFGS-M10

    dummy_lines = []
    for (algo_name, values), color, marker in zip(algorithm_curves.items(), colors, markers):
        # Create dummy lines for legend
        line = plt.Line2D(
            [], [], 
            label=algo_name, 
            linewidth=6, 
            color=color, 
            marker=marker, 
            markersize=15, 
            markerfacecolor="white"
        )
        dummy_lines.append(line)
    
    # -- Arrange the legend entries horizontally by using ncol=num_algos --
    ax_legend.legend(
        handles=dummy_lines, 
        loc="center",       # Position it in the center of this blank axis
        fontsize=20, 
        ncol=6      # All legend entries in one horizontal row
    )

    if fname is not None:
        plt.savefig(fname, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_descent_curves(algorithm_curves, 
                        xlabel="Iteration",
                        ylabel="Function Value",
                        title="Descent Curves",
                        use_log_scale=False,
                        figsize=(8, 6),
                        style="whitegrid",
                        legend_loc="lower right",
                        fname=None):
    """
    Plots descent curves for multiple algorithms on a single plot.
    
    Parameters
    ----------
    algorithm_curves : dict
        A dictionary where keys are algorithm names (str) and values 
        are lists (or arrays) of function values across iterations.
    
    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    title : str, optional
        Title for the plot.

    use_log_scale : bool, optional
        If True, use a logarithmic scale for the y-axis.

    figsize : tuple, optional
        Figure size in inches, e.g., (width, height).

    style : str, optional
        Seaborn style to use. Options include: "white", "whitegrid", 
        "dark", "darkgrid", "ticks".

    legend_loc : str, optional
        Location of the legend, e.g., "best", "upper right", etc.

    fname : str or None, optional
        If provided, save the figure to this file path. Otherwise, display the plot.
    """
    sns.set_style(style)
    plt.figure(figsize=figsize)

    # Define a color palette and a list of markers
    # The palette will have as many distinct colors as there are curves
    num_algos = len(algorithm_curves)
    colors = sns.color_palette("Paired", n_colors=num_algos)
    
    if len(colors) > 10:
        # Hack into the color of BFGS
        colors[6] = (0.0, 0.0, 0.0) # BFGS
        colors[7] = (0.75, 0.75, 0.75) # L-BFGS-M1
        colors[8] = (0.6, 0.6, 0.6) # L-BFGS-M3
        colors[9] = (0.45, 0.45, 0.45) # L-BFGS-M5
        colors[10] = (0.3, 0.3, 0.3) # L-BFGS-M10
    
    # Cycle through markers if there are more algorithms than markers
    markers = itertools.cycle(["o", "s", "^", "D", "v", ">", "<", "p", "h", "*", "1", "2"])

    for (algo_name, values), color, marker in zip(algorithm_curves.items(), 
                                                  colors, 
                                                  markers):
        # Plot each curve with a distinct color and marker
        n_points = len(values)
        markevery = max(1, n_points // 4)
        plt.plot(values, 
                 label=algo_name, 
                 linewidth=6, 
                 color=color, 
                 marker=marker, 
                 markersize=15, 
                 markerfacecolor="white",
                 markevery=markevery)
        
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel, fontsize=30)
    
    if "a1a" in title or "BA" in title:
        plt.ylabel(ylabel, fontsize=30)
        
    plt.title(title, fontsize=32)

    if use_log_scale:
        plt.yscale("log")
        
    # Optionally set a lower y-limit if needed
    if ylabel == "Gradient Norm":
        plt.ylim(bottom=1e-05, top=1.5 * max([v[0] for v in algorithm_curves.values()]))
    else:
        plt.ylim(bottom=1e-10, top=1.5 * max([v[0] for v in algorithm_curves.values()]))
        
    if len(colors) < 10:
        plt.legend(loc=legend_loc, fontsize=16)
        
    plt.tight_layout()
    
    # Save or show the figure
    if fname is not None:
        plt.savefig(fname, dpi=300)
        plt.close()
    else:
        plt.show()

# Example usage:
if __name__ == "__main__":
    curves = {
        "Gradient Descent": [10, 8, 6, 4, 2, 1, 0.5],
        "Adam": [10, 7, 4, 2, 1.5, 1.0, 0.7],
        "RMSProp": [10, 9, 7, 5, 3, 2, 1]
    }
    plot_descent_curves(curves, use_log_scale=True)
