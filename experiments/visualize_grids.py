import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_grid(grid, ax, title=""):
    cmap = ['#252525', '#0074D9','#FF4136','#37D449', '#FFDC00','#E6E6E6', '#F012BE','#FF871E', '#54D2EB',  '#8D1D2C', '#FFFFFF']
    grid_array = np.array(grid)
    ax.imshow(grid_array, cmap=mcolors.ListedColormap(cmap), vmin=0, vmax=len(cmap)-1)
    ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    ax.set_title(title)
    ax.axis('off')

def plot_task(task, task_id, show_or_save="show"):
    num_train_items = len(task['train'])
    num_test_items = len(task['test'])
    
    # Calculate the number of rows needed
    num_train_rows = (num_train_items + 1) // 2  # Two train pairs per row
    num_test_rows = num_test_items  # One test pair per row
    total_rows = num_train_rows + num_test_rows

    fig, axes = plt.subplots(total_rows, 4, figsize=(14, 3*total_rows))
    # Keep track of used axes
    used_axes = set()
    
    for idx, item in enumerate(task['train']):
        row = idx // 2
        col = (idx % 2) * 2
        plot_grid(item['input'], axes[row, col], title=f"Train Input {idx+1}")
        plot_grid(item['output'], axes[row, col + 1], title=f"Train Output {idx+1}")
        used_axes.update([(row, col), (row, col + 1)])

    for idx, item in enumerate(task['test']):
        row = num_train_rows + idx
        plot_grid(item['input'], axes[row, 0], title=f"Test Input {idx+1}")
        used_axes.add((row, 0))
        for col in range(1, 4):
            axes[row, col].axis('off')

    # Hide unused axes
    for row in range(total_rows):
        for col in range(4):
            if (row, col) not in used_axes:
                axes[row, col].axis('off')
    
    plt.suptitle(f"Task {task_id}")
    plt.tight_layout()

    if show_or_save == "show":
        plt.show()
    else:
        plt.savefig(f"task_{task_id}.png")
