from datetime import timedelta

import numpy as np
import pandas as pd
from keras.src.callbacks import History
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.font_manager as fm

from utilities import ensure_dir


def train_test_duration_display(training_duration: timedelta, subsecond_precision: int=3) -> str:
    """
    Format a python timedelta as a string with the format HH:MM:SS:ffffff where ffffff is subseconds
    shown to <subsecond_precision> decimal places.
    Modified code based on Claude Code (2026)
    """

    # guard subsecond_precision out of range
    if subsecond_precision < 0 or subsecond_precision > 6:
        raise ValueError("subsecond_precision must be between 0 and 6")

    total_seconds = int(training_duration.total_seconds())
    days, remainder = divmod(total_seconds, 86400)
    hours: int = days * 24 + remainder // 3600  # hours can exceed 23 for long durations
    minutes: int = (remainder % 3600) // 60
    seconds: int = remainder % 60
    microseconds: int = training_duration.microseconds
    subseconds: float = microseconds / (10 ** (6 - subsecond_precision)) # e.g. microseconds to milliseconds if subsecond_precision is 3
    subseconds_int: int = int(round(subseconds, 0)) # round subseconds to <subsecond_precision> decimal places

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{subseconds_int:02d}" # pad each time place with leading zeros


def save_confusion_matrix_png(y_true, y_pred, path, serialized_cm_path: str|None=None, labelled_cm_path: str|None=None, cm_title: str|None=None, details: dict|None=None, data_type=int):
    import sklearn
    print(sklearn.__version__)
    print(sklearn.metrics.confusion_matrix.__module__)

    import inspect
    sig = inspect.signature(sklearn.metrics.confusion_matrix)
    print(sig.parameters.keys())
    
    if not path:
        return
    ensure_dir(path)
    cm = confusion_matrix(y_true, y_pred)
    # cm = confusion_matrix(y_true, y_pred, sample_weight=np.ones_like(y_true, dtype=float))

    if cm_title is None:
        cm_title = "Confusion Matrix"
    
    print(f"{cm_title}:\n{cm}")

    # save confusion matrix to disk
    if serialized_cm_path:
        ensure_dir(serialized_cm_path)
        np.save(serialized_cm_path, cm)
    
    set_global_matplotlib_font()
    caption_font_size = 10

    if details is not None:
        caption = create_caption_from_details(details)
    else:
        dummy_details = {
            "DUMMY": "DUMMY",
            "subject": "Subject_3",
            "image_dimensions": "224x224",
            "patches_per_dim": "16",
            "epochs": "200",
            "batch_size": "256",
            "learning_rate": "0.0001",
            "val_split": "0.1",
            "training_set_size": "4320",
            "training_duration": "08:08:40.274",
            "max_val_acc": "0.9875",
            "max_val_acc_epoch": "13",
            "testing_set_size": "1200",
            "testing-duration": "00:00:20.846",
            "processor_type": "Apple M4 Max",
            "cpu_cores": "16",
            "gpu_cores": "40",
            "test_accuracy": "0.9733",
        }
        # caption = ""
        caption = create_caption_from_details(dummy_details)

    # print scikit-learn confusion matrix
    print(f"printing scikit-learn confusion_matrix")
    sklearn_cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[x for x in range(12)])
    # fig_cm, ax_cm = plt.subplots(figsize=(8,12), layout='constrained')
    fig_cm, ax_cm = plt.subplots(figsize=(8,12))
    sklearn_cm_display.plot(ax=ax_cm, cmap="viridis", colorbar=False)
    # plt.figtext(x=0.5, y=-0.75, s=caption, wrap=True, horizontalalignment='center', fontsize=caption_font_size)
    # fig1.text(x=0.5, y=-0.2, s=caption, ha='center', fontsize=caption_font_size)
    divider = make_axes_locatable(ax_cm)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig_cm.colorbar(sklearn_cm_display.im_, cax=cax)

    ax_cm.set_title(label=cm_title, loc='center', pad=10)
    fig_cm.text(x=0.5, y=0.25, s=caption, ha='center', va='top', fontsize=caption_font_size)
    fig_cm.tight_layout(rect=( 0, 0.25, 1, 1 ))
    # plt.tight_layout()

    # cax2 = divider.append_axes("bottom", size="5%", pad=0.2)
    # fig_cm.set_label("Confusion Matrix")
    # cax2.set_label(f"\n{caption}")
    
    # sklearn_cm_display.figure_.text(0.5, -0.2, caption, ha='center', fontsize=caption_font_size)
    # fig_cm.text(0.5, -0.2, caption, ha='center', fontsize=caption_font_size)
    # sklearn_cm_display.figure_.set_figheight(14)
    # sklearn_cm_display.figure_.set_figwidth(8)
    # sklearn_cm_display.figure_.tight_layout()
    # plt.tight_layout()
    # if cm_title:
    #     plt.title(cm_title)
    print(f"saving cm plot to {labelled_cm_path}")
    if labelled_cm_path:
        fig_cm.savefig(labelled_cm_path, bbox_inches='tight')
    # plt.show(bbox_inches='tight')
    plt.show()

    # print matplotlib confusion matrix
    if False:
        fig, ax = plt.subplots(figsize=(10, 18))
        im = ax.imshow(cm, interpolation="nearest", cmap="inferno")
        ax.set_title(cm_title)
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(f"Predicted\n\n{caption}", fontdict={'size': caption_font_size})
        ax.set_ylabel("True")
        # set the class labels on the x and y axes explicitly
        ax.set_xticks(np.arange(len(np.unique(y_pred))))
        ax.set_yticks(np.arange(len(np.unique(y_true))))
        fig.tight_layout()
        plt.show()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def create_caption_from_details(details: dict) -> str:
    key_val_separator: str = ": "
    fill_char: str = '.'

    caption: str = ""
    max_detail_width: int = 0

    # build caption text from details dictionary
    for key, value in details.items():
        detail_width: int = len(key) + len(key_val_separator) + len(str(value))
        if detail_width > max_detail_width:
            max_detail_width = detail_width
        caption += f"{key}{key_val_separator}{value}\n"
    # remove last separator
    caption = caption.rstrip("\n")
    # add padding to caption lines
    for line in caption.split("\n"):
        # option 1: pad right (left justify)
        # caption = caption.replace(line, line.ljust(max_detail_width, fill_char))

        # option 2: pad between key and value
        key: str = line.split(key_val_separator)[0]
        val: str = line.split(key_val_separator)[1]
        fill_width: int = max_detail_width - len(line)
        new_line: str = f"{key}{key_val_separator}{fill_char * fill_width}{val}"
        caption = caption.replace(line, new_line)
    return caption


def set_global_matplotlib_font(font_family_cascade: list[str]|None=None, default_font_type: list[str]|None=None) -> None:
    # all fonts in font_family_cascade must be system fonts detectable by matplotlib using fm.fontManager.ttflist
    # default_font_type is one of ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
    if font_family_cascade is None:
        font_family_cascade = ['Inconsolata', 'Andale Mono']

    font_type_fallback = default_font_type if default_font_type is not None else ['monospace']

    # check available fonts
    font_names = sorted({f.name for f in fm.fontManager.ttflist})
    # look for the preferred fonts in the found system fonts
    preferred_font_found = False
    selected_font: str|None = None
    for preferred_font in font_family_cascade:
        if preferred_font in font_names:
            preferred_font_found = True
            selected_font = preferred_font
            break
    if not preferred_font_found:
        selected_font = font_type_fallback[0]

    # return selected_font
    # set the global matplotlib font
    plt.rcParams['font.family'] = selected_font
    

def set_global_matplotlib_fontsize(font_size: int=12):
    plt.rcParams['font.size'] = font_size


def plot_history_separately(training_history: History, loss_plot_title: str|None=None, acc_plot_title: str|None=None, details: dict|None=None, save_plots: bool=False, plot_filename: str|None=None, val_acc_key: str|None=None, train_acc_key: str|None=None):
    """
    Creates side-by-side subplots to display the loss and accuracy history over
    epochs for both training and validation datasets. Training details and plot titles can be added.
    Plots can be saved or displayed, depending on the value of `save_plots`.

    Args:
        training_history (History): The training history object obtained from model training.
        loss_plot_title (str | None): Optional custom title for the loss plot. Defaults to 
            'Model Loss Over Epochs' if None.
        acc_plot_title (str | None): Optional custom title for the accuracy plot. Defaults
            to 'Model Accuracy Over Epochs' if None.
        details (dict | None): Optional dictionary containing additional details about the
            training run to be added as a caption to the plots.
        save_plots (bool): If True, saves the generated plots to a file. If False, displays the plots.
        plot_filename (str | None): Filename for saving the plots. If None and save_plots 
            is True, a default filename is used. Ignored if save_plots is False.
        val_acc_key (str | None): Optional key for the validation accuracy metric in the training history. 
        train_acc_key (str | None): Optional key for the training accuracy metric in the training history. 

    Raises:
        KeyError: Raised if expected keys like 'loss', 'val_loss', 'accuracy', or 'val_accuracy'
            are missing from the training history.

    Returns:
        None: This function does not return any value and either shows or saves the plots.
    """

    set_global_matplotlib_font()
    caption_font_size: int = 8

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # create plot caption
    if details is not None:
        caption = create_caption_from_details(details)
    else:
        caption = ""

    # Plot Training & Validation Loss
    # for ViT with attention output:
    # history.history.keys(): dict_keys(['loss', 'dense_19
    # _loss', 'dense_19_accuracy', 'val_loss', 'val_dense_19_loss', 'val_dense_19_accuracy'])
    loss_title: str = loss_plot_title if loss_plot_title is not None else 'Model Loss Over Epochs'

    ax1.plot(training_history.history['loss'], label='Train Loss', color='blue', linewidth=1)
    if 'val_loss' in training_history.history:
        ax1.plot(training_history.history['val_loss'], label='Val Loss', color='orange', linestyle='-', linewidth=1.5)
    ax1.set_title(loss_title)
    # adding the caption to the X label is the simplest way to display it
    ax1.set_xlabel(f'Epochs\n\n{caption}', fontdict={'size': caption_font_size})
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot Training & Validation Accuracy
    acc_title: str = acc_plot_title if acc_plot_title is not None else 'Model Accuracy Over Epochs'

    # Note: Use 'acc' instead of 'accuracy' if you are using an older Keras version
    # accuracy key will be sent in the arguments
    # acc_key = 'accuracy' if 'accuracy' in training_history.history else 'acc'
    if train_acc_key is not None:
        training_acc_key = train_acc_key
    elif 'accuracy' in training_history.history:
        training_acc_key = 'accuracy'
    else:
        training_acc_key = 'acc'
    
    print(f"using training_acc_key '{training_acc_key}'")
    
    if training_acc_key in training_history.history:
        ax2.plot(training_history.history[training_acc_key], label='Train Accuracy', color='blue', linewidth=1)
    
    if val_acc_key is not None:
        validation_acc_key = val_acc_key
    elif 'accuracy' in training_history.history:
        validation_acc_key = 'val_accuracy'
    else:
        validation_acc_key = 'val_acc'
    
    print(f"using vaidation_acc_key '{validation_acc_key}'")
    
    if validation_acc_key in training_history.history:
        ax2.plot(training_history.history[validation_acc_key], label='Val Accuracy', color='orange', linestyle='-', linewidth=1.5)
    ax2.set_title(acc_title)
    # ax2.set_xlabel(f'Epochs\n\n{caption}')
    ax2.set_xlabel(f'Epochs\n\n{caption}', fontdict={'size': caption_font_size})
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # add caption with training run details
    # NOTE: this method requires modifying the axis locations so that the axes are smaller than
    # the figure size. Easier to add the caption to the X label, unless adding the caption inside the
    # bounds of the axes, as an overlay on the plot itself.
    # plt.figtext(x=0.1, y=0.75, s=caption, wrap=True, horizontalalignment='left', fontsize=10)

    plt.tight_layout()

    if save_plots:
        if plot_filename is None:
            plot_filename = f"?_?_epochs_?_lr_?"
        print(f"Saving plot to {plot_filename}")
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_history_together(training_history: History):
    # Convert history dictionary to DataFrame
    history_df = pd.DataFrame(training_history.history)

    # Plot all metrics at once
    history_df.plot(figsize=(10, 6))
    plt.grid(True)
    # plt.gca().set_ylim(0, 1) # Optional: clamp y-axis between 0 and 1 for accuracy
    plt.xlabel("Epochs")
    plt.show()
    
