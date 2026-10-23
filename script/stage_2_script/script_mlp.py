
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy


# ---- MLP script for Stage 2 ----

if __name__ == '__main__':

    # Reproducibility
    np.random.seed(2)
    torch.manual_seed(2)

    # Locate the project root directory.
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )

    # Initialize dataset loader.
    data_obj = Dataset_Loader('handwritten digits', '')
    data_obj.dataset_source_folder_path = os.path.join(
        project_root, 'data', 'stage_2_data'
    )

    # The Setting class will load train.csv and test.csv separately.
    data_obj.dataset_source_file_name = 'train.csv'

    # Initialize MLP.
    method_obj = Method_MLP('multi-layer perceptron', '')

    # Initialize result saver.
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = os.path.join(
        project_root, 'result', 'stage_2_result'
    ) + os.sep
    result_obj.result_destination_file_name = 'mlp_prediction_result'

    # Use the provided train/test split.
    setting_obj = Setting_Train_Test_Split(
        'provided train/test split', ''
    )

    # Initialize evaluation.
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    # Run experiment.
    print('************ Start Stage 2 ************')

    setting_obj.prepare(
        data_obj,
        method_obj,
        result_obj,
        evaluate_obj
    )

    setting_obj.print_setup_summary()

    metrics, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')

    # Plot training convergence curves.
    epochs = range(1, len(method_obj.loss_history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, method_obj.loss_history, marker='o')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].grid(True)

    axes[1].plot(epochs, method_obj.accuracy_history, marker='o')
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)

    fig.tight_layout()

    result_folder = os.path.join(
        project_root, 'result', 'stage_2_result'
    )
    os.makedirs(result_folder, exist_ok=True)

    figure_path = os.path.join(
        result_folder, 'convergence_curves.png'
    )
    fig.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    print(f'Convergence curves saved to: {figure_path}')

    print('************ Finish Stage 2 ************')
