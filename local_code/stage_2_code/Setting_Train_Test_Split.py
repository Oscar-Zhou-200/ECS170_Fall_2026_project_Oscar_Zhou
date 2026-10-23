'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import os
from local_code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np

class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):
        # Locate the Stage 2 dataset folder.
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        dataset_folder = os.path.join(
            project_root, 'data', 'stage_2_data'
        )

        # Load the training set.
        self.dataset.dataset_source_folder_path = dataset_folder
        self.dataset.dataset_source_file_name = 'train.csv'
        train_data = self.dataset.load()

        # Load the testing set.
        self.dataset.dataset_source_file_name = 'test.csv'
        test_data = self.dataset.load()

        # Prepare data for the method.
        self.method.data = {
            'train': {
                'X': train_data['X'],
                'y': train_data['y']
            },
            'test': {
                'X': test_data['X'],
                'y': test_data['y']
            }
        }

        # Run the model.
        learned_result = self.method.run()

        # Save raw results.
        self.result.data = learned_result
        self.result.save()

        # Evaluate results.
        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None

        