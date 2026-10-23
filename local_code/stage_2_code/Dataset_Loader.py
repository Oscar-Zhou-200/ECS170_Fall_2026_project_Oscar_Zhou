'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
import csv
import os
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')

        X = []
        y = []

        file_path = os.path.join(
            self.dataset_source_folder_path,
            self.dataset_source_file_name
        )

        with open(file_path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)

            for row in reader:
                if not row:
                    continue

                # Skip the header row, if present.
                try:
                    elements = [int(value) for value in row]
                except ValueError:
                    continue

                # The first column is the label.
                y.append(elements[0])

                # The remaining 784 columns are pixel features.
                X.append(elements[1:])

        print(f'Loaded {len(X)} samples.')
        return {'X': X, 'y': y}
