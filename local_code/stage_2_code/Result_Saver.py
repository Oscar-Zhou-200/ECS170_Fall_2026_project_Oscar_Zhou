
"""
Result saver for Stage 2 experiments.
"""

import os
import pickle

from local_code.base_class.result import result


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')

        # Create the output directory if it does not exist.
        os.makedirs(
            self.result_destination_folder_path,
            exist_ok=True
        )

        # Use a clear filename for the Stage 2 result.
        output_path = os.path.join(
            self.result_destination_folder_path,
            self.result_destination_file_name + '.pkl'
        )

        with open(output_path, 'wb') as f:
            pickle.dump(self.data, f)

        print(f'Results saved to: {output_path}')
