#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import os
import numpy as np

# My libs
from utils.config import Config
from utils.tester import ModelTester
from models.KPCNN_model import KernelPointCNN
from models.KPFCNN_model import KernelPointFCNN

# Datasets
from datasets.Scannet import ScannetDataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def test_caller(path, step_ind, on_val):

    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    # Load model parameters
    config = Config()
    config.load(path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_color = 1.0
    config.validation_size = 500
    #config.batch_num = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    if config.dataset == 'Scannet':
        dataset = ScannetDataset(config.input_threads, load_test=(not on_val))
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Create subsample clouds of the models
    dl0 = config.first_subsampling_dl
    dataset.load_subsampled_clouds(dl0)

    # Initialize input pipelines
    if on_val:
        dataset.init_input_pipeline(config)
    else:
        dataset.init_test_input_pipeline(config)


    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    if config.dataset.startswith('Scannet'):
        model = KernelPointFCNN(dataset.flat_inputs, config)
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    chosen_step = np.sort(snap_steps)[step_ind]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    # Create a tester class
    tester = ModelTester(model, restore_snap=chosen_snap)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ############
    # Start test
    ############

    print('Start Test')
    print('**********\n')

    if config.dataset.startswith('Scannet'):
        if on_val:
            tester.test_cloud_segmentation_on_val(model, dataset)
        else:
            tester.test_cloud_segmentation(model, dataset)
    else:
        raise ValueError('Unsupported dataset')


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ##########################
    # Choose the model to test
    ##########################

    #
    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_Scannet': Automatically retrieve the last trained model on Scannet
    #
    #       > 'results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model
    #

    chosen_log = 'results/Log_2022-03-18_16-01-18'

    #
    #   You can also choose the index of the snapshot to load (last by default)
    #

    chosen_snapshot = -1

    #
    #   Eventually, you can choose to test your model on the validation set
    #

    on_val = True

    #
    #   If you want to modify certain parameters in the Config class, for example, to stop augmenting the input data,
    #   there is a section for it in the function "test_caller" defined above.
    #

    ###########################
    # Call the test initializer
    ###########################

    handled_logs = ['last_Scannet',]

    # Automatically retrieve the last trained model
    if chosen_log in handled_logs:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in handled_logs:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    # Let's go
    test_caller(chosen_log, chosen_snapshot, on_val)


