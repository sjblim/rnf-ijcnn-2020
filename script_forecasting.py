"""
script_forecasting.py

Evaluates normalised MSE of the RNF fo the power dataset.

Details
--------


Created by limsi on 24/03/2020
"""

import argparse
import os
import tensorflow as tf
import numpy as np

import configs
from libs.hyperparam_opt import HyperparamOptManager
from models import model_factory
from libs.utils import recreate_folder
from data_formatters.batchers import EfficientRnfBatcher, FullRnfBatcher

K = tf.keras.backend


if __name__ == "__main__":


    ########################################
    # User-defined settings
    model_name = 'rnf'  # 'lstm'
    expt_name = 'power'
    use_gpu = True
    mse_normaliser = 0.0439292598  # DeepAR MSE obtained in tests
    use_fixed_model = True         # Model from script_hyperparam_opt.py or script_train_fixed_params.py
    run_multistep = True           # Whether to run one-step-ahead or multistep forecasts
    ########################################

    print()
    print("# -----------------------------------------------------------")
    print("# Commencing forecasts ")
    print("# -----------------------------------------------------------")
    print()

    print("*** Setting devices ***")
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs={}".format(len(gpu_devices)))

    if use_gpu:
        print("Defaulting to {}".format(gpu_devices[0]))
        tf.config.set_visible_devices(gpu_devices[0], 'GPU')
    else:
        print("Using CPU only")
        my_devices = tf.config.list_physical_devices(device_type='CPU')
        tf.config.set_visible_devices(devices=my_devices, device_type='CPU')

    print("*** Creating datasets ***")
    df = configs.load_dataset(expt_name)
    data_formatter = configs.get_dataformatter(expt_name, model_name)

    print("train-valid-test splits")
    train, valid, test = data_formatter.split_data(df)

    print("*** Initialising hyperparam manager ***")
    param_ranges = configs.get_default_hyperparams(model_name)
    fixed_params = data_formatter.get_experiment_params()


    if use_fixed_model:
        model_folder = os.path.join(configs.MODEL_PATH, expt_name, model_name, 'fixed')
    else:
        model_folder = os.path.join(configs.MODEL_PATH, expt_name, model_name)
    opt_manager = HyperparamOptManager(param_ranges, fixed_params, model_folder)

    success = opt_manager.load_results()
    if not success:
        raise ValueError("Hyper-param manager cannot be loaded...")

    def one_step_forecast(test_data, opt_manager, model_name="rnf"):

        """
        N.b. Given our main use case is for real-time data streams where speed is key,
            we test the RNF by running it continuously (i.e. states are not reset during operation)
        """

        print("*** Running one-step-ahead predictions ***")

        # Get hyperparam stuff
        params = opt_manager.get_best_params()
        checkpoint_path = opt_manager.checkpoint_path

        # Param setup
        params['total_time_steps'] = int(1e4)
        params['minibatch_size'] = 1
        output_size = int(params['output_size'])
        hidden_layer_size = int(params['hidden_layer_size'])

        # Setup data
        batcher = configs.get_batcher(model_name)
        col_defn = data_formatter.get_column_definition()

        test_batches = batcher.batch(test_data, col_defn, lookback=params['total_time_steps'])

        test_inputs, test_outputs, test_flags = test_batches

        # To make both LSTM & RNF outputs consistent
        if not isinstance(test_inputs, list):
            test_inputs = [test_inputs]

        batches, timesteps, _= test_outputs.shape

        # Set model
        K.clear_session()

        states = [np.zeros((1, hidden_layer_size)) for _ in range(2)]
        model = model_factory.make_model(model_name, params, set_initial_states=True)
        _ = model.predict([ip[:1, ...] for ip in test_inputs]+states)
        model.load_weights(checkpoint_path)

        preds = []

        for i in range(batches):
            print("Predicting {}/{} trajs".format(i + 1, batches))
            outputs, state_h, state_c = model.predict([ip[i:i + 1, ...] for ip in test_inputs]+states)

            states = [state_h, state_c]
            means, stds = outputs[..., :output_size], outputs[..., output_size:]

            preds.append(means)

        preds = np.concatenate(preds, axis=0)

        mse = np.sum((preds-test_outputs)**2 * test_flags[..., np.newaxis]) / np.sum(test_flags[..., np.newaxis])

        return mse

    def multi_step_forecast(test_data,
                            opt_manager,
                            horizon=20,
                            inputs_unknown=True):

        print("*** Running multistep predictions ***")

        # Get hyperparam stuff
        params = opt_manager.get_best_params()
        checkpoint_path = opt_manager.checkpoint_path

        # -----------------------------------
        # Generating internal states at each time step
        print("Generating states...")

        # Param setup
        params['total_time_steps'] = int(1e4)
        params['minibatch_size'] = 1
        hidden_layer_size = int(params['hidden_layer_size'])

        # Setup data
        batcher = EfficientRnfBatcher
        col_defn = data_formatter.get_column_definition()

        test_batches = batcher.batch(test_data, col_defn, lookback=params['total_time_steps'])

        test_inputs, test_outputs, test_flags = test_batches

        # Set model
        states = [np.zeros((1, hidden_layer_size)) for _ in range(2)]
        K.clear_session()
        model = model_factory.make_rnf(params, set_initial_states=True, dump_states=True)  # use in filter mode
        _ = model.predict([ip[:1, ...] for ip in test_inputs]+states)
        model.load_weights(checkpoint_path)

        hs = []
        cs = []
        for i in range(test_inputs[0].shape[0]):
            print("Generating states for traj {}/{}".format(i+1, test_inputs[0].shape[0]))
            outputs= model.predict([ip[i:i+1, ...] for ip in test_inputs]+states)

            state_h, state_c, h, c = tuple(outputs)

            # For next iteration
            states = [h, c]

            if not hs:
                # Set first initial states to 0
                hs.append(np.zeros_like(state_h[:, :1, :]))
                cs.append(np.zeros_like(state_c[:, :1, :]))

            hs.append(state_h)
            cs.append(state_c)

        hs = np.concatenate(hs, axis=1)
        cs = np.concatenate(cs, axis=1)

        # -----------------------------------
        # Multi-step forecasting by projecting states with RNF in "missing data" mode
        print("Generating multistep forecasts....")

        # Param setup
        params['total_time_steps'] = horizon
        params['minibatch_size'] = 1024

        # Data Setup
        batcher = FullRnfBatcher
        col_defn = data_formatter.get_column_definition()

        test_batches = batcher.batch(test_data, col_defn, lookback=horizon)
        rnf_inputs, targets, active_entries = test_batches

        inputs, flags = rnf_inputs

        # Set flags depending on forecast type
        if inputs_unknown:
            # Switch off both inputs and targets
            inputs *= 0
            flags *= 0
        else:
            # Remove target input to ensure its not being used
            inputs[..., -1] *= 0
            # Set only targets to unknown
            flags[..., -1] *= 0

        total_batches = min(inputs.shape[0], hs[0].shape[0])
        initial_states = [hs[0, :total_batches, :], cs[0, :total_batches, :]]
        rnf_inputs = [inputs[:total_batches,...], flags[:total_batches,...]]
        targets, active_entries = targets[:total_batches,...], active_entries[:total_batches,...]

        # Model loading
        K.clear_session()
        model = model_factory.make_rnf(params, set_initial_states=True)
        _ = model.predict([ip[:1, ...] for ip in rnf_inputs+initial_states])
        model.load_weights(checkpoint_path)

        # Forecasting proper
        print("Predicting...")
        outputs, _, _ = model.predict(rnf_inputs+initial_states,
                                batch_size=params['minibatch_size'],
                                workers=params['multiprocessing_workers'],
                                use_multiprocessing=True)

        output_size = targets.shape[-1]
        means, stds = outputs[..., :output_size], outputs[..., output_size:]
        print('Evaluating forecasts...')
        mse = np.sum(((means - targets)**2 * active_entries[..., np.newaxis]), axis=(0,2)) \
                    / np.sum(active_entries[..., np.newaxis], axis=(0, 2))

        return mse

    # Functions to run.
    if run_multistep:
        multistep_mse = multi_step_forecast(test, opt_manager, horizon=20, inputs_unknown=True)

        # Average MSE over various horizons
        multistep_mse = np.cumsum(multistep_mse) / np.cumsum(np.ones_like(multistep_mse))

        print("Multistep MSEs:")
        print(multistep_mse)
        print()
        print("Multistep Normalised MSEs")
        print(multistep_mse / mse_normaliser)
        print()
    else:
        one_step_mse = one_step_forecast(test, opt_manager, model_name=model_name)
        print()
        print("One-step-ahead MSE={}, Norm MSE={}".format(one_step_mse, one_step_mse/mse_normaliser))
        print()





