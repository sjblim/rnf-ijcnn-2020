"""
script_hyperparam_opt.py


Created by limsi on 23/03/2020
"""

import argparse
import datetime as dte
import os
import tensorflow as tf
import numpy as np

import configs
from libs.hyperparam_opt import HyperparamOptManager
from models import model_factory

from libs.utils import recreate_folder

K = tf.keras.backend

if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Data download configs")

        parser.add_argument(
                "model_name",
                metavar="m",
                type=str,
                nargs="?",
                default="rnf",
                help="Model Name")

        parser.add_argument(
                "expt_name",
                metavar="e",
                type=str,
                nargs="?",
                default="power",
                choices=configs.VALID_EXPT_NAMES,
                help="Experiment Name. Default={}".format(",".join(configs.VALID_EXPT_NAMES)))

        parser.add_argument(
                "use_gpu",
                metavar="g",
                type=str,
                nargs="?",
                choices=["yes", "no"],
                default="yes",
                help="Whether to use gpu for training.")

        parser.add_argument(
                "restart_hyperparam_opt",
                metavar="o",
                type=str,
                nargs="?",
                choices=["yes", "no"],
                default="yes",
                help="Whether to re-run hyperparameter optimisation from scratch.")

        args = parser.parse_known_args()[0]

        return args.model_name, args.expt_name, args.use_gpu == "yes", args.restart_hyperparam_opt


    model_name, expt_name, use_gpu, restart_opt = get_args()

    ######################################
    # Defaults
    hyperparam_iterations = 50
    early_stopping = 1000  # disables early stopping here
    ######################################

    print()
    print("# -----------------------------------------------------------")
    print("# Commencing hyperparameter optimisation")
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

    model_folder = os.path.join(configs.MODEL_PATH, expt_name, model_name)
    opt_manager = HyperparamOptManager(param_ranges, fixed_params, model_folder)

    success = opt_manager.load_results()
    if success and not restart_opt:
        print("Loaded results from previous training")
    else:
        print("Creating new hyperparameter optimisation")
    opt_manager.clear()

    print("*** Commencing hyperparameter optimisation ***")
    print('Batching data')
    batcher = configs.get_batcher(model_name)
    col_defn = data_formatter.get_column_definition()
    time_steps = fixed_params['total_time_steps']
    train_batches = batcher.batch(train, col_defn, time_steps)
    valid_batches = batcher.batch(valid, col_defn, time_steps)
    test_batches = batcher.batch(test, col_defn, time_steps)

    # unpack
    train_inputs, train_outputs, train_flags = train_batches
    valid_inputs, valid_outputs, valid_flags = valid_batches
    test_inputs, test_outputs, test_flags = test_batches

    while len(opt_manager.results.columns) < hyperparam_iterations:

        K.clear_session()

        print()
        print("# -----------------------------------------------------------")
        print("# Running hyperparam optimisation {} of {} for {}".format(
            len(opt_manager.results.columns) + 1, hyperparam_iterations, expt_name))
        print("# -----------------------------------------------------------")
        print()

        params = opt_manager.get_next_parameters()
        model = model_factory.make_model(model_name, params)

        tmp_folder = os.path.join(model_folder, 'tmp')
        tmp_model = os.path.join(tmp_folder, "model")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping,
                min_delta=1e-4),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=tmp_model,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        model.fit(
            x=train_inputs,
            y=train_outputs,
            sample_weight=train_flags,
            epochs=params['num_epochs'],
            batch_size=params['minibatch_size'],
            validation_data=(valid_inputs, valid_outputs, valid_flags),
            callbacks=callbacks,
            shuffle=True,
            use_multiprocessing=True,
            workers=params['multiprocessing_workers'])


        try:
            model.load_weights(tmp_model)
            val_loss = model.evaluate(valid_inputs, valid_outputs, sample_weight=valid_flags)
            recreate_folder(tmp_folder) # ensures that it is clear for next run

        except:
            print('Cannot load model: {}'.format(tmp_model))
            val_loss = np.nan


        if np.allclose(val_loss, 0.) or np.isnan(val_loss):
            # Set all invalid losses to infinity.
            # N.b. val_loss only becomes 0. when the weights are nan.
            print("Skipping bad configuration....")
            val_loss = np.inf

        opt_manager.update_score(params, val_loss, model)

    print("*** Loading final model ***")
    K.clear_session()
    best_params = opt_manager.get_best_params()
    best_valid_score = opt_manager.best_score

    # Model loading
    model = model_factory.make_model(model_name, best_params)
    # required to initialise some states to load model
    basic_inputs, basic_outputs, _ = batcher.batch(train.iloc[:time_steps*2, :], # use small datasets
                                                   col_defn, time_steps)
    model.train_on_batch(basic_inputs, basic_outputs)

    model.load_weights(opt_manager.checkpoint_path)

    # Check to make sure weights are correctly loaded
    val_loss = model.evaluate(valid_inputs, valid_outputs, sample_weight=valid_flags)
    np.testing.assert_allclose(val_loss, best_valid_score, rtol=1e-6, atol=1e-6)

    print("Hyperparam optimisation completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    print()

    # Out of sample testing
    print("*** Computing Raw MSEs ***")

    def calc_mse(valid_inputs, valid_outputs, valid_flags):
        preds = model.predict(valid_inputs)
        output_size = valid_outputs.shape[-1]
        mse = np.mean((preds[..., :output_size] - valid_outputs) ** 2 * valid_flags[..., np.newaxis]) \
              / np.mean(valid_flags[..., np.newaxis])
        return mse

    print("Valid MSE={}".format(calc_mse(valid_inputs, valid_outputs, valid_flags)))
    print("Test MSE={}".format(calc_mse(test_inputs, test_outputs, test_flags)))  # This tests in encoder/decoder mode




