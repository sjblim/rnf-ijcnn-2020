"""
batchers.py

Provides custom batching functions for various models


Created by limsi on 23/03/2020
"""

import numpy as np
from data_formatters.base import InputTypes

class EfficientAutoregressiveBatcher(object):

    """
    Efficient batching that ensures each time step is only included once. Allows for easy reproducibility of results.

    """

    @classmethod
    def batch(cls, df, col_defn, lookback):
        target_cols = [defn[0] for defn in col_defn if defn[-1] == InputTypes.TARGET]
        input_cols = [defn[0] for defn in col_defn if defn[-1] in {InputTypes.OBSERVED_INPUT,
                                                                   InputTypes.KNOWN_INPUT,
                                                                   InputTypes.STATIC_INPUT}]
        id_col = [defn[0] for defn in col_defn if defn[-1] == InputTypes.ID]
        if len(id_col) > 1:
            raise ValueError("Multiple ID cols detected!!!")
        id_col = id_col[0]

        inputs = []
        outputs = []
        active_entries = []

        for id, sliced in df.groupby(id_col):

            ip = sliced[input_cols].values
            op = sliced[target_cols].values

            ips, ops, flags = cls.batch_single(ip, op, lookback)
            inputs.append(ips)
            outputs.append(ops)
            active_entries.append(flags)

        tup = (np.concatenate(l, axis=0) for l in [inputs, outputs, active_entries])
        return tup


    @staticmethod
    def batch_single(inputs, outputs, state_window):

        # Zero-pad data as required - record number of zero paddings for sequence length
        total_time_steps = inputs.shape[0]
        additional_time_steps_required = state_window - (total_time_steps % state_window)

        input_size = inputs.shape[-1]
        output_size = outputs.shape[-1]


        if additional_time_steps_required > 0:
            inputs = np.concatenate([inputs, np.zeros((additional_time_steps_required, input_size))])
            outputs = np.concatenate([outputs, np.zeros((additional_time_steps_required, output_size))])

        # Reshape inputs now
        inputs = inputs.reshape((-1, state_window, input_size))
        outputs = outputs.reshape((-1, state_window, output_size))

        batch_size = inputs.shape[0]
        sequence_lengths = [(state_window if i != batch_size - 1 else state_window - additional_time_steps_required)
                            for i in range(batch_size)]

        # Setup active entries
        active_entries = np.ones((outputs.shape[0], outputs.shape[1]))
        for i in range(outputs.shape[0]):
            active_entries[i, sequence_lengths[i]:] = 0


        return inputs[:-1], outputs[:-1], active_entries[:-1]


class EfficientRnfBatcher(EfficientAutoregressiveBatcher):

    @classmethod
    def batch(cls, df, col_defn, lookback):

        target_cols = [defn[0] for defn in col_defn if defn[-1] == InputTypes.TARGET]
        input_cols = [defn[0] for defn in col_defn if defn[-1] in {InputTypes.OBSERVED_INPUT,
                                                                   InputTypes.KNOWN_INPUT,
                                                                   InputTypes.STATIC_INPUT}]
        id_col = [defn[0] for defn in col_defn if defn[-1] == InputTypes.ID]
        if len(id_col) > 1:
            raise ValueError("Multiple ID cols detected!!!")
        id_col = id_col[0]

        inputs = []
        outputs = []
        active_entries = []

        for id, sliced in df.groupby(id_col):
            ip = sliced[input_cols].values
            op = sliced[target_cols].values

            ips, ops, flags = cls.batch_single(ip, op, lookback)
            inputs.append(ips)
            outputs.append(ops)
            active_entries.append(flags)

        inputs, outputs, active_entries = (np.concatenate(l, axis=0) for l in [inputs, outputs, active_entries])

        # Group into RNF input format
        inputs = np.concatenate([inputs, outputs], axis=-1)  # outputs put at end
        flags = np.stack([active_entries, active_entries], axis=-1)  # switched all inputs on by default, but customisable

        inputs = [inputs, flags]

        return inputs, outputs, active_entries

class FullRnfBatcher(EfficientRnfBatcher):

    """
    Batches by sliding a window of fixed size over the time series
    """

    @staticmethod
    def batch_single(inputs, outputs, state_window):


        def batch_by_sliding_window(data, lags):

            T = data.shape[0]
            return np.stack([data[i:T-lags+i+1] for i in range(lags)], axis=1)

        inputs = batch_by_sliding_window(inputs, state_window)
        outputs = batch_by_sliding_window(outputs, state_window)
        active_entries = np.ones_like(outputs[..., 0])

        return inputs, outputs, active_entries

