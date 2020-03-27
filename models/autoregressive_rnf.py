"""
autoregressive_rnf.py


Created by limsi on 24/03/2020
"""

import collections
import tensorflow as tf
import tensorflow_probability as tfp

K = tf.keras.backend
LSTMCell = tf.keras.layers.LSTMCell

# Helpful blocks
class GaussianMLPBlock(tf.keras.layers.Layer):

    def __init__(self, hidden_size, output_size):

        super().__init__()
        self.hidden_layer = tf.keras.layers.Dense(hidden_size, activation='elu')
        self.mean_layer = tf.keras.layers.Dense(output_size)
        self.std_layer = tf.keras.layers.Dense(output_size, activation='softplus')

    def call(self, inputs):
        hidden = self.hidden_layer(inputs)
        mean = self.mean_layer(hidden)
        std = self.std_layer(hidden)

        return mean, std

# ---------------------------------------------------------------------------------------
# RNF Layers
# - Implementation based on https://www.tensorflow.org/guide/keras/rnn#rnn_layers_and_rnn_cells

RNFInput = collections.namedtuple('RNFInput', ['data', 'flags'])


_dropout_cache = {} # for recurrent dropout
class RNFCell(tf.keras.layers.Layer):

    """
    Implements single RNF cell
    """

    def __init__(self,
                 hidden_layer_size,
                 output_size,
                 skip_rate,
                 dropout_rate,
                 alpha_x=1.0,  # propagation regularisation weight
                 alpha_y=1.0,  # error correction regularisation weight
                 activation='elu',
                 recurrent_activation='sigmoid',
                 states_only=False,
                 **kwargs):


        super().__init__(**kwargs)

        self.states_only = states_only
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        # Encoders

        self.state_transition_cell = LSTMCell(units=hidden_layer_size,
                                                  activation=activation,
                                                  recurrent_activation=recurrent_activation)
        self.input_dynamics_cell = LSTMCell(units=hidden_layer_size,
                                                              activation=activation,
                                                              recurrent_activation=recurrent_activation)
        self.error_correction_cell = LSTMCell(units=hidden_layer_size,
                                                              activation=activation,
                                                              recurrent_activation=recurrent_activation)
        # Dropout
        self.emissions_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.recurrent_dropout = tf.keras.layers.Dropout(dropout_rate)

        # Params
        self.skip_rate = skip_rate
        self.dropout_rate = dropout_rate
        self.state_size = self.error_correction_cell.state_size
        self.output_size = output_size * 2 if not self.states_only else self.state_size
        self.num_targets = output_size

        # Decoder
        self.emission_decoder = GaussianMLPBlock(hidden_layer_size,
                                                 output_size)
    def call(self,
             inputs,
             state):

        data, active_flags = tf.nest.flatten(inputs)

        # Targets should be concatenated at end for RNF -- handled in batching
        #  N.b. active flags should be (batch x 2), one for each "skippable" stage
        output_size = self.num_targets # as default output size is double
        cell_inputs = data[..., :-output_size]
        cell_observation = data[..., -output_size:]


        # State transition step.
        zeros = K.zeros_like(data[..., :1])  # To avoid reimplementing LSTM cell
        output1, state1 = self.state_transition_cell(zeros, state)
        state1 = [self.recurrent_dropout(state1[0]), state1[1]]
        output1 = self.emissions_dropout(output1)


        # Input dynamics step.
        output2, state2 = self.input_dynamics_cell(cell_inputs, state1)
        state2 = [self.recurrent_dropout(state2[0]), state2[1]]
        output2 = self.emissions_dropout(output2)

        def keep_function():
            keep_rate = 1-self.skip_rate

            keep = tfp.distributions.Bernoulli(probs=keep_rate, dtype=tf.float32).sample(
                                sample_shape=tf.shape(active_flags[:, :1]))

            return keep

        def state_switch(prev, cur, keep):

            return[prev[i]*(1-keep) + keep*cur[i] for i,_ in enumerate(prev)]

        keep2 = K.in_train_phase(keep_function()*active_flags[:, :1],   active_flags[:, :1])
        state2 = state_switch(state1, state2, keep2)  #skip2*state1 + (1-skip2)*state2

        # Error correction step.
        output3, state3 = self.error_correction_cell(cell_observation, state2)
        state3 = [self.recurrent_dropout(state3[0]), state3[1]]
        output3 = self.emissions_dropout(output3)

        keep3 = K.in_train_phase(keep_function()*active_flags[:, 1:2],   active_flags[:, 1:2])

        state3 = state_switch(state2, state3, keep3)  #skip3*state2 + (1-skip3)*state3


        # Parcel and determine outputs.
        def format_training_outputs():
            """Returns average negative log likelhood across all stages & number of active stages

            Likelihood directly generated to facilitate training process, and requires
            a custom loss function to train correctly (separately defined)

            N.b. shape = batch x (output_size)*2,
                i.e. likelihoods (batch x output_size)  + padded zeros (batch x size-1)  + counts (batch x 1)
            """

            def calc_negative_llhood(x):

                mean, std = self.emission_decoder(x)

                neg_llhood = -tfp.distributions.Normal(loc=mean, scale=std+1e-8).log_prob(cell_observation)

                return neg_llhood

            neg_llhoods = 0.0
            counts = 0.
            for lstm_op, flags, alpha in [(output1, 1., self.alpha_x),
                                          (output2, keep2, 1.),
                                          (output3, keep3, self.alpha_y)]:


                loss_function_weight = flags*alpha  # keep flag & regularisation alpha
                neg_llhoods += calc_negative_llhood(lstm_op)*loss_function_weight
                counts += flags*alpha  # required to recover the correct weighting in the loss function

            placeholder = K.concatenate([tf.zeros_like(neg_llhoods)[..., :-1], counts], axis=-1)
            outputs = K.concatenate([neg_llhoods, placeholder], axis=-1)

            return outputs

        def format_prediction_outputs():
            """Returns mean and standard deviation of forecasts
            N.b. shape = batch x (output_size)*2, i.e. means & stds
            """

            # Determine whether to output the state transition or input dynamics stage
            lstm_op = (1-keep2)*output1 + keep2*output2

            mean, std = self.emission_decoder(lstm_op)

            outputs = K.concatenate([mean, std], axis=-1)

            return outputs

        final_outputs = K.in_train_phase(format_training_outputs(),
                                          format_prediction_outputs())
        if self.states_only:
            return state3, state3
        else:
            return final_outputs, state3
