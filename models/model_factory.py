"""
model_factory.py


Created by limsi on 23/03/2020
"""


import tensorflow as tf

from models.autoregressive_rnf import RNFCell, RNFInput

import tensorflow_probability as tfp

K = tf.keras.backend

def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):

  """Returns simple Keras linear layer.
  Args:
    size: Output size
    activation: Activation function to apply if required
    use_time_distributed: Whether to apply layer across time
    use_bias: Whether bias should be included in layer
  """
  linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
  if use_time_distributed:
    linear = tf.keras.layers.TimeDistributed(linear)
  return linear


def make_rnf(params, dump_states=False, set_initial_states=False):

    # Sizes
    time_steps = int(params['total_time_steps'])
    input_size = int(params['input_size'])  # this includes target
    output_size = int(params['output_size'])

    # Network params
    hidden_layer_size = int(params['hidden_layer_size'])
    learning_rate = float(params['learning_rate'])
    max_gradient_norm = float(params['max_norm'])
    dropout_rate = float(params['dropout_rate'])
    skip_rate = float(params['skip_rate'])
    minibatch_size = int(params['minibatch_size'])

    Cell = RNFCell
    cell = Cell(hidden_layer_size, output_size, skip_rate, dropout_rate, states_only=dump_states)

    inputs = tf.keras.layers.Input(shape=(time_steps, input_size))
    flags = tf.keras.layers.Input(shape=(time_steps,2))

    if not set_initial_states:
        rnf = tf.keras.layers.RNN(cell, return_sequences=True, return_state=False)
        outputs = rnf(RNFInput(data=inputs, flags=flags))
        model_inputs = [inputs, flags]
    else:
        rnf= tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
        state_h = tf.keras.layers.Input(shape=(hidden_layer_size))
        state_c = tf.keras.layers.Input(shape=(hidden_layer_size))
        outputs = rnf(RNFInput(data=inputs, flags=flags), initial_state=[state_h, state_c])
        model_inputs = [inputs, flags, state_h, state_c]

    model = tf.keras.models.Model(model_inputs, outputs)

    print(model.summary())

    # Skip training compilation if we only want internal states
    if dump_states or set_initial_states:
        print("Skipping training settings...")
        return model

    # Setting up training
    adam = tf.keras.optimizers.Adam(
        lr=learning_rate, clipnorm=max_gradient_norm)

    def rnf_switch_loss(y, y_pred):

        # computes log likelihood int training phase, and mse in validation phase (changeable)

        def compute_train_loss():

            neg_llhoods = y_pred[..., :output_size]  # summed over all stages in rnf cell
            counts = y_pred[..., -1:]

            # Adjustment factor for equal weights
            return K.sum(neg_llhoods/counts, axis=-1)

        def compute_val_loss():

            mean, std = y_pred[..., :output_size], y_pred[..., output_size:]
            return tf.reduce_mean(tf.square(mean-y), axis=-1)


        return K.in_train_phase(compute_train_loss(), compute_val_loss())

    model.compile(loss=rnf_switch_loss, optimizer=adam, sample_weight_mode='temporal')

    return model


def make_lstm(params, set_initial_states=False):
    """
    Get LSTM with Gaussian output. Equivalent to Deep AR without scaling for one-step-ahead predictions

    """
    # Sizes
    time_steps = int(params['total_time_steps'])
    input_size = int(params['input_size'])  - 1 # no target
    output_size = int(params['output_size'])

    # Network params
    hidden_layer_size = int(params['hidden_layer_size'])
    learning_rate = float(params['learning_rate'])
    max_gradient_norm = float(params['max_norm'])
    dropout_rate = float(params['dropout_rate'])
    minibatch_size = int(params['minibatch_size'])

    inputs = tf.keras.layers.Input(
        shape=(
            time_steps,
            input_size,
        ))

    lstm_layer = tf.keras.layers.LSTM(
        hidden_layer_size,
        return_sequences=True,
        return_state=set_initial_states,
        stateful=False,
        activation='elu',
        recurrent_activation='sigmoid',
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        unroll=False,
        use_bias=True)

    if set_initial_states:
        h_in = tf.keras.layers.Input(shape=(hidden_layer_size))
        c_in = tf.keras.layers.Input(shape=(hidden_layer_size))
        lstm, state_h, state_c = lstm_layer(inputs, initial_state=[h_in, c_in])
    else:
        lstm = lstm_layer(inputs)

    lstm = tf.keras.layers.Dropout(rate=dropout_rate)(lstm)

    means = linear_layer(output_size, use_time_distributed=True)(lstm)
    stds = linear_layer(output_size, use_time_distributed=True, activation='softplus')(lstm)
    outputs = K.concatenate([means, stds], axis=-1)

    if set_initial_states:
        inputs = [inputs, h_in, c_in]
        outputs = [outputs, state_h, state_c]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    print(model.summary())

    adam = tf.keras.optimizers.Adam(
        lr=learning_rate, clipnorm=max_gradient_norm)

    def loss(y, y_pred):

        means,stds = y_pred[..., :output_size], y_pred[...,output_size:]
        neg_loglikelihood = -tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=stds+1e-8).log_prob(y)
        mse = K.mean(tf.square(y - means), axis=-1)

        return K.in_train_phase(neg_loglikelihood, mse)

    model.compile(
          loss=loss, optimizer=adam, sample_weight_mode='temporal')

    return model



# -------------------------------------------------------------------------------
# Core routine

def make_model(model_name, params, set_initial_states=False):
    print()
    print("Making model '{}' with params below:".format(model_name))
    print("# ---------------------------------------------------")
    for k in params:
        print(k, "=", params[k])
    print("# ---------------------------------------------------")

    if model_name == 'lstm':
        return make_lstm(params, set_initial_states=set_initial_states)
    elif model_name == 'rnf':
        return make_rnf(params, set_initial_states=set_initial_states)
    else:
        raise ValueError("Unrecognised model={}".format(model_name))