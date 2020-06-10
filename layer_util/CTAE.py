import keras
from keras.models import Model
from keras.layers import Input, Conv1D, LeakyReLU, MaxPool1D, LSTM, Bidirectional, TimeDistributed, Dense, Reshape, \
    BatchNormalization, Permute, Reshape, Lambda, RepeatVector, Multiply
from keras.layers import UpSampling2D, Conv2DTranspose
import keras.backend as K


def attention_block(inputs, timesteps, input_dim):
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, timesteps))(a)
    a = Dense(timesteps, activation='softmax')(a)
    if input_dim == 1:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def causal_dilated_CNN_block(input, filters, kernel_size, dilation_rate):
    encoded = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", dilation_rate=dilation_rate,
                     kernel_regularizer=keras.regularizers.l2(0.01))(input)
    encoded = LeakyReLU()(encoded)
    encoded = BatchNormalization(axis=-1)(encoded)

    encoded = Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", dilation_rate=dilation_rate,
                     kernel_regularizer=keras.regularizers.l2(0.01))(encoded)
    encoded = LeakyReLU()(encoded)
    # print("encoded.shape:", encoded.shape, encoded.dtype)
    res = Conv1D(filters=filters, kernel_size=1)(input)
    res = LeakyReLU()(res)
    encoded = BatchNormalization(axis=-1)(encoded)
    # print("res.shape:", res.shape, res.dtype)
    res = keras.layers.add([encoded, res])
    return res


def temporal_autoencoder_v2(input_dim, timesteps, n_filters=50, kernel_size=10, pool_size=10, n_units=[50, 1]):
    assert (timesteps % pool_size == 0)

    # Input
    x = Input(shape=(timesteps, input_dim), name='input_seq')

    encoded = causal_dilated_CNN_block(x, n_units[0], kernel_size, 1)

    for i in range(1):
        encoded = causal_dilated_CNN_block(encoded, n_units[0], kernel_size, pow(2, i + 1))

    encoded = MaxPool1D(pool_size)(encoded)

    encoded = Bidirectional(LSTM(n_units[0], return_sequences=True), merge_mode='concat')(encoded)
    encoded = LeakyReLU()(encoded)
    encoded = attention_block(encoded, (timesteps - pool_size) // pool_size + 1, n_units[0] * 2)
    encoded = Bidirectional(LSTM(n_units[1], return_sequences=True), merge_mode='concat')(encoded)
    encoded = LeakyReLU(name='latent')(encoded)

    # Decoder
    decoded = TimeDistributed(Dense(units=n_filters), name='dense')(encoded)  # sequence labeling
    decoded = LeakyReLU(name='act')(decoded)
    decoded = Reshape((-1, 1, n_filters), name='reshape')(decoded)
    decoded = UpSampling2D((pool_size, 1), name='upsampling')(decoded)
    decoded = Conv2DTranspose(input_dim, (kernel_size, 1), padding='same', name='conv2dtranspose')(decoded)
    output = Reshape((-1, input_dim), name='output_seq')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=output, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(timesteps // pool_size, 2 * n_units[1]), name='decoder_input')
    # Internal layers in decoder
    decoded = autoencoder.get_layer('dense')(encoded_input)
    decoded = autoencoder.get_layer('act')(decoded)
    decoded = autoencoder.get_layer('reshape')(decoded)
    decoded = autoencoder.get_layer('upsampling')(decoded)
    decoded = autoencoder.get_layer('conv2dtranspose')(decoded)
    decoder_output = autoencoder.get_layer('output_seq')(decoded)

    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder
