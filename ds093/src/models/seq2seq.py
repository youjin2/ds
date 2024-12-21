import random
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf


class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.lstms = [
            tf.keras.layers.LSTM(
                units=hidden_dim,
                return_state=True,
                return_sequences=True,
                dropout=dropout_rate,
            )
            for _ in range(num_layers)
        ]

    def call(self, inputs: tf.Tensor, states: Tuple[tf.Tensor, tf.Tensor] = None) -> tf.Tensor:
        x = inputs
        for layer in self.lstms:
            x, state_h, state_c = layer(x, initial_state=states)
            states = [state_h, state_c]

        outputs = x
        return outputs, state_h, state_c

    def initialize_hidden_state(self, batch_size: int):
        return tf.zeros((batch_size, self.hidden_dim)), tf.zeros((batch_size, self.hidden_dim))


class Decoder(tf.keras.models.Model):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.lstms = [
            tf.keras.layers.LSTM(
                units=hidden_dim,
                return_state=True,
                return_sequences=True,
                dropout=dropout_rate,
            )
            for _ in range(num_layers)
        ]
        self.dense = tf.keras.layers.Dense(output_dim, activation="softmax")

    def call(self, inputs: tf.Tensor, states: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        x = inputs
        for layer in self.lstms:
            x, state_h, state_c = layer(x, initial_state=states)
            states = [state_h, state_c]

        outputs = self.dense(x)
        return outputs, state_h, state_c


class Seq2Seq:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        max_length: int,
        dropout_rate: float = 0.1,
        teacher_forcing_ratio: float = 0.5,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.learning_rate = learning_rate

        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout_rate)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers, dropout_rate)

    def call(
        self,
        inputs: tf.Tensor,
        targets: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        # encoder initial states
        batch_size = tf.shape(inputs)[0]
        init_states = self.encoder.initialize_hidden_state(batch_size)

        # encoder outputs & states
        _, *encoder_states = self.encoder(inputs, states=init_states)
        decoder_states = encoder_states

        decoder_outputs = []
        decoder_input = tf.expand_dims(targets[:, 0], axis=1)

        for t in range(1, self.max_length):
            output, *decoder_states = self.decoder(decoder_input, decoder_states)
            decoder_outputs.append(output)

            # teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                # use target as next decoder input
                decoder_input = tf.expand_dims(targets[:, t], axis=1)
            else:
                # use decoder output as next decoder input
                decoder_input = tf.one_hot(tf.argmax(output, axis=-1), depth=self.output_dim)

        return tf.concat(decoder_outputs, axis=1)

    def build(self, optimizer: tf.keras.optimizers.Optimizer = None) -> None:
        inputs = tf.keras.layers.Input(shape=(None, self.input_dim))
        targets = tf.keras.layers.Input(shape=(None, self.output_dim))

        self.model = tf.keras.models.Model(
            inputs=[inputs, targets],
            outputs=self.call(inputs, targets, training=True)
        )
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=self.loss_function,
        )

    def loss_function(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction="none"
        )(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), dtype="float32")
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def fit(
        self,
        X: Tuple[tf.Tensor, tf.Tensor],
        y: tf.Tensor,
        batch_size: int = 256,
        epochs: int = 10,
        validation_split: float = 0.2,
        **kwargs,
    ) -> tf.keras.callbacks.History:
        return self.model.fit(
            x=X,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            **kwargs
        )

    def predict_sequence(
        self,
        input_sequence: np.ndarray,
        index_to_target: Dict[int, str],
        sos_token_index: int,
    ) -> str:
        if len(input_sequence.shape) != 2:
            raise ValueError("input sequence must be 2-dimensional")

        inputs = np.expand_dims(input_sequence, axis=0)
        _, *states = self.encoder(inputs, training=False)

        target_sequence = np.zeros((1, 1, self.output_dim))
        target_sequence[0, 0, sos_token_index] = 1
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            out, *states = self.decoder(target_sequence, states, training=False)
            token_idx = np.argmax(out[0, -1, :])
            char = index_to_target.get(token_idx, " ")
            decoded_sentence += char

            if char == "<eos>" or len(decoded_sentence) > self.max_length:
                stop_condition = True

            target_sequence = np.zeros((1, 1, self.output_dim))
            target_sequence[0, 0, token_idx] = 1.

        return decoded_sentence
