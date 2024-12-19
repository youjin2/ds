import re
from typing import Tuple, Dict

import tensorflow as tf


def scaled_dot_product_attention(
    query: tf.Tensor,
    key: tf.Tensor,
    value: tf.Tensor,
    mask: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    # query.shape: (batch_size, num_heads, len(query sentence), d_model//num_heads)
    # key.shape: (batch_size, num_heads, len(key sentence), d_model//num_heads)
    # value.shape: (batch_size, num_heads, len(value sentence), d_model//num_heads)
    # padding_mask.shape: (batch_size, 1, 1, len(key sentence))

    # attention score (QK^T)
    # attention_score.shape: (batch_size, num_heads, len(query sentence), len(key_sentence))
    attention_score = tf.matmul(query, key, transpose_b=True)

    # scaling (divide by sqrt(d_k=d_model//num_heads))
    dk = tf.cast(tf.shape(key)[-1], dtype="float32")
    logits = attention_score / tf.math.sqrt(dk)

    # masking (used in the decoder's masked-self attention)
    # softmax(-1e-9) ≈ 0
    if mask is not None:
        logits += (mask * -1e9)

    # applied softmax along the key axis
    # attention weight.shape: (batch_size, num_heads, len(query sentence), len(key sentence))
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # output.shape: (batch_size, num_heads, len(query sentence), d_model//num_heads)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights


def create_padding_mask(x: tf.Tensor) -> tf.Tensor:
    # <PAD> = 0
    mask = tf.cast(tf.math.equal(x, 0), dtype="float32")
    # (batch_size, 1, 1, len(key sentence))
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x: tf.Tensor) -> tf.Tensor:
    seq_len = tf.shape(x)[1]
    ltri_mat = tf.linalg.band_part(tf.ones((seq_len, seq_len)), num_lower=-1, num_upper=0)
    # mat[i, j] = 0 if i >= j, 1 if j > i
    look_ahead_mask = 1 - ltri_mat
    # also applies padding_mask
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position: int, d_model: int) -> None:
        super().__init__()

        # shape: (position, d_model)
        #  - position: the number of words in input sentence)
        #  - model: output dimension of each layer in transformer
        self.pos_encoding = self.encode(position, d_model)

    def get_angles(self, position: tf.Tensor, index: tf.Tensor, d_model: int) -> tf.Tensor:
        num = position
        denom = tf.pow(
            tf.constant(10000., dtype="float32"),
            tf.divide(
                # [0, 0, 1, 1, ..., d_model/2, d_model/2]
                2 * (index // 2),
                tf.constant(d_model, dtype="float32")
            )
        )
        return tf.divide(num, denom)

    def encode(self, position: int, d_model: int) -> tf.Tensor:
        angle_rads = self.get_angles(
            position=tf.range(position, dtype="float32")[:, tf.newaxis],
            index=tf.range(d_model, dtype="float32")[tf.newaxis, :],
            d_model=d_model,
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        stacked = tf.stack([sines, cosines], axis=-1)
        angle_rads = tf.reshape(stacked, shape=(tf.shape(sines)[0], -1))
        # angle_rads = np.zeros(angle_rads.shape)
        # angle_rads[:, 0::2] = sines
        # angle_rads[:, 1::2] = cosines

        # pos_encoding = tf.constant(angle_rads, dtype="float32")
        pos_encoding = tf.cast(angle_rads, dtype="float32")
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return pos_encoding

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # NOTE: input length may smaller thatn the vocabulary size
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        name: str = "multi_head_attention",
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be a multiple of num_heads"
            )

        self.d_model = d_model
        self.num_heads = num_heads

        # d_model = 512, num_heads = 8
        # => d_k = 8
        self.dk = d_model // num_heads

        # W_Q, W_K, W_V
        self.dense_q = tf.keras.layers.Dense(units=d_model)
        self.dense_k = tf.keras.layers.Dense(units=d_model)
        self.dense_v = tf.keras.layers.Dense(units=d_model)

        # W_O
        self.dense_o = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs: tf.Tensor, batch_size: int) -> tf.Tensor:
        inputs = tf.reshape(
            inputs,
            shape=(batch_size, -1, self.num_heads, self.dk)
        )
        # shape: (batch_size, num_heads, len(input sentence), d_model//num_heads)
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        q, k, v, mask = inputs["query"], inputs["key"], inputs["value"], inputs["mask"]
        batch_size = tf.shape(q)[0]

        # NOTE: (q, k, v) for each attention layer can be computed at once
        #   - e.g. X: (batch_size, len(input sentence))
        #     - X*W_Q = [X*W_{1, Q}, X*W_{2, Q}, ..., X*W_{d_model//num_heads, Q}]
        # q.shape: (batch_size, len(query sentence), d_model)
        # k.shape: (batch_size, len(key sentence), d_model)
        # v.shape: (batch_size, len(value sentence), d_model)
        q = self.dense_q(q)
        k = self.dense_k(k)
        v = self.dense_v(v)

        # NOTE: (q, k, v) for each attention layer are stacked in rows
        # q.shape: (batch_size, num_heads, len(query sentence), d_model//num_heads)
        # k.shape: (batch_size, num_heads, len(key sentence), d_model//num_heads)
        # v.shape: (batch_size, num_heads, len(value sentence), d_model//num_heads)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # shape: (batch_size, num_heads, len(query sentence), d_model//num_heads)
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        # (batch_size, len(query sentence), num_heads, d_model//num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, len(query sentence), d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        outputs = self.dense_o(concat_attention)
        return outputs


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        name: str = "encoder_layer",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self._name = name

        # self.inputs = tf.keras.Input(shape=(None, d_model), name=f"{name}_input")
        # self.padding_mask = tf.keras.Input(shape=(1, 1, None), name=f"{name}_padding_mask")
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, name=f"{name}_attention")
        self.dropout_att = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_dense = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_att = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.dense2 = tf.keras.layers.Dense(d_model, activation="relu")

    def call(self, x: tf.Tensor, padding_mask: tf.Tensor, training: bool = False):
        inputs = x
        attention = self.attention({
            "query": inputs,
            "key": inputs,
            "value": inputs,
            "mask": padding_mask,
        })

        # dropout -> residual connection + layer normalization
        attention = self.dropout_att(attention, training=training)
        attention = self.layer_norm_att(inputs + attention)
        outputs = self.dense1(attention)
        outputs = self.dense2(outputs)

        # dropout -> residual connection + layer normalization
        outputs = self.dropout_dense(outputs, training=training)
        outputs = self.layer_norm_dense(attention + outputs)

        return outputs


class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        vocab_size: int,
        num_layer: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        name: str = "encoder",
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self._name = name

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_embedding = PositionalEncoding(position=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.encoder_layers = [
            EncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=dropout_rate,
                name=f"{self._name}_layer_{i}"
            )
            for i in range(num_layer)
        ]

    def call(self, inputs: tf.Tensor, padding_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        embeddings = self.embedding(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, dtype="float32"))
        embeddings = self.pos_embedding(embeddings)
        outputs = self.dropout(embeddings, training=training)
        for layer in self.encoder_layers:
            outputs = layer(outputs, padding_mask)

        return outputs


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        name: str = "decoder_layeR",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self._name = name

        # self.inputs = tf.keras.Input(shape=(None, d_model), name=f"{name}_input")
        # self.padding_mask = tf.keras.Input(shape=(1, 1, None), name=f"{name}_padding_mask")
        self.attention1 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, name=f"{name}_attention1")
        self.attention2 = MultiHeadAttention(d_model=d_model, num_heads=num_heads, name=f"{name}_attention2")
        self.dropout_att1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_att2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_dense = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm_att1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_att2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_dense = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = tf.keras.layers.Dense(d_ff, activation="relu")
        self.dense2 = tf.keras.layers.Dense(d_model, activation="relu")

    def call(
        self,
        inputs: tf.Tensor,
        encoder_outputs: tf.Tensor,
        look_ahead_mask: tf.Tensor,
        padding_mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        # 1st layer: masked self attention
        attention1 = self.attention1({
            "query": inputs,
            "key": inputs,
            "value": inputs,
            "mask": look_ahead_mask,
        })

        # dropout -> residual connection + layer normalization
        attention1 = self.dropout_att1(attention1, training=training)
        attention1 = self.layer_norm_att1(inputs + attention1)

        # 2nd layer: encoder decoder attention
        attention2 = self.attention2({
            "query": attention1,
            "key": encoder_outputs,
            "value": encoder_outputs,
            "mask": padding_mask,
        })

        # dropout -> residual connection + layer normalization
        attention2 = self.dropout_att2(attention2, training=training)
        attention2 = self.layer_norm_att2(attention1 + attention2)

        # feed forward
        outputs = self.dense1(attention2)
        outputs = self.dense2(outputs)

        # dropout -> residual connection + layer normalization
        outputs = self.dropout_dense(outputs, training=training)
        outputs = self.layer_norm_dense(attention2 + outputs)

        return outputs


class Decoder(tf.keras.models.Model):
    def __init__(
        self,
        vocab_size: int,
        num_layer: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1,
        name: str = "decoder",
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self._name = name

        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_embedding = PositionalEncoding(position=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_layers = [
            DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout_rate=dropout_rate,
                name=f"{self._name}_layer_{i}"
            )
            for i in range(num_layer)
        ]

    def call(
        self,
        inputs: tf.Tensor,
        encoder_outputs: tf.Tensor,
        look_ahead_mask: tf.Tensor,
        padding_mask: tf.Tensor,
    ) -> tf.Tensor:
        embeddings = self.embedding(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, dtype="float32"))
        embeddings = self.pos_embedding(embeddings)
        outputs = self.dropout(embeddings)
        for layer in self.decoder_layers:
            outputs = layer(outputs, encoder_outputs, look_ahead_mask, padding_mask)

        return outputs


class Transformer:
    def __init__(
        self,
        vocab_size: int,
        num_layer: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_length: int,
        tokenizer: tf.keras.preprocessing.text.Tokenizer,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        name: str = "transformer",
    ) -> None:
        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.tokenizer = tokenizer
        self._name = name

        self.encoder = Encoder(
            vocab_size=vocab_size,
            num_layer=num_layer,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            name="encoder",
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            num_layer=num_layer,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            name="decoder",
        )

    def _loss_fn(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
        y_true = tf.reshape(y_true, shape=(-1, self.max_length-1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(y_true, y_pred)

        # ignore mask token
        mask = tf.cast(tf.not_equal(y_true, 0), dtype="float32")
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)

    def build(self, optimizer: tf.keras.optimizers.Optimizer = None) -> None:
        # define inputs
        inputs = tf.keras.Input(shape=(None,), name="inputs")
        dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

        # encoder padding mask
        enc_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask,
            output_shape=(1, 1, None),
            name="encoder_padding_mask",
        )(inputs)

        # decoder look ahead mask (used at the masked self attention in the 1st layer)
        dec_look_ahead_mask = self.decoder_look_ahead_mask = tf.keras.layers.Lambda(
            create_look_ahead_mask,
            output_shape=(1, None, None),
            name="decoder_look_ahead_mask"
        )(dec_inputs)

        # decoder padding mask (used at the enc-dec attention in the 2nd layer)
        dec_padding_mask = self.decoder_padding_mask = tf.keras.layers.Lambda(
            create_padding_mask,
            output_shape=(1, 1, None),
            name="decoder_padding_mask"
        )(inputs)

        # encoder/decoder outputs
        enc_outputs = self.encoder(inputs, enc_padding_mask)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_look_ahead_mask, dec_padding_mask)

        outputs = tf.keras.layers.Dense(units=self.vocab_size, name="outputs")(dec_outputs)

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.model = tf.keras.models.Model(
            inputs=[inputs, dec_inputs],
            outputs=outputs,
            name=self._name
        )
        self.model.compile(
            optimizer=optimizer,
            loss=self._loss_fn,
        )

    def fit(
        self,
        x: tf.Tensor = None,
        y: tf.Tensor = None,
        epochs: int = 1,
        validation_data: Tuple[tf.Tensor, tf.Tensor] = None,
        callbacks: Tuple[tf.keras.callbacks.Callback] = None,
    ) -> tf.keras.callbacks.History:
        return self.model.fit(
            x=x,
            y=y,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
        )

    def preprocess_sentence(self, sentence: str) -> str:
        # add space between word and puntuaction
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = sentence.strip()
        return sentence

    def evaluate(self, sentence: str) -> str:
        sos, eos = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size+1]
        sentence = self.preprocess_sentence(sentence)
        # sentence.shape: (1, length of tokens + 2)
        sentence = tf.expand_dims(
            sos + self.tokenizer.encode(sentence) + eos,
            axis=0
        )
        # outputs.shape: (1, 1)
        outputs = tf.expand_dims(sos, axis=0)

        # start prdiction
        for _ in range(self.max_length):
            prob = self.model(inputs=[sentence, outputs], training=False)

            # pick only the current prediction value
            prob = prob[:, -1:, :]
            pred_id = tf.cast(tf.argmax(prob, axis=-1), dtype="int32")

            if tf.equal(pred_id, eos[0]):
                break

            outputs = tf.concat([outputs, pred_id], axis=-1)

        return tf.squeeze(outputs, axis=0)

    def predict(self, sentence: str) -> str:
        # sequence of integer values
        prediction = self.evaluate(sentence)

        # index to character
        predicted_sentence = self.tokenizer.decode([
            i for i in prediction if i < self.tokenizer.vocab_size
        ])

        return predicted_sentence
