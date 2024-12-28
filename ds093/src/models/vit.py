from typing import List, Tuple

import tensorflow as tf


class PatcheLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_size: int,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size

    def call(
        self,
        images: tf.Tensor
    ) -> tf.Tensor:
        batch_size = tf.shape(images)[0]

        # (batch_size, num_patch_h, num_patch_w, patch_size**2*num_channel)
        # e.g. input = (48, 32, 3), patch_size = 4
        #   => patches = (12, 8, 4*4*3) = (12, 8, 48)
        patches = tf.image.extract_patches(
            images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        # (batch_size, num_patches, patch_size**2*num_channel)
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches


class PatchEncodeLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_patches: int,
        projection_dim: int,
    ) -> None:
        super().__init__()

        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim,
        )

    def call(self, patch: tf.Tensor) -> tf.Tensor:
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class FeedForwardLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_units: List[int],
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        self.layers = []
        for units in hidden_units:
            self.layers.append(tf.keras.layers.Dense(units, activation=tf.nn.gelu))
            self.layers.append(tf.keras.layers.Dropout(dropout_rate))

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        projection_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-6
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon

        self.layer_norm_inp = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm_att = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=dropout_rate,
        )
        self.ff = FeedForwardLayer(
            hidden_units=[projection_dim*2, projection_dim],
            dropout_rate=dropout_rate,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.layer_norm_inp(inputs)
        att_outputs = self.attention(x, x)
        # layer norm + residual connection
        inputs = inputs + att_outputs
        x = self.layer_norm_att(inputs)
        outputs = self.ff(x)
        # residual connection
        return inputs + outputs


class Encoder(tf.keras.models.Model):
    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        projection_dim: int,
        num_heads: int,
        num_encoder_blocks: int = 2,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.dropout_rate = dropout_rate

        self.patch_layer = PatcheLayer(patch_size)
        self.patch_encode_layer = PatchEncodeLayer(
            num_patches=num_patches,
            projection_dim=projection_dim
        )
        self.encoder_layers = [
            EncoderLayer(
                patch_size=patch_size,
                num_patches=num_patches,
                projection_dim=projection_dim,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_encoder_blocks)
        ]

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        patches = self.patch_layer(inputs)
        enc_patches = self.patch_encode_layer(patches)
        outputs = enc_patches
        for layer in self.encoder_layers:
            outputs = layer(outputs)
        return outputs


class VisionTransformer:
    def __init__(
        self,
        input_shape: Tuple[int],
        num_classes: int,
        resize: int = None,
        patch_size: int = 4,
        projection_dim: int = 64,
        num_heads: int = 4,
        num_encoder_blocks: int = 2,
        mlp_hidden_units: List[int] = [512, 128],
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.resize = resize if resize else input_shape[0]
        self.num_patches = int((self.resize / patch_size)**2)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.mlp_hidden_units = mlp_hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def build(self, optimizer: tf.keras.optimizers.Optimizer = None) -> None:
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        resized = tf.keras.layers.Resizing(self.resize, self.resize)(inputs)
        enc_outputs = Encoder(
            patch_size=self.patch_size,
            num_patches=self.num_patches,
            projection_dim=self.projection_dim,
            num_heads=self.num_heads,
            num_encoder_blocks=self.num_encoder_blocks,
            dropout_rate=self.dropout_rate,
        )(resized)

        enc_outputs = tf.keras.layers.LayerNormalization()(enc_outputs)
        enc_outputs = tf.keras.layers.Flatten()(enc_outputs)
        enc_outputs = tf.keras.layers.Dropout(self.dropout_rate)(enc_outputs)

        ff_outputs = FeedForwardLayer(
            hidden_units=self.mlp_hidden_units,
            dropout_rate=self.dropout_rate
        )(enc_outputs)

        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax")(ff_outputs)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics="accuracy",
        )

    def fit(
        self,
        x: tf.Tensor = None,
        y: tf.Tensor = None,
        batch_size: int = 128,
        epochs: int = 1,
        validation_data: Tuple[tf.Tensor, tf.Tensor] = None,
        callbacks: Tuple[tf.keras.callbacks.Callback] = None,
        **kwargs,
    ) -> tf.keras.callbacks.History:
        return self.model.fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            **kwargs,
        )
