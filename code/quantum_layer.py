import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pennylane as qml


@tf.keras.utils.register_keras_serializable(package="Custom")
class QuantumLayer(layers.Layer):
    def __init__(self, n_qubits, q_depth=2, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def build(self, input_shape):
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(self.q_depth, self.n_qubits, 3),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs_np = inputs.numpy()
        weights_np = self.q_weights.numpy()

        outputs = []
        for x in inputs_np:
            result = self.circuit(x, weights_np)
            outputs.append(np.array(result, dtype=np.float32))

        return tf.convert_to_tensor(np.array(outputs, dtype=np.float32))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_qubits

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_qubits": self.n_qubits,
            "q_depth": self.q_depth,
        })
        return config