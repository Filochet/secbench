###
# Copyright CEA (Commissariat à l'énergie atomique et aux
# énergies alternatives) (2017-2025)
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
###
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from ._network import ConvLayerSpec, DenseLayerSpec
from .helpers import key_scores, rank_of
from .metrics import cond_mean

logger = logging.getLogger()


class ProfiledAttack(abc.ABC):
    """
    Generic abstraction of a profiled side-channel attack.
    """

    def __init__(self, target_variable_fn=None):
        """
        Create a new instance.

        A function that computes target variables can be found. This function
        takes as first argument secret data (e.g., secret key) and as second
        argument public data (e.g., plaintexts)
        """
        self._target_variable_fn = target_variable_fn

    # ===
    # Methods to be reimplemented by custom models.
    # ===

    @abc.abstractmethod
    def fit(self, X, y, secret=None, **kwargs):
        """
        Fit the model using training data.

        If you defined a target variable function (``target_variable_fn``
        argument) in the constructor, you should also pass secret data here.
        The model will then be trained to predict ``z = self.target(secret, y)``
        instead of y.
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, X):
        """
        Return intermediate variables probability distribution for a given observation.

        :returns: an array of shape ``(n_traces, n_classes)``
        """
        pass

    # ===
    # Methods provided (can be overwritten in subclasses if needed).
    # ===

    def target(self, secret, *args, **kwargs) -> np.ndarray:
        """
        Compute the target variable that we predict from public and private information.
        """
        if self._target_variable_fn is None:
            raise AttributeError(
                "you must pass a target variable function ("
                "target_variable_fn) to the ProfiledAttack constructor in"
                "order to use this functionality."
            )
        return self._target_variable_fn(secret, *args, **kwargs)

    def predict_proba_log(self, X):
        """
        Same as :py:class:`ProfiledAttack.predict_proba`, but in logarithm domain.
        """
        # Add a small offset to avoid 0 logs.
        return np.log(self.predict_proba(X) + 1e-9)

    def key_scores(self, X, secret_values, *args, **kwargs) -> np.ndarray:
        """
        Compute the score of each key hypothesis.

        This functions internally calls :py:func:`secbench.processing.helpers.key_scores`, refer to
        its docstring for more information.
        """
        return key_scores(
            self.predict_proba_log(X),
            self._target_variable_fn,
            secret_values,
            *args,
            **kwargs,
        )

    def guessing_entropy(self, X, y, expected_secret, traces_selector, num_classes):
        """
        Compute a guessing entropy by computing key rank for different number of traces.

        :param X: attack data
        :param y: public labels associated with traces
        :param expected_secret: secret data used on the attack traces
        :param traces_selector: return a subset of indices to select traces.
            This is used to see the evolution of the rank with the number of
            traces.
        :param num_classes: number of classes being predicted.
        """
        pred_lg = self.predict_proba_log(X)
        rank = []
        for selector in traces_selector:
            scores = key_scores(
                pred_lg[selector], self.target, list(range(num_classes)), y[selector]
            )
            rank.append(rank_of(scores, expected_secret))
        return np.array(rank)


class SklearnModel(ProfiledAttack):
    """
    Wrap any Scikit-learn model.
    """

    def __init__(self, model, target_variable_fn=None):
        super().__init__(target_variable_fn=target_variable_fn)
        self._model = model

    def fit(self, X, y, secret=None, **kwargs):
        if secret is not None:
            y = self.target(secret, y)
        # Train the network to predict those from the traces.
        self._model.fit(X, y)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class ScaNetwork(ProfiledAttack):
    def __init__(
        self,
        network,
        target_variable_fn=None,
        add_softmax=True,
    ):
        super().__init__(target_variable_fn=target_variable_fn)
        self.__add_softmax = add_softmax
        self.__network = network

    def network(self):
        """
        Return the raw neural network used by this model.
        """
        return self.__network

    def load_weights(self, path):
        """
        Reload weights saved during a training.
        """
        self.__network.load_weights(path)

    def _compute_gradients(self, X):
        import tensorflow as tf

        x_input = tf.convert_to_tensor(X, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_input)
            T = self.__network([x_input], training=False)
            grads = tape.gradient(T, x_input)
        return grads

    def compute_gradients(self, X):
        import tensorflow as tf

        grads = self._compute_gradients(X)
        grads = tf.math.abs(grads)
        avg_grads = tf.reduce_mean(grads, axis=0)
        avg_grads = (avg_grads - np.amin(avg_grads)) / (
            np.amax(avg_grads) - np.amin(avg_grads)
        )
        return avg_grads

    # Methods provided
    def fit_raw(self, *args, **kwargs):
        return self.network().fit(*args, **kwargs)

    def fit(self, X, y, secret=None, **kwargs):
        # Compute target variables
        if secret is not None:
            y = self.target(secret, y)
        # Train the network to predict those from the traces.
        return self.fit_raw(X, y, **kwargs)

    def predict_proba(self, traces):
        from tensorflow import keras

        # We add a softmax layer at the end of the network
        # to normalize prediction probabilities.
        if self.__add_softmax:
            probability_model = keras.Sequential(
                [self.network(), keras.layers.Softmax()]
            )
        else:
            probability_model = self.network()
        return probability_model(traces)


@dataclass
class GenericNetworkBuilder:
    """
    Generic serializable specification of a neural network.

    :param conv_layers: Specification of convolutional layers (can be empty
        for MLP architectures).
    :param batch_normalization: whether batch normalization is applied at the
        input of the network.
    :param dense_layers: Specification of dense layers applied after
        convolutional layers.
    """

    batch_normalization: bool
    conv_layers: list[ConvLayerSpec]
    dense_layers: list[DenseLayerSpec]

    def make_network(self, num_samples: int, num_classes: Optional[int] = None):
        from keras.layers import (
            AveragePooling1D,
            BatchNormalization,
            Conv1D,
            Dense,
            Dropout,
            Flatten,
            Input,
        )
        from keras.models import Model
        from tensorflow import keras  # noqa: F401

        assert len(self.dense_layers) > 0, "network must have at least one dense_layer"
        last_layer = self.dense_layers[-1]
        if num_classes is not None and last_layer.size != num_classes:
            logger.info(
                f"replacing final network layer size {last_layer.size} -> {num_classes}."
            )
            self.dense_layers[-1] = replace(self.dense_layers[-1], size=num_classes)
        # fmt: off

        X = Input(shape=(num_samples, 1), name="input_layer")

        # Input Normalization
        if self.batch_normalization:
            hidden = BatchNormalization(axis=-2)(X)
        else:
            hidden = X

        for i, spec in enumerate(self.conv_layers):
            hidden = Conv1D(spec.kernel_count, spec.kernel_size,
                            activation=spec.activation, padding="same",
                            name=f"conv_layer{i}-filter")(hidden)
            if spec.dropout:
                hidden = Dropout(spec.dropout, name=f"conv_layer{i}-dropout")(
                    hidden)
            if spec.pool_size:
                hidden = AveragePooling1D(spec.pool_size,
                                          strides=spec.pool_stride,
                                          name=f"conv_layer{i}-avgpooling")(
                    hidden)

        # Dense layers
        hidden = Flatten()(hidden)

        for i, spec in enumerate(self.dense_layers):
            hidden = Dense(spec.size, activation=spec.activation,
                           name=f"dense_layer{i}-dense")(
                hidden)
            if spec.dropout:
                hidden = Dropout(spec.dropout, name=f"dense_layer{i}-dropout")(
                    hidden)

        output = hidden
        # fmt: on
        return Model(inputs=[X], outputs=output, name="sca_network")

    def compile(self, num_samples, num_classes=None, **kwargs):
        from tensorflow import keras

        compile_args = kwargs
        if "optimizer" not in compile_args:
            compile_args["optimizer"] = "adam"
        if "loss" not in compile_args:
            compile_args["loss"] = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
        if "metrics" not in compile_args:
            compile_args["metrics"] = [
                keras.metrics.SparseCategoricalAccuracy(name="acc")
            ]
        model = self.make_network(num_samples, num_classes=num_classes)
        model.compile(**compile_args)
        return model

    def build(self, num_samples, num_classes=None, target_variable_fn=None, **kwargs):
        """
        Create a :py:class:`ScaNetwork` instance using this neural network.

        :param num_samples: How many samples are passed as input to the neural
            network.
        :param num_classes: Specify the number of classes to predict.
            If specified, the last dense layer of the neural network will
            "patched" to have the correct number of neurons.
        :param target_variable_fn: optional function for calculating
            intermediate variable.
        :param kwargs: keyword arguments are forwarded to Tensorflow's
            compile method. If unset, we apply "sane" defaults.
        """
        return ScaNetwork(
            self.compile(num_samples, num_classes=num_classes, **kwargs),
            target_variable_fn=target_variable_fn,
        )




class ZBHVNetwork(ScaNetwork):
    """
    Architecture designed by Zaid et. al and proved to work on ASCAD.

    Source: Zaid, G., Bossuet, L., Habrard, A., & Venelli, A. (2020). Methodology for efficient CNN architectures in profiling attacks. IACR Transactions on Cryptographic Hardware and Embedded Systems, 1-36.
    """

    def __init__(
        self, n_samples, n_classes, activation="selu", target_variable_fn=None, **kwargs
    ):
        from keras.layers import (
            AveragePooling1D,
            BatchNormalization,
            Conv1D,
            Dense,
            Flatten,
            Input,
        )
        from keras.models import Model
        from tensorflow import keras

        act = activation
        X = Input(shape=(n_samples, 1), name="X_input")
        hidden = X

        # Block 1
        # This layer does not seems useful...
        # hidden = Conv1D(32, 1, activation=act, padding='same', name='block1_conv1')(hidden)
        # hidden = BatchNormalization()(hidden)
        # hidden = AveragePooling1D(2, strides=2, name='block1_pool')(hidden)

        # Block 2
        hidden = Conv1D(32, 25, activation=act, padding="same", name="block2_conv1")(
            hidden
        )
        hidden = BatchNormalization()(hidden)
        hidden = AveragePooling1D(25, strides=25, name="block2_pool")(hidden)
        # Block 3
        hidden = Conv1D(64, 3, activation=act, padding="same", name="block3_conv1")(
            hidden
        )
        hidden = BatchNormalization()(hidden)
        # NOTE: we can replace the pooling layer with a convolution layer
        # hidden = Conv1D(64, 11, strides=4)(hidden)
        hidden = AveragePooling1D(4, strides=4, name="block3_pool")(hidden)

        # Classification block

        hidden = Flatten(name="flatten")(hidden)

        hidden = Dense(15, activation=act, name="fc1")(hidden)
        hidden = Dense(15, activation=act, name="fc2")(hidden)
        hidden = Dense(15, activation=act, name="fc3")(hidden)

        output = Dense(n_classes, name="y_pred")(hidden)

        model = Model(inputs=[X], outputs=output, name="statistics_network")
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
        )
        super().__init__(model, target_variable_fn=target_variable_fn, **kwargs)


class ClassPCA(TransformerMixin, BaseEstimator):
    """
    This transformer performs a class PCA.

    First it computes mean of training data per classes, resulting in an array
    of shape (n_classes, n_features). Then it fits a PCA on this data.

    The resulting PCA is applied on any input data of
    shape (n_samples, n_features).
    """

    def __init__(
        self,
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        random_state=None,
    ):
        # NOTE: We need to fully duplicate the parameters to be compliant with
        # the BaseEstimator interface...
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        self.pca_ = PCA(
            n_components=self.n_components,
            copy=self.copy,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            tol=self.tol,
            iterated_power=self.iterated_power,
            random_state=self.random_state,
        )
        m = cond_mean(X, y, num_classes=len(encoder.classes_))
        self.pca_.fit(m)
        return self

    def transform(self, X):
        check_is_fitted(self, "pca_")
        return self.pca_.transform(X)