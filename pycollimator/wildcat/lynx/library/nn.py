"""Minimal example of a neural network block.

This uses equinox to start, but we should generalize to support other
JAX libraries as well as PyTorch, pending switchable backends.
"""
from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx  # JAX neural network library
import json

from ..library import FeedthroughBlock, ReduceBlock


# This could probably go into some kind of generic control.py file when we
# have more blocks like this that are useful for control.
class QuadraticCost(ReduceBlock):
    """LQR-type quadratic cost function for a state and input.

    Computes the cost as x'Qx + u'Ru, where Q and R are the cost matrices.
    In order to compute a running cost, combine this with an `Integrator`
    or `IntegratorDiscrete` block.
    """

    def __init__(self, Q, R, name=None):
        super().__init__(2, self._cost, name=name)
        self.Q = Q
        self.R = R

    def _cost(self, inputs):
        x, u = inputs
        return jnp.dot(x, jnp.dot(self.Q, x)) + jnp.dot(u, jnp.dot(self.R, u))


class MLP(FeedthroughBlock):
    def __init__(
        self,
        input_size=None,
        output_size=None,
        width_size=None,
        depth=None,
        key=None,
        activation_str="relu",
        final_activation_str=None,
        use_bias=True,
        use_final_bias=True,
        name=None,
        filename=None,
        **kwargs,
    ):
        def _match_activation(activation_str):
            # match activation_str:
            #     case "relu":
            #         return jax.nn.relu
            #     case "tanh":
            #         return jnp.tanh
            #     case _:
            #         return lambda x: x
            if activation_str == "relu":
                return jax.nn.relu
            elif activation_str == "sigmoid":
                return jax.nn.sigmoid
            elif activation_str == "tanh":
                return jnp.tanh
            else:
                return lambda x: x

        def _make(
            in_size,
            out_size,
            width_size,
            depth,
            key=None,
            activation_str="relu",
            final_activation_str=None,
            use_bias=False,
            use_final_bias=False,
        ):
            """
            see https://docs.kidger.site/equinox/examples/serialisation/ for rationale
            of implementation here.
            cant serialize the activation function, so we serialize a string representing
            a selection for activation function amongst a finite set of options.
            """
            return eqx.nn.MLP(
                in_size,
                out_size,
                width_size,
                depth,
                key=key,
                activation=_match_activation(activation_str),
                final_activation=_match_activation(final_activation_str),
                use_bias=use_bias,
                use_final_bias=use_final_bias,
            )

        if filename is None:
            self.net = _make(
                input_size,
                output_size,
                width_size,
                depth,
                key=key,
                activation_str=activation_str,
                final_activation_str=final_activation_str,
                use_bias=use_bias,
                use_final_bias=use_final_bias,
            )
            self.activation_str = activation_str
            self.final_activation_str = final_activation_str
        else:
            with open(filename, "rb") as f:
                hyperparams = json.loads(f.readline().decode())
                model = _make(key=jax.random.PRNGKey(0), **hyperparams)
                self.net = eqx.tree_deserialise_leaves(f, model)

        # Keep the NN params in the context, but save static data (functions, etc)
        nn_params, static = eqx.partition(self.net, eqx.is_array)

        # Declare the parameters as block parameters so that they can be
        # updated in-context during optimization.
        block_params = {"nn_params": nn_params}

        def _func(static, inputs, **parameters):
            net = eqx.combine(parameters["nn_params"], static)
            return net(inputs)

        super().__init__(
            partial(_func, static), name=name, parameters=block_params, **kwargs
        )

    def serialize(self, filename):
        nn_params, static = eqx.partition(self.net, eqx.is_array)
        hyperparams = {
            "in_size": nn_params.in_size,
            "out_size": nn_params.out_size,
            "width_size": nn_params.width_size,
            "depth": nn_params.depth,
            "activation_str": self.activation_str,
            "final_activation_str": self.final_activation_str,
            "use_bias": nn_params.use_bias,
            "use_final_bias": nn_params.use_final_bias,
        }
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, self.net)
