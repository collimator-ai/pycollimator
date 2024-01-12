from ..framework import LeafSystem, DependencyTicket
from ..math_backend import asarray

__all__ = [
    "SourceBlock",
    "FeedthroughBlock",
    "ReduceBlock",
]


def _add_parameters(sys, parameters):
    """Add parameters to a system.

    These will appear in created contexts, hence are optimizable via
    differentiation through simulations.
    """
    for key, val in parameters.items():
        # Check if the parameter is convertible to an array
        #   If not, it will be stored as an arbitrary object and it's up
        #   to the user to handle it appropriately.
        try:
            asarray(val)
            param_as_array = True
        except TypeError:
            param_as_array = False

        sys.declare_parameter(key, val, as_array=param_as_array)


class SourceBlock(LeafSystem):
    """Simple blocks with a single time-dependent output"""

    def __init__(
        self,
        func,
        name=None,
        system_id=None,
        parameters={},
    ):
        super().__init__(name=name, system_id=system_id)

        _add_parameters(self, parameters)

        def _callback(context):
            parameters = context[self.system_id].parameters
            # parameters = self.get_from_root(context).parameters.as_dict()
            return func(context.time, **parameters)

        self.declare_output_port(
            _callback,
            name="out_0",
            prerequisites_of_calc=[DependencyTicket.time],
        )


class FeedthroughBlock(LeafSystem):
    """Simple feedthrough blocks with a function of a single input"""

    def __init__(
        self,
        func,
        name=None,
        system_id=None,
        parameters={},
    ):
        super().__init__(
            name=name,
            system_id=system_id,
        )
        self.declare_input_port()

        _add_parameters(self, parameters)

        def _callback(context):
            inputs = self.eval_input(context)
            parameters = context[self.system_id].parameters
            # parameters = self.get_from_root(context).parameters.as_dict()
            return func(inputs, **parameters)

        self.declare_output_port(
            _callback, prerequisites_of_calc=[self.input_ports[0].ticket]
        )


class ReduceBlock(LeafSystem):
    def __init__(
        self,
        n_in,
        op,
        name=None,
        system_id=None,
        parameters={},
    ):
        super().__init__(
            name=name,
            system_id=system_id,
        )

        _add_parameters(self, parameters)

        for i in range(n_in):
            self.declare_input_port()

        def _compute_output(context):
            inputs = self.collect_inputs(context)
            parameters = context[self.system_id].parameters
            # parameters = self.get_from_root(context).parameters.as_dict()
            return op(inputs, **parameters)

        self.declare_output_port(_compute_output)
