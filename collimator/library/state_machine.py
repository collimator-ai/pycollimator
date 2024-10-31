# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

"""
StateMachine which meets the specification from design.
"""

import ast
from collections import namedtuple
import dataclasses
from dataclasses import dataclass, field
import inspect
from typing import List, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np

from collimator.framework import parameters
from collimator.framework.state import LeafState
from collimator.framework.error import StaticError, BlockInitializationError
from ..backend import numpy_api as cnp
from ..framework import LeafSystem, DependencyTicket


__all__ = [
    "StateMachine",
]


ALWAYS_TRUE_GUARD_ID = 1
ALWAYS_FALSE_GUARD_ID = 0


@dataclass(frozen=True)
class StateMachineTransition:
    """
    Attribues:
        guard:
            - must be python expression that evaluates to Bool.
            - may contain any number of <input_name>s or <output_name>s
        actions:
            - must be python statement.
            - may contain any number of <input_name>s or <output_name>s
    """

    guard_id: int = (ALWAYS_TRUE_GUARD_ID,)
    dst: int = 0
    action_ids: List[int] = field(default_factory=list)

    def __repr__(self):
        return f"StateMachineTransition(dst={self.dst}, guard_id={self.guard_id}, action_ids={self.action_ids})"


def smt_tree_flatten(smt: StateMachineTransition):
    children = (smt.dst, smt.guard_id, tuple(smt.action_ids))
    return children, None  # No auxiliary data


def smt_tree_unflatten(aux_data, children):
    dst, guard_id, action_ids = children
    return StateMachineTransition(
        dst=dst, guard_id=guard_id, action_ids=list(action_ids)
    )


jax.tree_util.register_pytree_node(
    StateMachineTransition, smt_tree_flatten, smt_tree_unflatten
)


@dataclass
class StateMachineState:
    """Captures all elements of a state"""

    name: str = None
    # ordered list of exit transitions, ordered by priority
    transitions: List[StateMachineTransition] = field(default_factory=list)

    def with_transitions(self, transitions):
        return dataclasses.replace(self, transitions=transitions)


class StateMachineRegistry:
    """This class is used to store the guards and actions of a state machine.

    To be compatible with jax operations, all guards must have the same signature
    and return type. The same is true for actions.
    """

    def __init__(self):
        self._actions = []

        # initial guard is always True, so that the first transition is always taken.
        # padded guards should be False, so that they are never taken.
        self._guards = [lambda *args, **kwargs: False, lambda *args, **kwargs: True]

    @property
    def guards(self):
        return self._guards

    @property
    def actions(self):
        return self._actions

    def register_guard(self, guard: Callable) -> int:
        if _is_lambda_body_true(guard):
            return ALWAYS_TRUE_GUARD_ID
        self._guards.append(guard)
        return len(self._guards) - 1

    def register_action(self, action: Callable) -> int:
        self._actions.append(action)
        return len(self._actions) - 1

    def make_transition(
        self, guard: Callable, dst: int = None, actions: List[Callable] = None
    ):
        if actions is None:
            actions = []
        guard_id = self.register_guard(guard)
        action_ids = [self.register_action(action) for action in actions]
        return StateMachineTransition(guard_id=guard_id, dst=dst, action_ids=action_ids)


@dataclass
class StateMachineData:
    """Captures all elements of a state machine"""

    registry: StateMachineRegistry
    states: Dict[int, StateMachineState]
    initial_state: int
    initial_actions: list[int] = field(default_factory=list)

    def set_unguarded_transition(self, st_idx, trn_idx):
        """Sets the unguarded transition at the end of the list."""
        st = self.states[st_idx]
        transitions_new = st.transitions.copy()
        unguarded_transition = transitions_new.pop(trn_idx)
        transitions_new.append(unguarded_transition)
        st_new = st.with_transitions(transitions=transitions_new)
        states_new = self.states.copy()
        states_new[st_idx] = st_new
        return dataclasses.replace(self, states=states_new)

    def to_padded_arrays(self) -> "StateMachineData":
        """
        Converts the StateMachineData transitions and action_ids lists to padded arrays.
        This is necessary for JIT-compilation with JAX.

        Returns:
            StateMachineData: A new instance with padded transitions and action_ids as arrays.
        """
        # Step 1: Determine maximum lengths
        max_transitions = (
            max(len(state.transitions) for state in self.states.values())
            if self.states
            else 0
        )
        action_ids_lengths = [
            len(transition.action_ids)
            for state in self.states.values()
            for transition in state.transitions
        ] + [len(self.initial_actions)]
        max_action_ids = max(action_ids_lengths) if action_ids_lengths else 0

        # Step 2: Define default transition
        default_transition = StateMachineTransition(
            guard_id=ALWAYS_FALSE_GUARD_ID, dst=-1, action_ids=[-1] * max_action_ids
        )

        # Step 3: Pad transitions and action_ids
        new_states = {}
        for state_id, state in self.states.items():
            # a. Pad transitions
            padded_transitions = []
            for transition in state.transitions:
                # Pad action_ids
                padded_action_ids = transition.action_ids + [-1] * (
                    max_action_ids - len(transition.action_ids)
                )
                padded_transition = StateMachineTransition(
                    guard_id=transition.guard_id,
                    dst=transition.dst,
                    action_ids=padded_action_ids,
                )
                padded_transitions.append(padded_transition)
            # b. Add padding transitions if needed
            num_pads = max_transitions - len(padded_transitions)
            if num_pads > 0:
                padded_transitions += [default_transition] * num_pads
            # c. Convert action_ids to JAX arrays
            padded_transitions = [
                StateMachineTransition(
                    guard_id=trans.guard_id,
                    dst=trans.dst,
                    action_ids=cnp.array(trans.action_ids, dtype=cnp.int32),
                )
                for trans in padded_transitions
            ]
            # d. Create new StateMachineState with JAX arrays
            new_state = StateMachineState(
                name=state.name, transitions=padded_transitions
            )
            new_states[state_id] = new_state

        # Step 4: Pad initial_actions and convert to JAX array
        if self.initial_actions:
            padded_initial_actions = self.initial_actions + [-1] * (
                max_action_ids - len(self.initial_actions)
            )
        else:
            padded_initial_actions = [-1] * max_action_ids
        padded_initial_actions = cnp.array(padded_initial_actions, dtype=cnp.int32)

        # Step 5: Create and return the new StateMachineData instance with JAX arrays
        new_data = StateMachineData(
            registry=self.registry,
            states=new_states,
            initial_state=self.initial_state,
            initial_actions=padded_initial_actions,
        )

        return new_data


def _validate_sm_transition(
    registry,
    st,
    st_idx,
    st_idx_list,
    trn_idx,
    t,
    unguarded_idx,
    invalid_guards,
    invalid_actions,
):
    # validate destination
    if t.dst not in st_idx_list:
        # this can only happen due to manual error when creating sm_data from python
        raise ValueError(
            f"StateMachine state[{st.name},{st_idx}] has exit transition with invalid destination index."
        )

    if t.guard_id >= len(registry.guards) or t.guard_id < 0:
        invalid_guards.append(t.guard_id)
    elif not callable(registry.guards[t.guard_id]):
        invalid_guards.append(t.guard_id)

    elif t.guard_id == ALWAYS_TRUE_GUARD_ID:
        if unguarded_idx is not None:
            msg = f"state[{st.name},{st_idx}] has more than one unguarded exit transition."
            raise ValueError(msg)
        else:
            unguarded_idx = trn_idx

    return unguarded_idx, invalid_guards, invalid_actions


def _raise_invalid_code_error(invalid_code):
    if not invalid_code:
        return

    msg_str_list = ["StateMachine has some guard/actions errors:\n"]
    for st_name, data in invalid_code.items():
        if data["guards"] or data["actions"]:
            msg_str_list.append(
                f"State '{st_name}' exit transitions have the following invalid entries:\n"
            )
        if data["guards"]:
            msg_str_list.append("\tguards:\n")
            for g in data["guards"]:
                msg_str_list.append(f"\t\t{g}\n")
        if data["actions"]:
            msg_str_list.append("\tactions:\n")
            for a in data["actions"]:
                msg_str_list.append(f"\t\t{a}\n")
    raise ValueError("".join(msg_str_list))


def _validate_sm_data(sm: StateMachineData):
    # validate states
    st_idx_list = list(sm.states.keys())
    if not st_idx_list:
        raise ValueError("StateMachine must have at least one state.")

    if not all(isinstance(idx, int) for idx in st_idx_list):
        raise ValueError("StateMachine state indices must be of type Int.")

    if sm.initial_state not in st_idx_list:
        raise ValueError(
            f"StateMachine initial_state index {sm.initial_state} does not correspond to a valid state index."
        )

    # validate transitions
    invalid_code = {}
    for st_idx, st in sm.states.items():
        unguarded_idx = None
        invalid_guards = []
        invalid_actions = []
        for trn_idx, t in enumerate(st.transitions):
            (
                unguarded_idx,
                invalid_guards,
                invalid_actions,
            ) = _validate_sm_transition(
                sm.registry,
                st,
                st_idx,
                st_idx_list,
                trn_idx,
                t,
                unguarded_idx,
                invalid_guards,
                invalid_actions,
            )

        if invalid_guards or invalid_actions:
            invalid_code[st.name] = {
                "guards": invalid_guards,
                "actions": invalid_actions,
            }

        if unguarded_idx is not None:
            sm = sm.set_unguarded_transition(st_idx, unguarded_idx)

    _raise_invalid_code_error(invalid_code)
    return sm


def _choose(idx, arr):
    return cnp.choose(jnp.array([idx]), arr, mode="clip")


def _is_lambda_body_true(f):
    """
    Checks if the body of the given lambda function is the constant True.

    Args:
        f (function): The lambda function to inspect.

    Returns:
        bool: True if the lambda returns True, False otherwise.
    """
    try:
        # Retrieve the source code of the lambda
        source = inspect.getsource(f).strip()
    except (OSError, TypeError):
        # Source code could not be retrieved
        return False

    # Parse the source code into an AST
    try:
        module = ast.parse(source)
    except SyntaxError:
        return False

    # Traverse the AST to find the Lambda node
    for node in ast.walk(module):
        if isinstance(node, ast.Lambda):
            # Check if the body of the lambda is a Constant with value True
            return isinstance(node.body, ast.Constant) and node.body.value is True

    return False


@parameters(static=["accelerate_with_jax"])
class StateMachine(LeafSystem):
    """Finite State Machine similar to Mealy Machine.
    https://en.wikipedia.org/wiki/Mealy_machine

    The state machine can be executed either periodically or by zero_crossings.

    Each state as 0 or more exit transitions. These are prioritized such that
    when 2 exits are simultaneously valid, the higher priority is executed.
    It is not allowed for a state to have more than one exit transition with no
    guard. Guardless exits only make sense in the periodic case.

    Each transitions may have 0 or more actions. Each action is a python
    statement that modifies the value of an output. When a transitions is executed
    (i.e. it's guard evaluates to true), its actions are then processed.

    If 'time' is needed for guards or actions, pass 'time' in from clock block.

    Whether executed periodically or by zero_crossings, the states are constant
    between transitions executions.
    In the zero_crossing case, all guards for transitions exiting the current
    state are continuously checked, and if any 'triggers', then the earlist point
    in time that any guard becomes true is determined, the actions of the earliest
    (and highest priority if multiple trigger simultaneously) guard are executed at
    that time, and the simulation continues afterwards.

    Input ports:
        User specified.

    Output ports:
        User specified.

    Parameters:
        dt:
            Either Float or None.
            When not None, state machine is executed periodically.
            When None, the transitions are monitored by zero_crossing
            events.
        accelerate_with_jax:
            Bool. When True, the actions and guards are JIT-compiled with JAX.
            Default is False.
    """

    def __init__(
        self,
        sm_data: StateMachineData,
        inputs: List[str] = None,  # [name]
        outputs: List[str] = None,  # [name]
        dt=None,
        time_mode: str = "agnostic",
        name: str = None,
        ui_id: str = None,
        accelerate_with_jax: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, ui_id=ui_id)

        if time_mode not in ["discrete", "agnostic"]:
            raise BlockInitializationError(
                f"Invalid time mode '{time_mode}' for PythonScript block", system=self
            )

        if time_mode == "discrete" and dt is None:
            raise BlockInitializationError(
                "When in discrete time mode, dt is required for block", system=self
            )

        if cnp.active_backend == "numpy" and accelerate_with_jax:
            raise BlockInitializationError(
                "Must use JAX numerical backend when accelerate_with_jax=True",
                system=self,
            )

        try:
            sm_data = _validate_sm_data(sm_data)
        except ValueError as e:
            raise StaticError(message=str(e), system=self) from e

        self._accelerate_with_jax = accelerate_with_jax

        if accelerate_with_jax:
            # inputs to many jax functions are expected to be jnp.arrays of
            # same shape, so we pad the arrays to the same shapes.
            self._sm = sm_data.to_padded_arrays()

            self._guards = cnp.array(
                [
                    [t.guard_id for t in self._sm.states[idx].transitions]
                    for idx in self._sm.states.keys()
                ]
            )

            self._dst = cnp.array(
                [
                    [t.dst for t in self._sm.states[idx].transitions]
                    for idx in self._sm.states.keys()
                ]
            )

            self._actions = cnp.array(
                [
                    [t.action_ids for t in self._sm.states[idx].transitions]
                    for idx in self._sm.states.keys()
                ]
            )
        else:
            self._sm = sm_data

        self.time_mode = time_mode
        _is_periodic = time_mode == "discrete"

        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        elif isinstance(outputs, dict):
            outputs = list(outputs.keys())

        # delcare inputs
        self._input_names = inputs
        for name in inputs:
            self.declare_input_port(name)

        self._output_names = outputs

        # Create the default discrete state values
        self._create_discrete_state_type(include_state_idx=_is_periodic)
        default_values = self._create_initial_discrete_state(
            include_state_idx=_is_periodic
        )
        self.declare_discrete_state(default_value=default_values, as_array=False)

        # Declare output ports for each state variable
        def _make_output_callback(o_port_name):
            def _output(time, state, *inputs, **parameters):
                return getattr(state.discrete_state, o_port_name)

            return _output

        for o_port_name in outputs:
            self.declare_output_port(
                _make_output_callback(o_port_name),
                name=o_port_name,
                prerequisites_of_calc=[DependencyTicket.xd],
                requires_inputs=False,
            )

        if _is_periodic:
            # delcare the periodic update event
            self.declare_periodic_update(
                self._discrete_update,
                period=dt,
                offset=dt,
            )
        else:
            # wrap the callback generation so that they do not get overwritten
            # in subsequent calls to declare_zero_crossing()
            def _make_guard_callback(t):
                def _guard(_time, state, *inputs, **parameters):
                    # Inputs are in order of port declaration, so they match `self._input_names`
                    inputs = dict(zip(self._input_names, inputs))
                    # get the values of the outputs as they are presently.
                    outputs = state.discrete_state._asdict()
                    # we do this so that when a guard goes False-True,
                    # it creates a zero-crossing that can be localized in time.
                    g = cnp.where(
                        self._sm.registry.guards[t.guard_id](**inputs, **outputs),
                        1.0,
                        -1.0,
                    )
                    return g

                return _guard

            def _make_reset_callback(t):
                def _reset(_time, state, *inputs, **p):
                    # Inputs are in order of port declaration, so they match `self._input_names`
                    inputs = dict(zip(self._input_names, inputs))
                    # get the values of the outputs as they are presently.
                    outputs = state.discrete_state._asdict()
                    if self._accelerate_with_jax:
                        updated_outputs = self._exec_actions_jax(
                            t.action_ids, inputs, outputs
                        )
                    else:
                        updated_outputs = self._exec_actions(
                            t.action_ids, inputs, outputs
                        )
                    return state.with_discrete_state(
                        value=self.DiscreteStateType(**updated_outputs)
                    )

                return _reset

            # declare zero-crossing driven events and mode
            self.declare_default_mode(self._sm.initial_state)
            self.declare_mode_output()
            for st_idx, st in self._sm.states.items():
                for t in st.transitions:
                    self.declare_zero_crossing(
                        guard=_make_guard_callback(t),
                        reset_map=_make_reset_callback(t),
                        direction="negative_then_non_negative",  # we only care when the guard transitions False->True
                        start_mode=st_idx,
                        end_mode=t.dst,
                    )

    def _create_discrete_state_type(self, include_state_idx=True):
        if include_state_idx:
            # unique identifier for the state machine state variable.
            # FIXME: is there a better way? was not allowed to use leading underscore
            # in data class.
            st_name = "active_state_index"

            if st_name in self._input_names or st_name in self._output_names:
                msg = f"StateMachine {self.name} has port with same name as state {st_name}, this is not allowed."
                raise StaticError(message=msg, system=self)

            self._st_name = st_name

            attribs = [st_name] + self._output_names
        else:
            attribs = self._output_names
        # declare the discrete_state as a namedtuple
        self.DiscreteStateType = namedtuple("DiscreteStateType", attribs)

    def _create_initial_discrete_state(self, include_state_idx=True):
        # execute the entry point actions
        inputs = {n: None for n in self._input_names}  # FIXME: get the inputs
        outputs = {n: None for n in self._output_names}

        initial_outputs = self._exec_actions(self._sm.initial_actions, inputs, outputs)

        # check if any initial_outputs is NaN
        for k, v in initial_outputs.items():
            if np.any(np.isnan(v)):
                msg = (
                    "StateMachine has NaN values in the initial outputs. "
                    "Inputs can't be used in initial actions."
                )
                raise BlockInitializationError(message=msg, system=self)

        # enforce that all outputs have been initialized
        initialized_output_names = set(initial_outputs.keys())
        all_output_names = set(self._output_names)
        uninitialized_output_names = all_output_names.difference(
            initialized_output_names
        )
        if uninitialized_output_names:
            msg = f"StateMachine does not initialize the following output values in the entry point actions: {uninitialized_output_names}"
            raise BlockInitializationError(message=msg, system=self)

        # get and save the output dtype,shape for use in creating the jax.pure_callback
        self.output_port_params = {
            o_port_name: {"dtype": jnp.array(val).dtype, "shape": jnp.array(val).shape}
            for o_port_name, val in initial_outputs.items()
        }

        # prepare the initial state
        if include_state_idx:
            return self.DiscreteStateType(
                active_state_index=self._sm.initial_state,
                **initial_outputs,
            )

        return self.DiscreteStateType(**initial_outputs)

    def _filter_locals(self, local_env):
        # remove any bindings from locals that are not outputs.
        filtered_locals = {}
        for key, value in local_env.items():
            if key in self._output_names:
                filtered_locals[key] = value
        return filtered_locals

    def _exec_actions(self, action_ids, inputs, outputs):
        # execute actions, in context with inputs values, when done
        # all actions, filter out any variable bindings that do
        # not correspond to outputs, then repack as dict of jnp.arrays
        updated_outputs = {}
        for action_id in action_ids:
            if action_id == -1:  # padded actions are -1
                continue
            input_args = [inputs[k] for k in self._input_names]
            output_args = [outputs[k] for k in self._output_names]
            output = self._sm.registry.actions[action_id](*input_args, *output_args)
            updated_outputs.update(output)

        updated_outputs = self._filter_locals(updated_outputs)
        updated_outputs = {k: jnp.array(v) for k, v in updated_outputs.items()}
        return updated_outputs

    def _exec_actions_jax(self, action_ids, inputs, outputs):
        """Execute actions in a JAX-compatible way."""

        def _exec_action(action_id):
            input_args = [inputs[k] for k in self._input_names]
            output_args = [outputs[k] for k in self._output_names]
            return cnp.cond(
                action_id == -1,  # padded actions are -1
                lambda: ({k: v for k, v in outputs.items()}, True),
                lambda: (
                    cnp.switch(
                        action_id,
                        self._sm.registry.actions,
                        *input_args,
                        *output_args,
                    ),
                    False,
                ),
            )

        # TODO: cnp.vmap (implement numpy version)
        action_outputs = jax.vmap(_exec_action)(action_ids)

        def _accumulate_outputs(carry, outputs):
            output, is_pad = outputs
            update = cnp.cond(
                is_pad, lambda: carry, lambda: {k: v for k, v in output.items()}
            )
            carry.update(update)
            return carry, carry

        init = {**outputs}
        updated_outputs, _ = cnp.scan(_accumulate_outputs, init, action_outputs)

        updated_outputs = self._filter_locals(updated_outputs)

        return updated_outputs

    def _numpy_callback(self, present_state_index, inputs, outputs):
        """
        The concept here is to evaluate all possible exit transitions from
        the active state, and then just return the updated (state,output values)
        for the successful transition. In the case no transitions are successful,
        we just return the present state and presen_outputs. Since we have ordered
        the possible transitions in order of priority, executing the lowest index
        successful trasition is 'correct' behavior.

        jax-compatible version of this function is `_jax_callback`.
        """
        # get the active state index, and the possible exit transitions
        present_state_index = int(present_state_index)
        actv_trns = self._sm.states[present_state_index].transitions

        # evaluate the guard for each possible exit transition.
        evaluated_guards = [
            self._sm.registry.guards[transition.guard_id](**inputs, **outputs)
            for transition in actv_trns
        ]

        if np.any(evaluated_guards):
            actv_trn = actv_trns[evaluated_guards.index(True)]
            new_state = actv_trn.dst
            updated_outputs = self._exec_actions(actv_trn.action_ids, inputs, outputs)
            new_outputs = []
            for k in self._output_names:
                output = updated_outputs[k] if k in updated_outputs else outputs[k]
                new_outputs.append(np.array(output))
            retval = [np.array(new_state), new_outputs]
        else:
            outputs = [np.array(outputs[k]) for k in self._output_names]
            retval = [np.array(present_state_index), outputs]

        return retval

    def _jax_callback(self, present_state_index, inputs, outputs):
        """
        The concept here is to evaluate all possible exit transitions from
        the active state, and then just return the updated (state,output values)
        for the successful transition. In the case no transitions are successful,
        we just return the present state and present_outputs. Since we have ordered
        the possible transitions in order of priority, executing the lowest index
        successful trasition is 'correct' behavior.
        """

        active_guards = _choose(present_state_index, self._guards)

        input_args = [inputs[k] for k in self._input_names]
        output_args = [outputs[k] for k in self._output_names]
        # evaluate the guard for each possible exit transition.
        evaluated_guards = cnp.array(
            [
                cnp.switch(
                    guard_id,
                    self._sm.registry.guards,
                    *input_args,
                    *output_args,
                )
                for guard_id in active_guards
            ]
        )

        def on_true():
            # Find the first active transition where the guard is True
            active_dst = _choose(present_state_index, self._dst)
            active_actions = _choose(present_state_index, self._actions)

            if np.size(evaluated_guards) == 0:
                # no guards are True, so we return the present state and outputs
                # note that adding jnp.size(evaluated_guards) > 0 to the cnp.cond
                # still evaluates the true branch, so we need to check the size
                # here.
                new_outputs = [jnp.array(outputs[k]) for k in self._output_names]
                return cnp.array(present_state_index), new_outputs

            idx = cnp.argmax(
                evaluated_guards
            )  # TODO: check that it returns the first index (lowest priority)

            new_state = _choose(idx, active_dst)
            action_ids = _choose(idx, active_actions)

            updated_outputs = self._exec_actions_jax(action_ids, inputs, outputs)
            new_outputs = []
            for k in self._output_names:
                output = updated_outputs[k] if k in updated_outputs else outputs[k]
                new_outputs.append(cnp.array(output))
            return new_state.squeeze(), new_outputs

        def on_false():
            new_outputs = [jnp.array(outputs[k]) for k in self._output_names]
            return cnp.array(present_state_index), new_outputs

        return cnp.cond(
            cnp.any(evaluated_guards),
            on_true,
            on_false,
        )

    def _discrete_update(self, _time, state: LeafState, *inputs, **params):
        # persent state index
        actv_state = state.discrete_state.active_state_index

        # Inputs are in order of port declaration, so they match `self._input_names`
        inputs = dict(zip(self._input_names, inputs))

        # get the values of the outputs as they are presently.
        outputs = {
            key: value
            for key, value in state.discrete_state._asdict().items()
            if key not in {self._st_name}
        }

        if self._accelerate_with_jax:
            new_state, new_outputs = self._jax_callback(actv_state, inputs, outputs)
        else:
            # build jax.pure_callback result_shape_dtypes
            # its a nested list like: [actv_st, [outp0, outp1, ... outpN]]
            result_shape_dtypes = [jax.ShapeDtypeStruct((), jnp.int64)]  # actv_state
            result_shape_dtypes_outps = []
            for var in self._output_names:
                port = self.output_port_params[var]
                result_shape_dtypes_outps.append(
                    jax.ShapeDtypeStruct(port["shape"], np.dtype(port["dtype"]))
                )
            result_shape_dtypes.append(result_shape_dtypes_outps)

            if cnp.active_backend == "numpy":
                new_state, new_outputs = self._numpy_callback(
                    actv_state,
                    inputs,
                    outputs,
                )
            else:
                # TODO: implement jax.custom_jvp to raise useful error when trying
                # to differentiate when accelerate_with_jax is False
                new_state, new_outputs = jax.pure_callback(
                    self._numpy_callback,
                    result_shape_dtypes,
                    actv_state,
                    inputs,
                    outputs,
                )

        outputs_dict = {k: v for k, v in zip(self._output_names, new_outputs)}

        return self.DiscreteStateType(
            active_state_index=new_state,
            **outputs_dict,
        )
