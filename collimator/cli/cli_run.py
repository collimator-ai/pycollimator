#!env python
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


import functools
import json
import os

import click

import collimator
from collimator.logging import logger


def _show_progress(start: float, end: float, current: float):
    elapsed = current - start
    total = end - start
    percent = 100 * elapsed / total
    logger.info(f"Progress: {percent:.1f}% ({elapsed:.3f}/{total:.3f})")


@click.command(
    help="Run a serialized collimator model from the command line.\n\n\n"
    + "This tool is intended for test purposes. Compatibility with "
    + "serialized model.json is not guaranteed.",
    name="run",
)
@click.option(
    "--model",
    default="model.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    show_default=True,
    help="Path to model JSON to load and run",
)
@click.option(
    "--signalsdir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory for signal output files (npy format)",
)
@click.option(
    "--logsdir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory for Wildcat log files and timing statistics",
)
@click.option(
    "--no-output",
    is_flag=True,
    help="Disable all output recording (for performance testing)",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Stop after check (before JIT)",
)
@click.option(
    "--end_time",
    default=None,
    type=float,
    help="Override end time of the simulation",
)
@click.option(
    "--progress",
    is_flag=True,
    help="Show progress during simulation",
)
def collimator_run(
    model: os.PathLike,
    logsdir=None,
    signalsdir=None,
    no_output=False,
    end_time=None,
    progress=False,
    check_only=False,
):
    if no_output and signalsdir is not None:
        click.echo("Disabling all output recording (ignores signalsdir)")
        signalsdir = None

    logger.info('Loading "%s"... (pid: %s)', model, os.getpid())

    modeldir = os.path.dirname(model)
    app = collimator.load_model(
        modeldir, model=model, npydir=signalsdir, logsdir=logsdir, check=False
    )
    logger.info("Checking model with %s...", app.simulator_options.math_backend)
    app.check(write_signals_json=logsdir is not None)

    if check_only:
        return

    if end_time is not None:
        app.sim_context.stop_time = float(end_time)

    if progress:
        _show_progress(0, app.sim_context.stop_time, 0.0)
        app.simulator_options.major_step_callback = functools.partial(
            _show_progress, 0.0, app.sim_context.stop_time
        )

    logger.info("Starting simulation...")
    _ = app.simulate(
        start_time=app.sim_context.start_time, stop_time=app.sim_context.stop_time
    )


# Alternative top-level function that is somewhat friendlier for use from
# the debugger.
def run(modeldir, model_json, output_dir):
    # workdir = output_dir
    npydir = os.path.join(output_dir, "npy")
    logsdir = os.path.join(output_dir, "logs")
    os.makedirs(npydir, exist_ok=True)
    os.makedirs(logsdir, exist_ok=True)

    # FIXME: workdir=workdir not passed
    model = collimator.load_model(modeldir, model_json, npydir=npydir, logsdir=logsdir)

    with open(os.path.join(modeldir, model_json), encoding="utf-8") as f:
        model_spec = json.load(f)
        model_config = model_spec["configuration"]

        solver_config = model_config["solver"]
        stop_time = model_config["stop_time"]
        # sample_time = model_config["sample_time"]  # <-- do we need this here?

        simulator_options = collimator.SimulatorOptions(
            rtol=solver_config["relative_tolerance"],
            atol=solver_config["absolute_tolerance"],
            min_minor_step_size=solver_config["min_step"],
            max_minor_step_size=solver_config["max_step"],
            # TODO: the default here is `variable_step`, but should be a specific method name
            # method=solver_config["type"],
        )

    model.simulate(
        stop_time=stop_time,
        simulator_options=simulator_options,
    )
