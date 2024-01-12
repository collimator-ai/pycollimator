import click
import lynx
import os
import json

from lynx.simulation import ODESolverOptions


@click.command()
@click.option(
    "--modeldir",
    default=".",
    show_default=True,
    help="[--modeldir <dirname>] Directory which contains all model.json & submodel.json files",
)
@click.option(
    "--model",
    default="model.json",
    show_default=True,
    help="--model model.json",
)
@click.option(
    "--workdir",
    show_default=True,
    default=".",
    help="[--workdir <dirname>] Directory where Lynx runs, contains any inputs files needed for operation",
)
@click.option(
    "--datadir",
    help="[--datadir <dirname>] Directory for signal output files.",
)
@click.option(
    "--logsdir",
    help="[--log-dir <dirname>] Directory for Lynx log files and timing statistics",
)
@click.option(
    "--no-output",
    is_flag=True,
    help="Disable all output recording (for performance testing)",
)
def lynx_cli(
    modeldir, model, workdir=None, datadir=None, logsdir=None, no_output=False
):
    cwd = os.getcwd()

    click.echo(
        "lynx_cli inputs:\n\tcwd={}\n\tmodeldir={}\n\tmodel={}\n\tworkdir={}\n\tdatadir={}\n\tlogsdir={}".format(
            cwd, modeldir, model, workdir, datadir, logsdir
        )
    )

    # collect the
    model_dir_part = os.path.dirname(model)
    if model_dir_part:
        raise ValueError(
            "'model' arg must not contain 'path/to/', only file name, eg 'model.json'"
        )

    # FIXME: workdir=workdir not passed
    model = lynx.load_model(modeldir, model=model, datadir=datadir, logsdir=logsdir)

    _ = model.simulate(
        write_binary_results=(datadir is not None and not no_output),
        record_outputs=not no_output,
    )  # @am. presently do nothing with logs.
    # it is expected that eventually results are written to file by call to
    # simulate()


# Alternative top-level function that is somewhat friendlier for use from
# the debugger.
def run(modeldir, model_json, output_dir):
    # workdir = output_dir
    datadir = os.path.join(output_dir, "data")
    logsdir = os.path.join(output_dir, "logs")
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(logsdir, exist_ok=True)

    # FIXME: workdir=workdir not passed
    model = lynx.load_model(modeldir, model_json, datadir=datadir, logsdir=logsdir)

    with open(os.path.join(modeldir, model_json)) as f:
        model_spec = json.load(f)
        model_config = model_spec["configuration"]

        solver_config = model_config["solver"]
        stop_time = model_config["stop_time"]
        # sample_time = model_config["sample_time"]  # <-- do we need this here?

        ode_options = ODESolverOptions(
            rtol=solver_config["relative_tolerance"],
            atol=solver_config["absolute_tolerance"],
            min_step_size=solver_config["min_step"],
            max_step_size=solver_config["max_step"],
            # TODO: the default here is `variable_step`, but should be a specific method name
            # method=solver_config["type"],
        )

    model.simulate(
        t=stop_time,
        ode_options=ode_options,
        write_binary_results=True,
    )


if __name__ == "__main__":
    lynx_cli(".", "model.json")
