import json
import math
import time

import pandas as pd

from typing import List, Union
from .api import Api
from .diagrams import Block
from .global_variables import GlobalVariables
from .log import Log
from .models import Model, ModelParameters, load_model
from .simulation_hashed_file import SimulationHashedFile
from .utils import is_uuid
from .results import SimulationResults, LinearizationResult


class SimulationLogLine:
    def __init__(self, data):
        self._data = data
        self._str: str = ""
        self._parse(data)

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._data}>"

    def _parse(self, data):
        try:
            js_data = data
            if isinstance(data, str):
                js_data = json.loads(data)
            ts = ""
            if js_data.get("timestamp") is not None:
                ts = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(js_data["timestamp"])
                )
                ms = math.floor((js_data["timestamp"] % 1) * 1000)
                ts += f".{ms:03} "
            parsed_line = f"{ts}{js_data.get('level')} {js_data.get('message')}"
            for k, v in js_data.items():
                if k not in ["timestamp", "level", "message"]:
                    parsed_line += f" {k}={v}"
            self._str = parsed_line
        except Exception as e:
            Log.trace(f"Failed to parse log line: {e}")
            self._str = str(data)


class Simulation:
    def __init__(self, data: dict, model: Model):
        self.model = model
        self._data = data
        self._logs: List[SimulationLogLine] = None
        self._results: pd.DataFrame = None

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self) -> str:
        if Log.is_level_above("DEBUG"):
            return (
                f"<{self.__class__.__name__} status='{self.status}' uuid='{self.uuid}'>"
            )
        return f"<{self.__class__.__name__} status='{self.status}'>"

    @property
    def uuid(self) -> str:
        return self._data["uuid"]

    @property
    # this is last status, not latest as it doesn't request to backend.
    def status(self) -> str:
        return self._data["status"]

    @property
    def model_uuid(self) -> str:
        return self._data["model_uuid"]

    @property
    def results_available(self) -> bool:
        return self._data.get("results_available") is True

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_running(self) -> bool:
        return self.status != "created" and not self.is_failed and not self.is_completed

    @property
    def is_started(self) -> bool:
        return self.status != "created"

    @property
    def logs(self) -> List[SimulationLogLine]:
        if self._logs is None:
            logs = Api.simulation_logs(self.model_uuid, self.uuid)
            self._logs = [SimulationLogLine(s) for s in str(logs).splitlines()]
        return self._logs

    @property
    def compilation_logs(self) -> SimulationLogLine:
        logs = self._data.get("compilation_logs")
        if logs is None:
            return None
        if isinstance(logs, list):
            return SimulationLogLine(logs[0])
        return SimulationLogLine(logs)

    def show_logs(self) -> None:
        # It would be awesome to use markdown to pretty-print these logs
        if self.is_failed:
            print(self.compilation_logs)
            return
        for log_line in self.logs:
            print(log_line)

    @property
    def results(self) -> SimulationResults:
        # TODO: cache continuous results
        if not self.results_available:
            Log.warning("Simulation results may not be available yet:", self)
            # But we will try downloading them anyway since that's what the user asked for
        results = Api.simulation_results(self.model_uuid, self.uuid)
        return SimulationResults(results, self.model)

    def _signal_results(self, paths: List[str]) -> SimulationResults:
        if not self.results_available:
            Log.warning("Simulation results may not be available yet:", self)
            # But we will try downloading them anyway since that's what the user asked for
        results_pds = [
            SimulationResults(result, self.model).to_pandas()
            for result in Api.signal_results(self.model_uuid, self.uuid, paths)
        ]
        merged_results_csv = pd.concat(results_pds, axis=1).to_csv()
        return SimulationResults(merged_results_csv, self.model)

    def get_results(
        self, signals: Union[List[str], str] = None, wait=True
    ) -> SimulationResults:
        # :param signals: List of signal paths to fetch results of or a single signal path, defaults to None (all signals)
        # :type signals: List[str] | str, optional
        # :param wait: Whether or not to wait for the simulation to complete, defaults to True
        # :type wait: bool, optional
        # :return: Simulation results of specified signal, or all results if none specified.
        # :rtype: SimulationResults
        """Get simulation results.

        Args:
            signals: List of signal paths to fetch results of or a single signal path, defaults to None (all signals)
            wait: Whether or not to wait for the simulation to complete, defaults to True

        Returns:
            Simulation results of specified signal, or all results if none specified.
        """
        if wait:
            self.wait()
        if signals is not None:
            return self._signal_results(
                [signals] if isinstance(signals, str) else signals
            )
        else:
            return self.results

    def update(self):
        Log.trace("Updating simulation:", self)
        self._data = Api.simulation_get(self.model_uuid, self.uuid)
        return self

    def start(self):
        Log.debug("Starting simulation:", self)
        self._data = Api.simulation_start(self.model_uuid, self.uuid)

    def stop(self):
        Log.debug("Stopping simulation:", self)
        self._data = Api.simulation_stop(self.model_uuid, self.uuid)

    def wait(self):
        if self.is_completed:
            return

        if not self.is_started:
            Log.debug("Simulation not started:", self)
            self.start()

        start_ts = time.time()
        while self.is_running:
            Log.debug(
                f"Waiting for simulation {math.trunc(time.time() - start_ts)}s:",
                self,
            )
            self.update()
            if self.status == "completed":
                break
            time.sleep(1)
        Log.debug("Simulation completed:", self)

    def to_pandas(self, wait=True):
        return self.get_results(wait=wait).to_pandas()

    def _upload_hashed_files(self, model, simulation):
        blocks = model._get_blocks_with_data()
        for block_uuid in blocks:
            input_data = blocks[block_uuid]
            sdf = SimulationHashedFile(input_data)
            resp = sdf.upload(model.uuid, simulation.uuid)
            Log.trace(f"Uploaded hashed file for block '{block_uuid}':", resp)


def _run_simulation_common(
    model,
    linearization,
    parameters,
    wait,
    no_sync,
    ignore_cache,
    snapshot_uuid,
    recorded_signals: list = None,
) -> Simulation:
    Log.trace("run_simulation:", model.__repr__())
    if not isinstance(model, Model):
        model = load_model(model)

    if not no_sync:
        model.sync()

    if linearization is not None:
        if is_uuid(linearization):
            linearization = {"submodel_uuid": linearization}
        elif isinstance(linearization, Block):
            linearization = {"submodel_uuid": linearization.uuid}
        else:
            raise ValueError(("Invalid argument 'linearization'"))

    overrides = {}

    # Prepare DataSource files
    hashed_files = model._get_hashed_files()
    block_overrides = []
    for name_path, hashed_file in hashed_files.items():
        block_override = {
            "path": name_path,
            "parameters": {
                "file_name": {
                    "value": f'__hashed_file("{hashed_file.hash}", "{hashed_file.content_type}")'
                }
            },
        }
        block_overrides.append(block_override)

    if len(block_overrides) > 0:
        overrides["block_overrides"] = block_overrides

    # List of selected blocks
    signal_ids = []
    for signal_id in recorded_signals or []:
        if isinstance(signal_id, Block):
            sid = signal_id.path
        elif isinstance(signal_id, str):
            sid = signal_id
        else:
            raise ValueError(
                f"Invalid argument 'recorded_signals', expected Block or str, got {type(signal_id)}: {signal_id}"
            )
        signal_ids.append(sid)

    if len(signal_ids) > 0:
        Log.trace("Recorded signals:", signal_ids)
        recorded_signals = {"signal_ids": signal_ids}
        overrides["recorded_signals"] = recorded_signals

    # Send simulation specific parameters
    parameters = {
        **model.parameters.to_api_data(),
        **(ModelParameters(parameters or {}).to_api_data()),
    }

    # Simulation-specific configuration
    configuration = model.configuration.to_dict()

    # Target (hack for experiments)
    target = GlobalVariables._get_instance().target or "pycollimator"

    sim_request = {
        "version": model.version,
        "no_start": True,
        "configuration": configuration,
        "model_overrides": overrides,
        "parameters": parameters,
        "ignore_cache": ignore_cache,
        "target": target,
    }

    if linearization is not None:
        sim_request["linearization"] = linearization

    if snapshot_uuid is not None:
        sim_request["snapshot_uuid"] = snapshot_uuid

    sim = Simulation(Api.simulation_create(model.uuid, sim_request), model)

    if sim.status == "created":
        # Upload DataSource files and then update the ModelOverrides
        sim._upload_hashed_files(model, sim)
        sim.start()

    if wait:
        sim.wait()

    if sim.is_failed:
        Log.error(("Simulation failed! Use .show_logs() to see the logs."))

    return sim


def run_simulation(
    model: Union[Model, str],
    parameters: dict = None,
    wait=True,
    no_sync=False,
    ignore_cache=False,
    snapshot_uuid:str=None,
    recorded_signals: list[str] = None,
) -> Simulation:
    """
    Run a simulation remotely on the given model.

    Args:
        model: Model or model UUID
        parameters: Simulation parameters
        wait: Wait for the simulation to complete
        no_sync: Do not sync the model before running the simulation
        ignore_cache: Ignore the cache and run the simulation from scratch
        snapshot_uuid: Use a snapshot of the model
        recorded_signals: List of signals to record (Block or signal id)
    """
    return _run_simulation_common(
        model,
        None,
        parameters,
        wait,
        no_sync,
        ignore_cache,
        snapshot_uuid,
        recorded_signals,
    )


def linearize(
    model,
    submodel,
    parameters: dict = None,
    no_sync=False,
    ignore_cache=False,
    snapshot_uuid=None,
):
    """
    Linearize the submodel within a model.
    """
    if not isinstance(model, Model):
        model = load_model(model)

    if not isinstance(submodel, Block):
        submodel = model.find_block(submodel)

    sim = _run_simulation_common(
        model, submodel, parameters, True, no_sync, ignore_cache, snapshot_uuid
    )

    if not sim.is_completed:
        Log.error(("Simulation could not complete!"))
        sim.show_logs()
        return sim

    lin_results_csv = Api.linearization_results_csv(model.uuid, sim.uuid)
    res = LinearizationResult._from_csv(lin_results_csv)
    return res
