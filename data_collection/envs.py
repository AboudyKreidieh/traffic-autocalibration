"""Example of a merge network with human-driven vehicles.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.
"""

from flow.core.params import SumoParams, EnvParams, \
    NetParams, InitialConfig, InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.scenarios.merge import MergeScenario
from flow.scenarios.merge import ADDITIONAL_NET_PARAMS as MERGE_NET_PARAMS
from flow.controllers import IDMController, ContinuousRouter, GridRouter
from flow.envs.merge import WaveAttenuationMergePOEnv
from flow.envs.merge import ADDITIONAL_ENV_PARAMS as MERGE_ENV_PARAMS
from flow.envs.loop.loop_accel import AccelEnv
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS as LOOP_ENV_PARAMS
from flow.scenarios.loop import LoopScenario
from flow.scenarios.loop import ADDITIONAL_NET_PARAMS as LOOP_NET_PARAMS
from flow.scenarios.grid import SimpleGridScenario


def merge_env(inflow_rate, cf_params, num, render=None):
    """
    Perform a simulation of vehicles on a merge.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a merge.
    """
    sim_params = SumoParams(
        render=False,
        sim_step=0.2,
        restart_instance=True)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, cf_params),
        car_following_params=SumoCarFollowingParams(
            speed_mode=9,
        ),
        num_vehicles=5)

    env_params = EnvParams(
        additional_params=MERGE_ENV_PARAMS,
        sims_per_step=5,
        warmup_steps=0)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=inflow_rate,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="inflow_merge",
        vehs_per_hour=100,
        departLane="free",
        departSpeed=7.5)

    additional_net_params = MERGE_NET_PARAMS.copy()
    additional_net_params["merge_lanes"] = 1
    additional_net_params["highway_lanes"] = 1
    additional_net_params["pre_merge_length"] = 1000
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="uniform", perturbation=5.0)

    scenario = MergeScenario(
        name="merge-{}-{}".format(inflow_rate, num),
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    return WaveAttenuationMergePOEnv(env_params, sim_params, scenario)


def ring_env(length, cf_params, num, render=None):
    """
    Perform a simulation of vehicles on a ring road.

    Parameters
    ----------
    render : bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on a ring road.
    """
    sim_params = SumoParams(sim_step=0.1, render=False)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, cf_params),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=22)

    env_params = EnvParams(additional_params=LOOP_ENV_PARAMS)

    additional_net_params = LOOP_NET_PARAMS.copy()
    additional_net_params['length'] = length
    net_params = NetParams(additional_params=additional_net_params)

    initial_config = InitialConfig(spacing='random')

    scenario = LoopScenario(
        name="ring-{}-{}".format(length, num),
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config)

    return AccelEnv(env_params, sim_params, scenario)


def grid_env(inflow_rate, cf_params, num, render=None):
    """
    Perform a simulation of vehicles on a grid.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use the gui during execution

    Returns
    -------
    exp: flow.core.experiment.Experiment
        A non-rl experiment demonstrating the performance of human-driven
        vehicles and balanced traffic lights on a grid.
    """
    inner_length = 300
    long_length = 500
    short_length = 300
    N_ROWS = 1
    N_COLUMNS = 1
    num_cars_left = 2
    num_cars_right = 2
    num_cars_top = 2
    num_cars_bot = 2
    tot_cars = (num_cars_left + num_cars_right) * N_COLUMNS \
        + (num_cars_top + num_cars_bot) * N_ROWS

    grid_array = {
        "short_length": short_length,
        "inner_length": inner_length,
        "long_length": long_length,
        "row_num": N_ROWS,
        "col_num": N_COLUMNS,
        "cars_left": num_cars_left,
        "cars_right": num_cars_right,
        "cars_top": num_cars_top,
        "cars_bot": num_cars_bot
    }

    sim_params = SumoParams(
        restart_instance=True,
        sim_step=0.5,
        render=False)

    if render is not None:
        sim_params.render = render

    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, cf_params),
        routing_controller=(GridRouter, {}),
        car_following_params=SumoCarFollowingParams(
            min_gap=2.5,
            decel=10.5,  # avoid collisions at emergency stops
            speed_mode=9,
        ),
        num_vehicles=tot_cars)

    env_params = EnvParams(
        sims_per_step=2,
        additional_params=LOOP_ENV_PARAMS)

    inflow = InFlows()
    inflow.add(
        veh_type="human",
        edge="bot0_0",
        vehs_per_hour=inflow_rate/4,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="right0_0",
        vehs_per_hour=inflow_rate/4,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="top0_1",
        vehs_per_hour=inflow_rate/4,
        departLane="free",
        departSpeed=10)
    inflow.add(
        veh_type="human",
        edge="left1_0",
        vehs_per_hour=inflow_rate/4,
        departLane="free",
        departSpeed=10)

    additional_net_params = {
        "grid_array": grid_array,
        "speed_limit": 35,
        "horizontal_lanes": 1,
        "vertical_lanes": 1,
        "traffic_lights": False,
    }
    net_params = NetParams(
        inflows=inflow,
        no_internal_links=False,
        additional_params=additional_net_params)

    initial_config = InitialConfig(spacing='custom')

    scenario = SimpleGridScenario(
        name="grid-{}-{}".format(inflow_rate, num),
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        # traffic_lights=tl_logic
    )

    return AccelEnv(env_params, sim_params, scenario)
