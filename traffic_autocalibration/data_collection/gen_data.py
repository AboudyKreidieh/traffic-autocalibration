"""Example of a merge network with human-driven vehicles.

In the absence of autonomous vehicles, the network exhibits properties of
convective instability, with perturbations propagating upstream from the merge
point before exiting the network.
"""

import numpy as np
import random
from flow.core.experiment import Experiment
from flow.core.util import ensure_dir
from envs import merge_env, grid_env, ring_env
from exps import FlowDensityExperiment, DataGenerationExperiment
from collections import defaultdict
import csv


CF_PARAMS = [
    {'v0': 70.0, 'T': 1.12, 'a': 1.23, 'b': 3.10, 'delta': 4.0, 's0': 0.5,
     'noise': 0.1},
    {'v0': 70.0, 'T': 1.44, 'a': 0.973, 'b': 0.993, 'delta': 4.0, 's0': 0.5,
     'noise': 0.1},
    {'v0': 16.1, 'T': 1.31, 'a': 1.56, 'b': 0.626, 'delta': 4.0, 's0': 0.5,
     'noise': 0.1},
    # {'v0': 100/3, 'T': 1.00, 'a': 1.00, 'b': 1.50, 'delta': 4.0, 's0': 2.0,
    #  'noise': 0.1},
    # {'v0': 15.0, 'T': 1.00, 'a': 1.00, 'b': 1.50, 'delta': 4.0, 's0': 2.0,
    #  'noise': 0.1}
]


MERGE_RANGE = [
    [1000, 1750],
    [1000, 1750],
    [1000, 1750],
]

GRID_RANGE = [
    [1500, 2500],
    [1500, 2500],
    [1500, 2500],
]

RING_RANGE = [
    [200, 500],
    [200, 500],
    [200, 500],
]

TOTAL_STEPS = 10e6


if __name__ == "__main__":
    # ----------------------------------------------------------------------- #
    # ------------------- Used to find appropriate ranges ------------------- #
    # ----------------------------------------------------------------------- #

    # # for cf params
    # for i, param in enumerate(CF_PARAMS):
    #     # merge environment for inflows from 1000-3000
    #     for inflow in np.arange(1000, 3100, 100):
    #         env = merge_env(inflow, param, i)
    #         exp = FlowDensityExperiment(env)
    #         exp.run(10, 1000)
    #
    #     # intersection environment for inflows from 400-3000
    #     for inflow in np.arange(400, 2100, 100):
    #         env = grid_env(inflow, param, i)
    #         exp = FlowDensityExperiment(env)
    #         exp.run(10, 1000)
    #
    #     # intersection environment for inflows from 400-3000
    #     for length in np.arange(300, 510, 10):
    #         env = ring_env(length, param, i)
    #         exp = FlowDensityExperiment(env)
    #         exp.run(10, 1000)

    # ----------------------------------------------------------------------- #
    # ---------------------- Used for data collection ----------------------- #
    # ----------------------------------------------------------------------- #
    #
    # for cf params
    for i, param in enumerate(CF_PARAMS):
        # create necessary directories
        ensure_dir('./data/'.format(i))

        # # merge environment
        # data = defaultdict(list)
        # while len(data.get('speed', [])) < TOTAL_STEPS:
        #     inflow = random.uniform(MERGE_RANGE[i][0], MERGE_RANGE[i][1])
        #     env = merge_env(inflow, param, i, False)
        #     exp = DataGenerationExperiment(env, 'merge')
        #
        #     # collect new samples
        #     new_data = exp.run(1, 30000)
        #
        #     # extend the new data
        #     for key in new_data.keys():
        #         data[key].extend(new_data[key])
        #
        #     print("Progress: {} %".format(
        #             len(data.get('speed', [])) / TOTAL_STEPS * 100))
        #
        # # save data to csv
        # with open("./data/merge-{}.csv".format(i), "w") as outfile:
        #     writer = csv.writer(outfile)
        #     writer.writerow(data.keys())
        #     writer.writerows(zip(*data.values()))

        # intersection environment for inflows from 400-3000
        data = defaultdict(list)
        while len(data.get('speed', [])) < TOTAL_STEPS:
            inflow = random.uniform(GRID_RANGE[i][0], GRID_RANGE[i][1])
            env = grid_env(inflow, param, i, False)
            exp = DataGenerationExperiment(env, 'grid')

            # collect new samples
            new_data = exp.run(1, 50000)

            # extend the new data
            for key in new_data.keys():
                data[key].extend(new_data[key])

            print("Progress: {} %".format(
                    len(data.get('speed', [])) / TOTAL_STEPS * 100))

        # save data to csv
        with open("./data/grid-{}.csv".format(i), "w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data.keys())
            writer.writerows(zip(*data.values()))

        # # ring environment
        # data = defaultdict(list)
        # while len(data.get('speed', [])) < TOTAL_STEPS:
        #     length = random.uniform(RING_RANGE[i][0], RING_RANGE[i][1])
        #     flag = 0
        #     while flag == 0:
        #         try:
        #             env = ring_env(length, param, i, False)
        #             exp = DataGenerationExperiment(env, 'ring')
        #             # collect new samples
        #             new_data = exp.run(1, 3000)
        #             flag = 1
        #         except Exception as e:
        #             print(e)
        #             flag = 0
        #
        #     # extend the new data
        #     for key in new_data.keys():
        #         data[key].extend(new_data[key])
        #
        #     print("Progress: {} %".format(
        #             len(data.get('speed', [])) / TOTAL_STEPS * 100))
        #
        # # save data to csv
        # with open("./data/ring-{}.csv".format(i), "w") as outfile:
        #     writer = csv.writer(outfile)
        #     writer.writerow(data.keys())
        #     writer.writerows(zip(*data.values()))
