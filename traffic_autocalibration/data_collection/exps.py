from flow.core.util import ensure_dir
from flow.core.experiment import Experiment
import csv
import numpy as np
from collections import defaultdict
from copy import deepcopy


class FlowDensityExperiment(Experiment):

    def run(self, num_runs, num_steps, rl_actions=None):
        """Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        num_steps : int
            number of steps to be performs in each run of the experiment
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step
        """
        if rl_actions is None:
            def rl_actions(*_):
                return None

        dir_path = './flow_density/{}'.format(self.env.scenario.orig_name)
        ensure_dir(dir_path)
        res = {'vels': [], 'outflows': [], 'densities': []}
        for i in range(num_runs):
            vel = []
            outflow = []
            density = []
            state = self.env.reset()
            for j in range(num_steps):
                state, reward, done, _ = self.env.step(rl_actions(state))
                vel.append(np.mean(
                    self.env.k.vehicle.get_speed(self.env.k.vehicle.get_ids()))
                )
                outflow.append(
                    self.env.k.vehicle.get_outflow_rate(time_span=30)
                )
                density.append(
                    self.env.k.vehicle.num_vehicles /
                    self.env.k.scenario.length()
                )
                if done:
                    break
            res['vels'] = vel
            res['outflows'] = outflow
            res['densities'] = density
            print("Round {0}, done".format(i))

            with open("{}/{}.csv".format(dir_path, i), "w") \
                    as outfile:
                writer = csv.writer(outfile)
                writer.writerow(res.keys())
                writer.writerows(zip(*res.values()))

        self.env.terminate()

        return None


MAX_DATAPOINTS = 50000
MIN_ACCEL = -5


def get_key(item):
    return item[0]


class DataGenerationExperiment(Experiment):

    def __init__(self, env, type):
        super().__init__(env)
        self.prev_speed = {}
        self.data = defaultdict(list)
        self.index = {}
        self.type = type

    def run(self, num_runs, num_steps, rl_actions=None):
        """Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        num_steps : int
            number of steps to be performs in each run of the experiment
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step
        """
        for i in range(num_runs):
            self.env.reset()
            for j in range(num_steps):
                state, reward, done, _ = self.env.step(None)
                if self.data_length < MAX_DATAPOINTS:
                    self.append_data()
                else:
                    break
            print("Round {0}, done".format(i))

        self.env.terminate()

        for key in self.data.keys():
            self.data[key] = np.asarray(self.data[key])

        # remove all None values
        good_indices = np.logical_and(self.data['accel'] != None,
                                      self.data['prev_accel'] != None)

        # remove values with large decelerations
        good_indices[good_indices] = np.logical_and(
            self.data['accel'][good_indices] > MIN_ACCEL,
            self.data['prev_accel'][good_indices] > MIN_ACCEL)

        # implement the removal for all keys in the dataset
        for key in self.data.keys():
            self.data[key] = self.data[key][good_indices]

        return self.data

    @property
    def data_length(self):
        return len(self.data.get('speed', []))

    def append_data(self):
        k = self.env.k

        for veh_id in k.vehicle.get_ids():
            # append the time
            self.data['time'].append(self.env.time_counter)

            # append the speed
            self.data['speed'].append(k.vehicle.get_speed(veh_id))

            # append the acceleration
            if veh_id in self.prev_speed:
                accel = (k.vehicle.get_speed(veh_id) -
                         self.prev_speed[veh_id]) / self.env.sim_step
            else:
                accel = None

            self.data['accel'].append(accel)

            # append the previous acceleration
            self.data['prev_accel'].append(None)
            if veh_id in self.index:
                self.data['prev_accel'][self.index[veh_id]] = accel

            # leader and follower data
            self.data['headway'].append(k.vehicle.get_headway(veh_id))
            self.data['tailway'].append(k.vehicle.get_headway(
                k.vehicle.get_follower(veh_id)))
            self.data['lead_speed'].append(k.vehicle.get_speed(
                k.vehicle.get_leader(veh_id)))
            self.data['follower_speed'].append(k.vehicle.get_speed(
                k.vehicle.get_follower(veh_id)))

        # collect previous speed values to compute the acceleration
        num_vehicles = k.vehicle.num_vehicles
        dist_to_intersection = {}
        self.prev_speed = {}
        self.index.clear()
        for i, veh_id in enumerate(k.vehicle.get_ids()):
            self.prev_speed[veh_id] = deepcopy(k.vehicle.get_speed(veh_id))
            self.index[veh_id] = self.data_length - num_vehicles + i - 1

            # compute distance to intersection
            if self.type == 'merge':
                dist = self._merge_dist_to_intersection(veh_id)
            elif self.type == 'grid':
                dist = self._grid_dist_to_intersection(veh_id)
            else:
                dist = 0
            self.data['dist_to_intersec'].append(dist)
            dist_to_intersection[veh_id] = dist

        # compute the opponent distance to intersection
        for veh_id in k.vehicle.get_ids():
            if self.type == 'merge':
                dist = self._merge_opp_dist_to_intersection(
                    veh_id, dist_to_intersection)
            elif self.type == 'grid':
                dist = self._grid_opp_dist_to_intersection(
                    veh_id, dist_to_intersection)
            else:
                dist = 0
            self.data['opp_dist_to_intersec'].append(dist)

    def _get_leader(self, edge):
        # in case only one edge is presented
        if type(edge) == str:
            edge = [edge]

        # collect the names of all vehicles within the specified edges
        ids_by_edge = []
        for edge_i in edge:
            ids_by_edge.extend(self.env.k.vehicle.get_ids_by_edge(edge_i))

        # sort the vehicles in the edge by increasing position
        veh = [(self.env.k.vehicle.get_position(veh_id), veh_id)
               for veh_id in ids_by_edge]
        veh = sorted(veh, key=get_key)

        # return the id of the vehicle at the front
        return veh[-1][1]

    def _merge_dist_to_intersection(self, veh_id):
        edge = self.env.k.vehicle.get_edge(veh_id)
        pos = self.env.k.vehicle.get_position(veh_id)

        if edge == "inflow_highway":
            dist = 1100.1 - pos
        elif edge == ":left":
            dist = 1000.1 - pos
        elif edge == "left":
            dist = 1000 - pos
        elif edge == "inflow_merge":
            dist = 200.1 - pos
        elif edge == ":bottom":
            dist = 100.1 - pos
        elif edge == "bottom":
            dist = 100 - pos
        else:
            dist = 0

        return dist

    def _merge_opp_dist_to_intersection(self, veh_id, dist_to_intersection):
        edge = self.env.k.vehicle.get_edge(veh_id)

        if edge == "bottom":
            if len(self.env.k.vehicle.get_ids_by_edge("left")) > 0:
                lead_veh = self._get_leader('left')
            elif len(self.env.k.vehicle.get_ids_by_edge(":left")) > 0:
                lead_veh = self._get_leader(':left')
            elif len(self.env.k.vehicle.get_ids_by_edge("inflow_highway")) > 0:
                lead_veh = self._get_leader('inflow_highway')
            else:
                lead_veh = None
        elif edge == "left":
            if len(self.env.k.vehicle.get_ids_by_edge("bottom")) > 0:
                lead_veh = self._get_leader('bottom')
            elif len(self.env.k.vehicle.get_ids_by_edge(":bottom")) > 0:
                lead_veh = self._get_leader(':bottom')
            elif len(self.env.k.vehicle.get_ids_by_edge("inflow_merge")) > 0:
                lead_veh = self._get_leader('inflow_merge')
            else:
                lead_veh = None
        else:
            lead_veh = None

        dist = dist_to_intersection.get(lead_veh, 0)

        return dist

    def _grid_dist_to_intersection(self, veh_id):
        edge = self.env.k.vehicle.get_edge(veh_id)
        pos = self.env.k.vehicle.get_position(veh_id)

        if edge in ["right0_0", "top0_1", "left1_0", "bot0_0"]:
            dist = 300 - pos
        else:
            dist = 0

        return dist

    def _grid_opp_dist_to_intersection(self, veh_id, dist_to_intersection):
        edge = self.env.k.vehicle.get_edge(veh_id)

        if edge in ["right0_0", "left1_0"]:
            if len(self.env.k.vehicle.get_ids_by_edge("top0_1")) > 0 or \
                    len(self.env.k.vehicle.get_ids_by_edge("bot0_0")) > 0:
                lead_veh = self._get_leader(["top0_1", "bot0_0"])
            else:
                lead_veh = None
        elif edge in ["top0_1", "bot0_0"]:
            if len(self.env.k.vehicle.get_ids_by_edge("right0_0")) > 0 or \
                    len(self.env.k.vehicle.get_ids_by_edge("left1_0")) > 0:
                lead_veh = self._get_leader(["right0_0", "left1_0"])
            else:
                lead_veh = None
        else:
            lead_veh = None

        dist = dist_to_intersection.get(lead_veh, 0)

        return dist
