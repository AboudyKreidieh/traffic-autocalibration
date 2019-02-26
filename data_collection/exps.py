from flow.core.util import ensure_dir
from flow.core.experiment import Experiment
import csv
import numpy as np


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
