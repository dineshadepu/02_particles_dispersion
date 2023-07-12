#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import cycle, product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
from pysph.solver.utils import load, get_files
from automan.api import (Automator, Simulation, filter_cases, filter_by_name)

import numpy as np
import matplotlib
matplotlib.use('agg')
from cycler import cycler
from matplotlib import rc, patches, colors
from matplotlib.collections import PatchCollection

rc('font', **{'family': 'Helvetica', 'size': 12})
rc('legend', fontsize='medium')
rc('axes', grid=True, linewidth=1.2)
rc('axes.grid', which='both', axis='both')
# rc('axes.formatter', limits=(1, 2), use_mathtext=True, min_exponent=1)
rc('grid', linewidth=0.5, linestyle='--')
rc('xtick', direction='in', top=True)
rc('ytick', direction='in', right=True)
rc('savefig', format='pdf', bbox='tight', pad_inches=0.05,
   transparent=False, dpi=300)
rc('lines', linewidth=1.5)
rc('axes', prop_cycle=(
    cycler('color', ['tab:blue', 'tab:green', 'tab:red',
                     'tab:orange', 'm', 'tab:purple',
                     'tab:pink', 'tab:gray']) +
    cycler('linestyle', ['-.', '--', '-', ':',
                         (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3, 2, 1, 1)), (0, (3, 2, 2, 1, 1, 1)),
                         ])
))


n_core = 24
n_thread = 24 * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


class Vyas2021ReboundKinematics3d(Problem):
    def get_name(self):
        return 'vyas_2021_rebound_kinematics_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/vyas_2021_rebound_kinematics_3d.py' + backend

        samples = 5000
        # E = 70 * 1e9
        # nu = 0.3
        # G = E / (2. * (1. + nu))
        # E_star = E / (2. * (1. - nu**2.))

        fric_coeff = 0.1
        kr = 1e8
        kf = 1e4

        dt = 1e-6
        # Base case info
        self.case_info = {
            'angle_2': (dict(
                samples=samples,
                velocity=5.,
                angle=2.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=2.'),

            'angle_5': (dict(
                samples=samples,
                velocity=5.,
                angle=5.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=5.'),

            'angle_10': (dict(
                samples=samples,
                velocity=5.,
                angle=10.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=10.'),

            'angle_15': (dict(
                samples=samples,
                velocity=5.,
                angle=15.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=15.'),

            'angle_20': (dict(
                samples=samples,
                velocity=5.,
                angle=20.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=20.'),

            'angle_25': (dict(
                samples=samples,
                velocity=5.,
                angle=25.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=25.'),

            'angle_30': (dict(
                samples=samples,
                velocity=5.,
                angle=30.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=30.'),

            'angle_35': (dict(
                samples=samples,
                velocity=5.,
                angle=35.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=35.'),

            'angle_40': (dict(
                samples=samples,
                velocity=5.,
                angle=40.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=40.'),

            'angle_45': (dict(
                samples=samples,
                velocity=5.,
                angle=45.,
                kr=kr,
                kf=kf,
                fric_coeff=fric_coeff,
                timestep=dt,
                ), 'Angle=45.'),

            # 'angle_60': (dict(
            #     samples=samples,
            #     velocity=5.,
            #     angle=60.,
            #     fric_coeff=fric_coeff,
            #     ), 'Angle=60.'),

            # 'angle_70': (dict(
            #     samples=samples,
            #     velocity=5.,
            #     angle=70.,
            #     fric_coeff=fric_coeff,
            #     ), 'Angle=70.'),

            # 'angle_80': (dict(
            #     samples=samples,
            #     velocity=5.,
            #     angle=80.,
            #     fric_coeff=fric_coeff,
            #     ), 'Angle=80.'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       contact_force_model="Mohseni",
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_theta_vs_omega()

    def plot_theta_vs_omega(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))
            theta_exp = data[name]['theta_exp']
            omega_exp = data[name]['omega_exp']

        non_dim_theta = []
        non_dim_omega = []

        for name in self.case_info:
            non_dim_theta.append(data[name]['non_dim_theta'])
            non_dim_omega.append(data[name]['non_dim_omega'])

        plt.plot(non_dim_theta, non_dim_omega, '^-', label='Simulated')
        plt.plot(theta_exp, omega_exp, 'v-', label='Thornton et al. (Exp)')
        plt.xlabel(r'$\theta^{*}_i$')
        plt.ylabel(r'$\omega^{*}_r$')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('theta_vs_omega.pdf'))
        plt.clf()
        plt.close()

        # save the data for comparison between primary and secondary flip
        path = os.path.abspath(__file__)
        tmp = os.path.dirname(path)

        new_directory = os.path.join(tmp, "outputs",
                                     "vyas_2021_rebound_kinematics_3d_compare_flipped")

        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        res = os.path.join(new_directory, "vyas_no_flipped")

        np.savez(res,
                 theta_exp=theta_exp,
                 omega_exp=omega_exp,
                 non_dim_theta=non_dim_theta,
                 non_dim_omega=non_dim_omega)


if __name__ == '__main__':
    PROBLEMS = [
        Vyas2021ReboundKinematics3d,  # Done
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
