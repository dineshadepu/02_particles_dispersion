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


n_core = 32
n_thread = n_core * 2
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


class Skillen2013WaterEntryHalfBuoyant(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'skillen_2013_water_entry_half_buoyant'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/skillen_2013_circular_water_entry.py' + backend

        # Base case info
        self.case_info = {
            'rho_500_N_20': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=20,
                rigid_body_rho=500
                ), 'N=20'),

            'rho_500_N_50': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=50,
                rigid_body_rho=500
                ), 'N=50'),

            'rho_500_N_80': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=80,
                rigid_body_rho=500
                ), 'N=80'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       tf=0.16,
                       use_edac=None, nu=1e-6,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])
        t_exp = data[rand_case]['t_exp']
        t_BEM = data[rand_case]['t_BEM']
        t_SPH = data[rand_case]['t_SPH']
        penetration_exp = data[rand_case]['penetration_exp']
        penetration_BEM = data[rand_case]['penetration_BEM']
        penetration_SPH = data[rand_case]['penetration_SPH']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.clf()
        for name in self.case_info:
            t_current = data[name]['t_current']
            penetration_current = data[name]['penetration_current']

            plt.plot(t_current, penetration_current, label=self.case_info[name][1])
        plt.plot(t_exp, penetration_exp, "^", label='Experimental')
        plt.plot(t_SPH, penetration_SPH, "-+", label='delta-plus SPH, N=200')
        # plt.plot(t_BEM, penetration_BEM, "--", label='BEM')
        plt.plot(t_BEM, penetration_BEM, label='BEM')

        plt.xlabel('t (g / D)^{1/2}')
        plt.ylabel('Penetration (y - y_0)/D')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('penetration_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================


class Skillen2013WaterEntryNeutrallyBuoyant(Problem):
    """
    Pertains to Figure 14 (b)
    """
    def get_name(self):
        return 'skillen_2013_water_entry_neutrally_buoyant'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/skillen_2013_circular_water_entry.py' + backend

        # Base case info
        self.case_info = {
            'rho_1000_N_20': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=20,
                rigid_body_rho=1000
                ), 'N=20'),

            'rho_1000_N_50': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=50,
                rigid_body_rho=1000
                ), 'N=50'),

            'rho_1000_N_80': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=80,
                rigid_body_rho=1000
                ), 'N=80'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       tf=0.20,
                       use_edac=None, nu=1e-6,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])
        t_exp = data[rand_case]['t_exp']
        t_BEM = data[rand_case]['t_BEM']
        t_SPH = data[rand_case]['t_SPH']
        penetration_exp = data[rand_case]['penetration_exp']
        penetration_BEM = data[rand_case]['penetration_BEM']
        penetration_SPH = data[rand_case]['penetration_SPH']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.clf()
        for name in self.case_info:
            t_current = data[name]['t_current']
            penetration_current = data[name]['penetration_current']

            plt.plot(t_current, penetration_current, label=self.case_info[name][1])
        plt.plot(t_exp, penetration_exp, "^", label='Experimental')
        plt.plot(t_SPH, penetration_SPH, "-+", label='delta-plus SPH, N=200')
        # plt.plot(t_BEM, penetration_BEM, "--", label='BEM')
        plt.plot(t_BEM, penetration_BEM, label='BEM')

        plt.xlabel('t (g / D)^{1/2}')
        plt.ylabel('Penetration (y - y_0)/D')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('penetration_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================


if __name__ == '__main__':
    PROBLEMS = [
        Skillen2013WaterEntryHalfBuoyant,
        Skillen2013WaterEntryNeutrallyBuoyant,
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
