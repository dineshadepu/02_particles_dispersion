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


n_core = 6
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
                       use_edac=None, nu=1e-6, max_s=1,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        # self.plot_displacement()
        conditions = {'d0': 1e-2 * 0.55}

        # # schematic
        # self.plot_prop(conditions=conditions, size=0.1,
        #                show_fluid=True, fmin=0, fmax=18000, show_structure=True,
        #                smin=-3*1e6, smax=3*1e6, fcmap='rainbow', show_fcmap=False,
        #                scmap='winter', show_fsmap=False, times=[0],
        #                fname_prefix="schematic", only_colorbar=False)

        # self.plot_prop(conditions=conditions, size=0.1,
        #                show_fluid=True, fmin=0, fmax=18000, show_structure=False,
        #                smin=-3*1e6, smax=3*1e6, fcmap='rainbow', show_fcmap=True,
        #                scmap='winter', show_fsmap=False, times=[0],
        #                fname_prefix="colorbar", only_colorbar=True)

        # # snapshots at different timesteps
        # self.plot_prop(conditions=conditions, size=0.2,
        #                show_fluid=True, fmin=0, fmax=18000, show_structure=True,
        #                smin=-3*1e6, smax=3*1e6, fcmap='rainbow', show_fcmap=False,
        #                scmap='viridis', show_fsmap=False, times=[0.3],
        #                fname_prefix="snap", only_colorbar=False)

        # # save colorbar
        # self.plot_prop(conditions=conditions, size=0.2,
        #                show_fluid=True, fmin=0, fmax=18000, show_structure=True,
        #                smin=-3*1e6, smax=3*1e6, fcmap='rainbow', show_fcmap=True,
        #                scmap='viridis', show_fsmap=True, times=[0.3],
        #                fname_prefix="colorbar", only_colorbar=True)

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])
        t_analytical = data[rand_case]['t_analytical']
        y_analytical = data[rand_case]['y_analytical']
        t_ng_2020 = data[rand_case]['t_ng_2020']
        y_ng_2020 = data[rand_case]['y_ng_2020']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_analytical, y_analytical, "-", label='Analytical')
        plt.plot(t_ng_2020, y_ng_2020, "-", label='SPH-VCPM (Ng et al. 2020)')
        for name in self.case_info:
            t_ctvf = data[name]['t_ctvf']
            y_ctvf = data[name]['y_ctvf']

            plt.plot(t_ctvf, y_ctvf, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('y - amplitude')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_amplitude.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

    def plot_prop(self, conditions=None,
                  fcmap='rainbow', scmap='rainbow', figsize=(10, 4), size=20,
                  dpi=300, show_fluid=True, fmin=-6, fmax=6,
                  show_structure=True, smin=-6, smax=6, show_fcmap=True,
                  show_fsmap=True, times=None, only_colorbar=False,
                  fname_prefix=''):
        if conditions is None:
            conditions = {}
        if times is None:
            times = [1, 2, 3]
        aspect = 70
        pad = 0.
        size_boundary = 0.1

        for case in filter_cases(self.cases, **conditions):
            filename_w_ext = os.path.basename(case.base_command.split(' ')[1])
            filename = os.path.splitext(filename_w_ext)[0]
            files = get_files(case.input_path(), filename)
            logfile = case.input_path(f'{filename}.log')
            files = get_files_at_given_times_from_log(files, times, logfile)
            for file, t in zip(files, times):
                fig, ax = plt.subplots(1, 1, figsize=figsize)
                data = load(file)
                t = data['solver_data']['t']
                f = data['arrays']['fluid']
                s1 = data['arrays']['tank']
                s = data['arrays']['gate']
                s2 = data['arrays']['gate_support']
                label = rf"p"
                val = f.get('p')
                if show_fluid == True:
                    tmp = ax.scatter(
                        f.x, f.y, c=val, s=size, rasterized=True, cmap=fcmap,
                        edgecolor='none', vmin=fmin, vmax=fmax
                    )

                    if show_fcmap == True:
                        # cbarf = fig.colorbar(tmp, ax=ax, shrink=0.8, label=f'{label}',
                        #                      pad=0.01, aspect=20)
                        cbarf = fig.colorbar(tmp, ax=ax, label=f'{label}',
                                             pad=pad, aspect=aspect,
                                             format='%.0e')
                        cbarf.ax.tick_params(labelsize='xx-small')

                if show_structure == True:
                    tmp2 = ax.scatter(s.x, s.y, c=s.sigma00, s=size, rasterized=True,
                                      cmap=scmap,
                                      edgecolor='none',
                                      vmin=smin, vmax=smax)

                    if show_fsmap == True:
                        # cbars = fig.colorbar(tmp2, ax=ax, shrink=0.8, label=r'$\sigma_{00}$',
                        #                      pad=0.01, aspect=20)
                        cbars = fig.colorbar(tmp2, ax=ax, label=r'$\sigma_{00}$',
                                             pad=pad, aspect=aspect,
                                             format='%.0e')
                        cbars.ax.tick_params(labelsize='xx-small')

                msg = r"$t = $" + f'{t:.1f}'
                # xmax = f.x.max()
                # ymax = f.y.max()
                # ax.annotate(
                #     msg, (xmax*1.2, ymax*1.0), fontsize='small',
                #     bbox=dict(boxstyle="square,pad=0.3", fc='white')
                # )

                if show_fluid == True and show_structure == True:
                    ax.scatter(s1.x, s1.y, c=s1.m, cmap='viridis', s=size_boundary,
                               rasterized=True)
                    ax.scatter(s2.x, s2.y, c=s2.m, cmap='viridis', s=size_boundary,
                               rasterized=True)
                # ax.title()
                ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid()
                ax.set_aspect('equal')

                if only_colorbar is True:
                    ax.remove()

                fig.savefig(self.output_path(f'{fname_prefix}_t_{t:.1f}.png'),
                            dpi=dpi)

                plt.clf()
                plt.close()

    def get_colorbar_limits(self, conditions=None, fval='p', sval='sigma00',
                            times=[0]):
        if conditions is None:
            conditions = {}
        if times is None:
            times = [1, 2, 3]

        fmin = 0.
        fmax = 0.
        smin = 0.
        smax = 0.

        for case in filter_cases(self.cases, **conditions):
            filename_w_ext = os.path.basename(case.base_command.split(' ')[1])
            filename = os.path.splitext(filename_w_ext)[0]
            files = get_files(case.input_path(), filename)
            logfile = case.input_path(f'{filename}.log')
            files = get_files_at_given_times_from_log(files, times, logfile)
            for file, t in zip(files, times):
                data = load(file)
                f = data['arrays']['fluid']
                s = data['arrays']['gate']

                if fmin > min(f.get(fval)):
                    fmin = min(f.get(fval))

                if fmax < max(f.get(fval)):
                    fmax = max(f.get(fval))

                if smin > min(s.get(sval)):
                    smin = min(s.get(sval))

                if smax < max(s.get(sval)):
                    smax = max(s.get(sval))

        return fmin, fmax, smin, smax


if __name__ == '__main__':
    PROBLEMS = [
        Skillen2013WaterEntryHalfBuoyant
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
