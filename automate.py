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
# n_core = 6
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


class PlanePoiseuilleFlow2D(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'plane_poiseuille_flow_2D'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/plane_poiseuille_flow_2D.py' + backend

        # Base case info
        self.case_info = {
            'case_1': (dict(
                tf=50,
                ), 'Case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


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


class Sun20152DWaterWedgeEntry(Problem):
    """Paper name: Numerical simulation of interactions between free surface and
    rigid body using a robust SPH method

    Pertains to figure 16 and figure 17

    """
    def get_name(self):
        return 'sun_2015_2d_water_wedge_entry'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/sun_2015_2d_water_wedge_entry.py' + backend

        # Base case info
        self.case_info = {
            'dx_10': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=100,
                spacing=10.*1e-3,
                ), 'dx = 10 mm'),

            'dx_5': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=100,
                spacing=5*1e-3,
                ), 'dx = 5 mm'),

            'dx_025': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=200,
                spacing=2.5*1e-3,
                # Set specific dt for this case for stable simulation
                timestep=5e-6
                ), 'dx = 2.5 mm')
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       tf=40*1e-3,
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
        t_exp_v_vel = data[rand_case]['t_exp_v_vel']
        t_BEM_v_vel = data[rand_case]['t_BEM_v_vel']
        t_SPH_v_vel = data[rand_case]['t_SPH_v_vel']
        v_vel_exp = data[rand_case]['v_vel_exp']
        v_vel_BEM = data[rand_case]['v_vel_BEM']
        v_vel_SPH = data[rand_case]['v_vel_SPH']
        fcm_y_SPH = data[rand_case]['fcm_y_SPH']
        t_SPH_fcm_y = data[rand_case]['t_SPH_fcm_y']

        # ========================
        # Variation of y velocity
        # ========================
        plt.clf()
        for name in self.case_info:
            t_current = data[name]['t_current']
            v_vel_current = data[name]['v_vel_current']

            plt.plot(t_current, -v_vel_current, label=self.case_info[name][1])
        plt.plot(t_exp_v_vel, v_vel_exp, "^", label='EXP. (Yettou et al., 2006)')
        plt.plot(t_SPH_v_vel, v_vel_SPH, "-+", label='dx = 2.5 mm, SPH (Sun et al., 2015)')
        plt.plot(t_BEM_v_vel, v_vel_BEM, "--", label='BEM (Zhao and Faltinsen, 1993)')

        plt.title('Variation in velocity')
        plt.xlabel('Time (ms)')
        plt.ylabel('Wedge velocity (m/s)')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ============================
        # Variation of y velocity ends
        # ============================
        # ============================
        # Variation of force velocity
        # ============================
        plt.clf()
        for name in self.case_info:
            t_current = data[name]['t_current']
            fcm_y_current = data[name]['fcm_y_current']

            plt.plot(t_current, fcm_y_current, label=self.case_info[name][1])
        plt.plot(t_SPH_fcm_y, fcm_y_SPH, "-+", label='dx = 2.5 mm, SPH (Sun et al., 2015)')

        plt.title('Variation in force')
        plt.xlabel('Time (ms)')
        plt.ylabel('Force (N)')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('force_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ============================
        # Variation of force
        # ============================


class Hashemi2012NeutrallyBuoyantCircularCylinderInShearFlow(Problem):
    """
    Pertains to Figure 4 of Ng 2021 Numerical computation of fluid solid mixture
    flow using SPH-VCPM-DEM method
    """
    def get_name(self):
        return 'ng_2021_two_cylinders_in_shear_flow'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/ng_2021_3_1_1_two_cylinders_in_shear_flow.py' + backend

        # Base case info
        self.case_info = {
            'N_10': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=10,
                ), 'N=10'),

            'N_20': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=20,
                ), 'N=20'),

            'N_30': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=30,
                ), 'N=30'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       tf=60,
                       no_use_edac=None, re=0.75,
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
        t_SPH_Hashemi = data[rand_case]['t_SPH_Hashemi']
        u_cm_SPH_Hashemi = data[rand_case]['u_cm_SPH_Hashemi']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.clf()
        for name in self.case_info:
            t_current = data[name]['t_current']
            u_cm_current = data[name]['u_current']

            plt.plot(t_current, u_cm_current, label=self.case_info[name][1])
        plt.plot(t_SPH_Hashemi, u_cm_SPH_Hashemi, "-+", label='Hashemi et al. (2012)')

        plt.xlabel('t')
        plt.ylabel('U velocity')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('u_cm_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================


class Ng2021TwoCylindersInShearFlow(Problem):
    """
    Pertains to Figure 4 of Ng 2021 Numerical computation of fluid solid mixture
    flow usingn SPH-VCPM-DEM method
    """
    def get_name(self):
        return 'ng_2021_two_cylinders_in_shear_flow'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/ng_2021_3_1_1_two_cylinders_in_shear_flow.py' + backend

        # Base case info
        self.case_info = {
            'N_10': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=10,
                ), 'N=10'),

            'N_20': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=20,
                ), 'N=20'),

            'N_30': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=500,
                N=30,
                ), 'N=30'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       tf=60,
                       no_use_edac=None, re=0.75,
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
        t_SPH_Hashemi = data[rand_case]['t_SPH_Hashemi']
        u_cm_SPH_Hashemi = data[rand_case]['u_cm_SPH_Hashemi']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.clf()
        for name in self.case_info:
            t_current = data[name]['t_current']
            u_cm_current = data[name]['u_current']

            plt.plot(t_current, u_cm_current, label=self.case_info[name][1])
        plt.plot(t_SPH_Hashemi, u_cm_SPH_Hashemi, "-+", label='Hashemi et al. (2012)')

        plt.xlabel('t')
        plt.ylabel('U velocity')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('u_cm_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================


class Hashemi2012FallingCircularCylinderInClosedChannel(Problem):
    """Paper name: A modiﬁed SPH method for simulating motion of rigid bodies in
    Newtonian ﬂuid ﬂows

    Pertains to figure 8

    """
    def get_name(self):
        return 'hashemi_2012_falling_circular_cylinder_in_closed_channel'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/hashemi_2012_falling_circular_cylinder_in_closed_channel.py' + backend

        # Base case info
        self.case_info = {
            'N_10': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=100,
                N=10,
                ), 'N = 10'),

            'N_15': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=200,
                N=15,
                # Set specific dt for this case for stable simulation
                timestep=3e-6
                ), 'N = 15'),

            'N_20': (dict(
                scheme='wcsph',
                alpha=0.0,
                pfreq=1000,
                N=20,
                # Set specific dt for this case for stable simulation
                timestep=2e-6
                ), 'N = 20'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       tf=0.9,
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
        t_v_vel_FEM= data[rand_case]['t_v_vel_FEM']
        v_vel_FEM= data[rand_case]['v_vel_FEM']
        t_y_pos_FPM_PST= data[rand_case]['t_y_pos_FPM_PST']
        y_pos_FPM_PST= data[rand_case]['y_pos_FPM_PST']

        # ========================
        # Variation of y velocity
        # ========================
        plt.clf()
        for name in self.case_info:
            t_current = data[name]['t_current']
            v_current = data[name]['v_current']

            plt.plot(t_current, v_current, label=self.case_info[name][1])
        plt.plot(t_v_vel_FEM, v_vel_FEM, "^", label='Glowinski 2001 (FEM)')

        plt.title('Variation in velocity')
        plt.xlabel('Time')
        plt.ylabel('Velocity (m/s)')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('velocity_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ============================
        # Variation of y velocity ends
        # ============================

        # ========================
        # Variation of y position
        # ========================
        plt.clf()
        for name in self.case_info:
            t_current = data[name]['t_current']
            y_current = data[name]['y_current']

            plt.plot(t_current, y_current, label=self.case_info[name][1])
        plt.plot(t_y_pos_FPM_PST, y_pos_FPM_PST, "o", label='Zhang 2019 FPM PST')

        plt.title('Variation in position')
        plt.xlabel('Time')
        plt.ylabel('Position m')
        plt.legend(prop={'size': 12})
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('position_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ============================
        # Variation of y velocity ends
        # ============================


if __name__ == '__main__':
    PROBLEMS = [
        # Problem  no 0 (for fluid validation)
        PlanePoiseuilleFlow2D,
        # Problem  no 1 (500 density)
        Skillen2013WaterEntryHalfBuoyant,
        # Problem  no 1 (1000 density)
        Skillen2013WaterEntryNeutrallyBuoyant,
        # Problem  no 2
        Sun20152DWaterWedgeEntry,
        # Problem  no 3
        Hashemi2012FallingCircularCylinderInClosedChannel,
        # Problem  no 4
        Hashemi2012NeutrallyBuoyantCircularCylinderInShearFlow,
        # Problem  no 5
        Ng2021TwoCylindersInShearFlow,
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
