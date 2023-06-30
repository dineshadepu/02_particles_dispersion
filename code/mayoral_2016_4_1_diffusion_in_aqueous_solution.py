"""Diffusion in aqueous solution

python dam_break_2d.py --openmp --integrator gtvf --no-internal-flow --pst sun2019 --no-set-solid-vel-project-uhat --no-set-uhat-solid-vel-to-u --no-vol-evol-solid --no-edac-solid --surf-p-zero -d dam_break_2d_etvf_integrator_gtvf_pst_sun2019_output --pfreq 1 --detailed-output


"""
import numpy as np
import math

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
from fluids import (get_particle_array_fluid, FluidsScheme)

from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from geometry import get_files_at_given_times_from_log


def set_the_concentration(x, y, concentration):
    pass

class DiffusionInAqueousSolution(Application):
    def initialize(self):
        super(DiffusionInAqueousSolution, self).initialize()

        self.hdx = 1.
        self.dx = 0.01
        self.dim = 2

        self.gx = 0.
        self.gy = 0.
        self.gz = 0.
        self.ro = 1000.0
        self.vref = 1.
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 2.0
        self.p0 = self.ro*self.c0**2

        # fluid dimensions
        self.fluid_length = 1.
        self.fluid_height = 1.
        self.diff_coeff = 1e-4
        self.concentration = 1e-6
        # print(self.boundary_equations)

    def create_particles(self):
        xf, yf = get_2d_block(dx=self.dx, length=self.fluid_length,
                              height=self.fluid_height)
        xf += -min(xf)
        h = self.hdx * self.dx
        self.h = h
        m = self.dx**2 * self.ro
        m_tank = (self.dx)**2 * self.ro
        h_tank = self.hdx * self.dx
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=0.,
                                         h=h, m=m, rho=self.ro)
        fluid.is_static[:] = 1
        # set the
        indices = []
        for i in range(len(fluid.x)):
            if (fluid.y[i] + self.dx / 10 < self.dx) and (fluid.y[i] - self.dx / 10 > -self.dx):
                indices.append(i)
        print(indices)

        indices_limit = []
        for i in range(len(indices)):
            if fluid.x[indices[i]] >= 0.45 and fluid.x[indices[i]] <= 0.55:
                indices_limit.append(indices[i])
        fluid.add_property('concentration_indices_plot')
        fluid.concentration_indices_plot[indices] = 1
        fluid.concentration[indices_limit] = self.concentration
        fluid.diff_coeff[:] = self.diff_coeff

        return [fluid]

    def create_scheme(self):
        wcsph = FluidsScheme(
            fluids=['fluid'], boundaries=[],
            dim=2, rho0=self.ro, c0=self.c0, pb=self.p0, nu=None,
            gy=self.gy, alpha=0.05)

        s = SchemeChooser(default='wcsph', wcsph=wcsph)
        return s

    def configure_scheme(self):
        # dt = 0.125*self.h/(co + vref)
        dt = 1e-4
        h0 = self.hdx * self.dx
        scheme = self.scheme
        self.output_at_times = np.array([1.0, 2.0])
        self.scheme.configure_solver(dt=dt, tf=self.tf, output_at_times=self.output_at_times)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'concentration'
        ''')

    def post_process(self, fname):
        import os
        from pysph.solver.utils import load, get_files
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from pysph.solver.utils import iter_output

        info = self.read_info(fname)
        output_files = self.output_files

        # print(self.output_files)
        output_times = np.array([1.0, 2.0])
        logfile = os.path.join(
            os.path.dirname(fname),
            'mayoral_2016_4_1_diffusion_in_aqueous_solution.log')
        to_plot = get_files_at_given_times_from_log(output_files, output_times,
                                                    logfile)

        for sd, fluid in iter_output(to_plot, 'fluid'):
            _t = sd['t']
            # fluid.z = 1.

        # ====================================
        # Analytical solution
        # ====================================
        z_analytical = np.linspace(0.35, 0.65, 1000)
        conc_analytical = np.zeros(len(z_analytical))
        z_1 = 0.45
        z_0 = 0.5
        z_2 = 0.55
        conc_0 = 1e-6
        diff_coeff = 1e-4
        time = [1., 2.]
        for t in time:
            for i in range(len(z_analytical)):
                tmp = conc_0 / 2.
                if z_analytical[i] <= z_0:
                    conc_analytical[i] = tmp * math.erfc((z_1 - z_analytical[i]) / (2. * np.sqrt(diff_coeff * t)))
                if z_analytical[i] > z_0:
                    conc_analytical[i] = tmp * math.erfc((z_analytical[i] - z_2) / (2. * np.sqrt(diff_coeff * t)))
        # ====================================
        # Analytical solution
        # ====================================

        # Save the solution in a file to reuse while automation
        if 'info' in fname:
            res = os.path.join(os.path.dirname(fname), "results.npz")
        else:
            res = os.path.join(fname, "results.npz")

        np.savez(res,
                 z_sph=z_sph,
                 conc_sph=conc_sph,
                 z_analytical=z_analytical,
                 conc_analytical=conc_analytical)

        plt.clf()
        plt.plot(z_sph, conc_sph, "-", label='SPH')
        plt.plot(z_analytical, conc_analytical, "-", label='Analytical')

        plt.title('Diffusion in aqueous solution')
        plt.xlabel('z')
        plt.ylabel('Concentration')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "z_vs_concentration.pdf")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = DiffusionInAqueousSolution()
    app.run()
    app.post_process(app.info_filename)
