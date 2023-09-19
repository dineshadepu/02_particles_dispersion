"""Hydrostatic tank

"""
import numpy as np

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
import pysph.tools.geometry as G
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
from fluids import (get_particle_array_fluid,
                    get_particle_array_boundary,
                    FluidsScheme)

from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from geometry import get_fluid_tank_3d


class HydrostaticTank(Application):
    def initialize(self):
        # x - axis
        self.fluid_length = 0.2
        # y - axis
        self.fluid_height = 0.11
        # z - axis
        self.fluid_depth = 0.2

        # x - axis
        self.tank_length = 0.2
        # y - axis
        self.tank_height = 0.14
        # z - axis
        self.tank_depth = 0.2
        self.tank_layers = 3

        self.hdx = 1.
        self.dx = 0.05
        self.dim = 3

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.rho0 = 1000.0
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 1.0
        self.p0 = self.rho0*self.c0**2
        # print(self.boundary_equations)

    def create_particles(self):
        xf, yf, zf, xt, yt, zt = get_fluid_tank_3d(self.fluid_length, self.fluid_height,
                                                   self.fluid_height,
                                                   self.tank_length,
                                                   self.tank_height,
                                                   self.tank_layers - 1,
                                                   self.dx, self.dx,
                                                   hydrostatic=True)
        h = self.hdx * self.dx
        self.h = h
        m = self.dx**self.dim * self.rho0
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=zf, h=h, m=m, rho=self.rho0)
        tank = get_particle_array_boundary(name='tank', x=xt, y=yt, z=zt, h=h, m=m, rho=self.rho0)
        # G.remove_overlap_particles(
        #     fluid, tank, self.dx, dim=self.dim
        # )

        return [fluid, tank]

    def create_scheme(self):
        wcsph = FluidsScheme(
            fluids=['fluid'], boundaries=['tank'],
            dim=self.dim, rho0=self.rho0, c0=self.c0, pb=self.p0, nu=self.nu,
            gy=self.gy, alpha=0.05)

        s = SchemeChooser(default='wcsph', wcsph=wcsph)
        return s

    def configure_scheme(self):
        # dt = 0.125*self.h/(co + vref)
        dt = 5e-5
        h0 = self.hdx * self.dx
        scheme = self.scheme
        self.scheme.configure_solver(dt=dt, tf=self.tf)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid_1']
        b.scalar = 'vmag'
        b = particle_arrays['fluid_2']
        b.scalar = 'vmag'
        ''')


if __name__ == '__main__':
    app = HydrostaticTank()
    app.run()
    app.post_process(app.info_filename)
