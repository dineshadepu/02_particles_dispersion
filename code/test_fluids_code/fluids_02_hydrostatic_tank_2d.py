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
from geometry import hydrostatic_tank_2d


class TwoFluidBlocksColliding(Application):
    def initialize(self):
        self.fluid_length = 1.4
        self.fluid_height = 1.8
        self.container_height = 2.
        self.container_width = 4
        self.nboundary_layers = 4

        self.hdx = 1.
        self.dx = 0.05
        self.dim = 2

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.ro = 1000.0
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 1.0
        self.p0 = self.ro*self.c0**2
        # print(self.boundary_equations)

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length, self.fluid_height,
                                      self.container_height, self.nboundary_layers - 1,
                                      self.dx, self.dx)
        h = self.hdx * self.dx
        self.h = h
        m = self.dx**2 * self.ro
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=0., h=h, m=m, rho=self.ro)
        tank = get_particle_array_boundary(name='tank', x=xt, y=yt, z=0., h=h, m=m, rho=self.ro)
        G.remove_overlap_particles(
            fluid, tank, self.dx, dim=self.dim
        )

        return [fluid, tank]

    def create_scheme(self):
        wcsph = FluidsScheme(
            fluids=['fluid'], boundaries=['tank'],
            dim=2, rho0=self.ro, c0=self.c0, pb=self.p0, nu=self.nu,
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
    app = TwoFluidBlocksColliding()
    app.run()
    app.post_process(app.info_filename)
