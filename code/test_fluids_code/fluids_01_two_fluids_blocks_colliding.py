"""Hydrostatic tank

python dam_break_2d.py --openmp --integrator gtvf --no-internal-flow --pst sun2019 --no-set-solid-vel-project-uhat --no-set-uhat-solid-vel-to-u --no-vol-evol-solid --no-edac-solid --surf-p-zero -d dam_break_2d_etvf_integrator_gtvf_pst_sun2019_output --pfreq 1 --detailed-output


"""
import numpy as np

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


fluid_column_height = 2.0
fluid_column_width = 1.0
container_height = 4.0
container_width = 4
nboundary_layers = 4


class TwoFluidBlocksColliding(Application):
    def initialize(self):
        super(TwoFluidBlocksColliding, self).initialize()

        self.hdx = 1.
        self.dx = 0.1
        self.dim = 2

        self.gx = 0.
        self.gy = 0.
        self.gz = 0.
        self.ro = 1000.0
        self.vref = 1.
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 1.0
        self.p0 = self.ro*self.c0**2
        # print(self.boundary_equations)

    def create_particles(self):
        xf1, yf1 = get_2d_block(dx=self.dx, length=fluid_column_height, height=fluid_column_height)
        h = self.hdx * self.dx
        self.h = h
        m = self.dx**2 * self.ro
        m_tank = (self.dx)**2 * self.ro
        h_tank = self.hdx * self.dx
        fluid_1 = get_particle_array_fluid(name='fluid_1', x=xf1, y=yf1, z=0., h=h, m=m, rho=self.ro)
        fluid_1.u[:] = 1.

        fluid_2 = get_particle_array_fluid(name='fluid_2', x=xf1, y=yf1, z=0., h=h, m=m, rho=self.ro)
        fluid_2.u[:] = -1.
        fluid_2.x[:] += fluid_column_height * 1.2

        return [fluid_1, fluid_2]

    def create_scheme(self):
        wcsph = FluidsScheme(
            fluids=['fluid_1', 'fluid_2'], boundaries=[],
            dim=2, rho0=self.ro, c0=self.c0, pb=self.p0, nu=None,
            gy=self.gy, alpha=0.05)

        s = SchemeChooser(default='wcsph', wcsph=wcsph)
        return s

    def configure_scheme(self):
        # dt = 0.125*self.h/(co + vref)
        dt = 1e-4
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
