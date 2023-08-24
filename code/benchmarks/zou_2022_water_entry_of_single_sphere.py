"""Hydrostatic tank

"""
import numpy as np

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.tools.geometry import get_3d_sphere
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

from rigid_body import (get_particle_array_rigid_body,
                        set_linear_velocity_of_rigid_body,
                        set_angular_velocity)
from rigid_fluid_coupling import (ParticlesFluidScheme,
                                  add_rigid_fluid_properties_to_rigid_body)


class HydrostaticTank(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=0.0125,
            help="Reynolds number of flow."
        )
        group.add_argument(
            "--remesh", action="store", type=float, dest="remesh", default=0,
            help="Remeshing frequency (setting it to zero disables it)."
        )

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        self.re = self.options.re
        # ======================
        # ======================

        # ======================
        # Dimensions
        # ======================
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

        self.rigid_body_length = 0.2
        self.rigid_body_height = 0.2
        self.rigid_body_diameter = 25.4 * 1e-3
        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.rigid_body_rho = 2000.

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 3
        # ======================
        # Physical properties and consants ends
        # ======================

        # ======================
        # Numerical properties
        # ======================
        self.hdx = 1.
        self.dx = self.rigid_body_diameter / 10
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 0.09
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.05

        # Setup default parameters.
        dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        # dt_viscous = 0.125 * self.h**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h/(self.gy))

        self.dt = min(dt_cfl, dt_force)
        # ==========================
        # Numerical properties ends
        # ==========================

    def create_particles(self):
        xf, yf, zf, xt, yt, zt = get_fluid_tank_3d(self.fluid_length,
                                                   self.fluid_height,
                                                   self.fluid_depth,
                                                   self.tank_length,
                                                   self.tank_height,
                                                   self.tank_layers - 1,
                                                   self.dx, self.dx,
                                                   hydrostatic=True)
        h = self.hdx * self.dx
        self.h = h
        m = self.dx**self.dim * self.fluid_rho
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=zf, h=h, m=m, rho=self.fluid_rho)
        tank = get_particle_array_boundary(name='tank', x=xt, y=yt, z=zt, h=h, m=m, rho=self.fluid_rho)

        # =========================
        # create rigid body
        # =========================
        x, y, z = get_3d_sphere(dx=self.dx,
                                r=self.rigid_body_diameter/2.)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        dem_id = body_id
        m = self.rigid_body_rho * self.dx**2
        h = self.h
        rad_s = self.dx / 2.
        # y[:] += 2. * dx
        # y[:] -= 2. * self.rigid_body_length
        # x[:] = min(fluid.x) - min(x) + self.rigid_body_length
        y[:] += 0.3
        x[:] += 0.3
        rigid_body = get_particle_array_rigid_body(name='rigid_body',
                                                   x=x,
                                                   y=y,
                                                   z=z,
                                                   h=h,
                                                   m_rb=m,
                                                   dem_id=dem_id,
                                                   body_id=body_id)

        # print("rigid body total mass: ", rigid_body.total_mass)
        rigid_body.rho[:] = self.fluid_rho
        G.remove_overlap_particles(
            fluid, rigid_body, self.dx, dim=self.dim
        )
        add_rigid_fluid_properties_to_rigid_body(rigid_body)
        # set_linear_velocity_of_rigid_body(rigid_body, [1., 1., 0.])
        rigid_body.m[:] = self.fluid_rho * self.dx**self.dim
        # =========================
        # create rigid body ends
        # =========================

        return [fluid, tank, rigid_body]

    def create_scheme(self):
        wcsph = ParticlesFluidScheme(
            fluids=['fluid'],
            boundaries=['tank'],
            rigid_bodies=["rigid_body"],
            dim=0.,
            rho0=0.,
            c0=0.,
            pb=0.,
            nu=0.,
            gy=0.,
            alpha=0.)
        s = SchemeChooser(default='wcsph', wcsph=wcsph)
        return s

    def configure_scheme(self):
        tf = 100.0
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            rho0=self.fluid_rho,
            c0=self.c0,
            pb=self.p0,
            nu=self.nu,
            gy=self.gy,
            alpha=self.alpha)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=1000)
        print("dt = %g"%self.dt)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['rigid_body']
        b.scalar = 'm'
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')


if __name__ == '__main__':
    app = HydrostaticTank()
    app.run()
    app.post_process(app.info_filename)
