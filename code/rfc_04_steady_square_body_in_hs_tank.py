"""Hydrostatic tank

"""
import numpy as np
import os

from pysph.solver.utils import load, get_files

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
import pysph.tools.geometry as G
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser

from boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from geometry import hydrostatic_tank_2d

from fluids import (get_particle_array_fluid,
                    get_particle_array_boundary)
from rigid_body import (get_particle_array_rigid_body,
                        set_linear_velocity_of_rigid_body,
                        set_angular_velocity)
from rigid_fluid_coupling import (ParticlesFluidScheme,
                                  add_rigid_fluid_properties_to_rigid_body)


class SquareBodyInHSTank(Application):
    def initialize(self):
        self.fluid_length = 1.4
        self.fluid_height = 1.8
        self.container_height = 2.
        self.container_width = 4
        self.nboundary_layers = 4

        self.hdx = 1.
        self.dx = 0.035
        self.dim = 2

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.ro = 1000.0
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10.0 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 2.0
        self.p0 = self.ro*self.c0**2
        # print(self.boundary_equations)

        # rigid body properties
        self.rigid_body_length = 0.2
        self.rigid_body_height = 0.2
        self.rigid_body_rho = 500.

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

        # create rigid body
        x, y = get_2d_block(dx=self.dx,
                            length=self.rigid_body_length,
                            height=self.rigid_body_height)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        dem_id = body_id
        m = self.rigid_body_rho * self.dx**2
        h = self.h
        rad_s = self.dx / 2.
        y[:] += max(fluid.y) - min(y)
        y[:] += 2. * self.dx
        y[:] -= 2. * self.rigid_body_length
        rigid_body = get_particle_array_rigid_body(name='rigid_body',
                                                   x=x,
                                                   y=y,
                                                   z=0,
                                                   h=h,
                                                   m_rb=m,
                                                   dem_id=dem_id,
                                                   body_id=body_id)

        print("rigid body total mass: ", rigid_body.total_mass)
        rigid_body.rho[:] = self.ro
        G.remove_overlap_particles(
            fluid, rigid_body, self.dx, dim=self.dim
        )
        # set_linear_velocity_of_rigid_body(rigid_body, [1., 1., 0.])
        # set_angular_velocity(rigid_body, [0., 0., 2 * np.pi])

        add_rigid_fluid_properties_to_rigid_body(rigid_body)
        rigid_body.m[:] = self.ro * self.dx**2.

        return [fluid, tank, rigid_body]

    def create_scheme(self):
        wcsph = ParticlesFluidScheme(
            fluids=['fluid'],
            boundaries=['tank'],
            # boundaries=['tank', 'rigid_body'],
            # rigid_bodies=[],
            rigid_bodies=["rigid_body"],
            dim=2, rho0=self.ro, c0=self.c0, pb=self.p0, nu=None,
            gy=self.gy, alpha=0.05)

        s = SchemeChooser(default='wcsph', wcsph=wcsph)
        return s

    def configure_scheme(self):
        # dt = 0.125*self.h/(co + vref)
        dt = 5e-5
        h0 = self.hdx * self.dx
        scheme = self.scheme
        self.scheme.configure_solver(dt=dt, tf=self.tf)

    # def create_equations(self):
    #     eqns = self.scheme.get_equations()

    #     # Adami boundary conditions
    #     adamibc_eq = []
    #     adamibc_eq.append(
    #         AdamiBoundaryConditionExtrapolateNoSlipFixedParticles(
    #             dest="structure", sources=["structure"]))

    #     eqns.groups[0].append(Group(adamibc_eq))

    #     # make accelerations zero on specific indices
    #     makeaccelzero_eq = []
    #     makeaccelzero_eq.append(MakeAccelerationZeroOnSelectedIndices(
    #         dest="structure", sources=None))

    #     eqns.groups[1].append(Group(makeaccelzero_eq))

    #     # print(eqns)
    #     return eqns

    def customize_output(self):
        self._mayavi_config('''
        # b = particle_arrays['fluid_1']
        # b.scalar = 'vmag'
        # b = particle_arrays['fluid_2']
        # b.scalar = 'vmag'
        ''')

    def post_process(self, fname):
        import matplotlib.pyplot as plt

        output_files = get_files(os.path.dirname(fname))

        from pysph.solver.utils import iter_output

        files = output_files

        force_x = []
        force_y = []
        force_z = []
        t = []

        for sd, rigid_body in iter_output(files, 'rigid_body'):
            _t = sd['t']
            force_x.append(rigid_body.force[0])
            force_y.append(rigid_body.force[1])
            force_z.append(rigid_body.force[2])
            t.append(_t)
        plt.plot(t, force_x, label="force x")
        plt.plot(t, force_y, label="force y")
        plt.plot(t, force_z, label="force z")
        plt.legend()
        fig = os.path.join(self.output_dir, "t_vs_force.png")
        plt.savefig(fig, dpi=300)
        plt.show()


if __name__ == '__main__':
    app = SquareBodyInHSTank()
    app.run()
    app.post_process(app.info_filename)
