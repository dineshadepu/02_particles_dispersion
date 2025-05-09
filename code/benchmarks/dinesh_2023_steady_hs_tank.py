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
from geometry import hydrostatic_tank_2d, create_circle_1, translate_system_with_left_corner_as_origin
# from geometry import hydrostatic_tank_2d, create_circle_1

from rigid_body import (get_particle_array_rigid_body,
                        set_linear_velocity_of_rigid_body,
                        set_angular_velocity,
                        move_body_to_new_center,
                        get_center_of_mass)
from rigid_fluid_coupling import (ParticlesFluidScheme,
                                  add_rigid_fluid_properties_to_rigid_body)

def check_time_make_zero(t, dt):
    if t < 0.1:
        return True
    else:
        return False


class MakeForcesZeroOnRigidBody(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.

    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')

        frc = dst.force
        trq = dst.torque

        frc[:] = 0
        trq[:] = 0


class Problem(Application):
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
        self.fluid_length = 0.02
        # y - axis
        self.fluid_height = 0.06
        # z - axis
        self.fluid_depth = 0.2

        # x - axis
        self.tank_length = 0.2
        # y - axis
        self.tank_height = 0.08
        # z - axis
        self.tank_depth = 0.2
        self.tank_layers = 3

        self.rigid_body_length = 0.2
        self.rigid_body_height = 0.2
        self.rigid_body_diameter = 2. * 0.00125
        self.rigid_body_center = [0.01, 0.04]
        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.rigid_body_rho = 1250.

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2
        # ======================
        # Physical properties and consants ends
        # ======================

        # ======================
        # Numerical properties
        # ======================
        self.hdx = 1.
        self.dx = self.rigid_body_diameter / 8
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.00
        self.tf = 0.9
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.05

        # Setup default parameters.
        dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        dt_viscous = 1e5
        if self.nu > 1e-12:
            dt_viscous = 0.125 * self.h**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h/(np.abs(self.gy)))
        print("dt_cfl", dt_cfl, "dt_viscous", dt_viscous, "dt_force", dt_force)

        self.dt = min(dt_cfl, dt_force)
        self.dt = 5e-6
        # ==========================
        # Numerical properties ends
        # ==========================

    def create_particles(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length, self.fluid_height,
                                             self.tank_height, self.tank_layers,
                                             self.dx, self.dx)
        zt = np.zeros_like(xt)
        zf = np.zeros_like(xf)

        # move fluid such that the left corner is at the origin of the
        # co-ordinate system
        translation = translate_system_with_left_corner_as_origin(xf, yf, zf)
        xt[:] = xt[:] - translation[0]
        yt[:] = yt[:] - translation[1]
        zt[:] = zt[:] - translation[2]

        m = self.dx**self.dim * self.fluid_rho
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=zf, h=self.h, m=m, rho=self.fluid_rho)
        tank = get_particle_array_boundary(name='tank', x=xt, y=yt, z=zt, h=self.h, m=m, rho=self.fluid_rho)

        # =========================
        # create rigid body
        # =========================
        x, y = create_circle_1(self.rigid_body_diameter,
                               self.dx)
        center = [0.01, 0.04, 0.]
        z = np.zeros_like(x)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        dem_id = body_id
        m = self.rigid_body_rho * self.dx**self.dim * np.ones_like(x)
        h = self.h
        rad_s = self.dx / 2.
        # get center of mass of the body
        xcm = get_center_of_mass(x, y, z, m)
        move_body_to_new_center(xcm, x, y, z, center)
        # move_body_to_new_center(xcm, x, y, z, [0., 0.8, 0.])

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
            dim=2,
            rho0=0.,
            h=0.,
            c0=0.,
            pb=0.,
            nu=0.,
            gy=0.,
            alpha=0.)
        s = SchemeChooser(default='wcsph', wcsph=wcsph)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            rho0=self.fluid_rho,
            h=self.h,
            c0=self.c0,
            pb=self.p0,
            nu=self.nu,
            gy=self.gy,
            alpha=self.alpha)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=1000)
        print("dt = %g"%self.dt)

    def create_equations(self):
        # print("inside equations")
        eqns = self.scheme.get_equations()

        # Apply external force
        zero_frc = []
        zero_frc.append(
            MakeForcesZeroOnRigidBody("rigid_body", sources=None))

        # print(eqns.groups)
        eqns.groups[-1].append(Group(equations=zero_frc,
                                         condition=check_time_make_zero))

        return eqns

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['rigid_body']
        b.scalar = 'm'
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load, get_files
        from pysph.solver.utils import iter_output
        import os

        info = self.read_info(fname)
        files = self.output_files
        # y = []
        # v = []
        # u = []
        t = []

        fluid_max_y = []
        fluid_max_p = []

        for sd, rigid_body, fluid in iter_output(files, 'rigid_body', 'fluid'):
            _t = sd['t']
            # y.append(rigid_body.xcm[1])
            # u.append(rigid_body.vcm[0])
            # v.append(rigid_body.vcm[1])
            t.append(_t)

            fluid_max_y.append(max(fluid.y))
            fluid_max_p.append(max(fluid.p))

        # Data from literature
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        data_y_vel_hashemi_2012_fem = np.loadtxt(os.path.join(
            directory, 'hashemi_2012_falling_cylinder_fem_data.csv'), delimiter=',')

        t_has, vel_has = data_y_vel_hashemi_2012_fem[:, 0], data_y_vel_hashemi_2012_fem[:, 1]

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res, t_has=t_has, vel_has=vel_has)

        # # ========================
        # # Variation of y velocity
        # # ========================
        # plt.clf()
        # plt.plot(t_has, vel_has, "-", label='FEM')
        # plt.plot(t, u, "o", label='Current')

        # plt.title('Variation in y-velocity')
        # plt.xlabel('t')
        # plt.ylabel('y-velocity')
        # plt.legend()
        # fig = os.path.join(os.path.dirname(fname), "y_velocity_with_t.png")
        # plt.savefig(fig, dpi=300)
        # # ========================
        # # x amplitude figure
        # # ========================

        # ========================
        # Variation of max p
        # ========================
        plt.clf()
        plt.plot(t, fluid_max_p, "o", label='Max pressure variation')

        plt.title('Variation in pressure')
        plt.xlabel('t')
        plt.ylabel('pressure')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "pressure_vs_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # ends variation of max p
        # ========================

        # ========================
        # Variation of max p
        # ========================
        plt.clf()
        plt.plot(t, fluid_max_y, "o", label='Max y variation')

        plt.title('Variation in max y')
        plt.xlabel('t')
        plt.ylabel('max y')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "max_y_vs_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # ends variation of max p
        # ========================



if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
