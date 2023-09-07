"""Skillen circular water entry

Run it using:

python skillen_2013_circular_water_entry.py --openmp --use-edac --arti --alpha 0. --max-s 200 --pfreq 100 --nu 1e-3 --dx 0.0015

"""
import numpy as np
import sys

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.tools.geometry import get_3d_sphere
import pysph.tools.geometry as G
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
sys.path.insert(0, "./../")
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
    if t < 0.0:
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
        # group.add_argument(
        #     "--remesh", action="store", type=float, dest="remesh", default=0,
        #     help="Remeshing frequency (setting it to zero disables it)."
        # )
        group.add_argument("--N", action="store", type=int, dest="N",
                           default=30,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--rigid-body-rho", action="store", type=float,
                           dest="rigid_body_rho", default=500,
                           help="Density of rigid cylinder")

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        # self.re = self.options.re
        # ======================
        # ======================

        # ======================
        # Dimensions
        # ======================
        # x - axis
        self.fluid_length = 1.
        # y - axis
        self.fluid_height = 0.3
        # z - axis
        self.fluid_depth = 0.0

        # x - axis
        self.tank_length = 1.
        # y - axis
        self.tank_height = 0.4
        # z - axis
        self.tank_depth = 0.0
        self.tank_layers = 3

        # self.rigid_body_length = 0.2
        # self.rigid_body_height = 0.2
        self.rigid_body_diameter = 0.11
        self.rigid_body_center = [0.00, 0.3 + 0.5]
        self.rigid_body_velocity = 2.955
        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.rigid_body_rho = self.options.rigid_body_rho

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
        self.N = self.options.N
        self.dx = self.rigid_body_diameter / self.N
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = self.options.nu
        self.tf = 0.18
        # self.tf = 0.56 - 0.3192
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.00

        # Setup default parameters.
        dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        dt_viscous = 1e5
        if self.nu > 1e-12:
            dt_viscous = 0.125 * self.h**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h/(np.abs(self.gy)))
        print("dt_cfl", dt_cfl, "dt_viscous", dt_viscous, "dt_force", dt_force)

        self.dt = min(dt_cfl, dt_force)
        print("Computed stable dt is: ", self.dt)
        self.dt = 1e-5
        # ==========================
        # Numerical properties ends
        # ==========================

    def create_particles(self):
        from wall_normal import get_normals
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length, self.fluid_height,
                                             self.tank_height, self.tank_layers,
                                             self.dx, self.dx, False)
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

        if self.options.nrbc == True:
            props = ['j3', 'pta', 'j3v', 'j2v', 'j2', 'vta', 'uta', 'j1'] # required for non-reflecting boundary
            for pa in (fluid, tank):
                for prop in props:
                    pa.add_property(prop)
                pa.pta[:] = 0.0
                pa.uta[:] = 0.0
                pa.vta[:] = 0.0

            get_normals(tank, dim=2, domain=self.domain)

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_rho * self.gy * (max(fluid.y) - fluid.y[:])
        fluid.c0_ref[0] = self.c0
        fluid.p0_ref[0] = self.p0

        fluid.add_output_arrays(['auhat', 'avhat', 'awhat', 'uhat', 'vhat', 'what'])

        # =========================
        # create rigid body
        # =========================
        x, y = create_circle_1(self.rigid_body_diameter,
                               self.dx)
        center = [self.fluid_length/2., self.fluid_height + self.rigid_body_diameter/2. + self.dx, 0.]
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
                                                   m=self.fluid_rho * self.dx**self.dim,
                                                   rho=self.fluid_rho,
                                                   dem_id=dem_id,
                                                   body_id=body_id)

        # print("rigid body total mass: ", rigid_body.total_mass)
        # rigid_body.rho[:] = self.fluid_rho
        G.remove_overlap_particles(
            fluid, rigid_body, self.dx, dim=self.dim
        )
        add_rigid_fluid_properties_to_rigid_body(rigid_body)
        set_linear_velocity_of_rigid_body(rigid_body, [0., -self.rigid_body_velocity, 0.])
        indices_rb_inside_cond = (rigid_body.normal_norm == 0.)
        indices = np.where(indices_rb_inside_cond == True)
        rigid_body.remove_particles(indices[0])
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

        # initial position of the cylinder
        fname = files[0]
        data = load(fname)
        rigid_body = data['arrays']['rigid_body']
        y_0 = rigid_body.xcm[1]

        y = []
        v = []
        u = []
        t = []

        for sd, rigid_body in iter_output(files, 'rigid_body'):
            _t = sd['t']
            y.append(rigid_body.xcm[1])
            u.append(rigid_body.vcm[0])
            v.append(rigid_body.vcm[1])
            t.append(_t)
        # non dimentionalize it
        penetration_current = (np.asarray(y)[::1] - y_0) / self.rigid_body_diameter
        t_current = np.asarray(t)[::1] * (9.81 / self.rigid_body_diameter)**0.5

        # Data from literature
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        # We use Sun 2018 accurate and efficient water entry paper data for validation
        if self.rigid_body_rho == 500.:
            data_y_penetration_sun_2018_exp = np.loadtxt(os.path.join(
                directory, 'sun_2018_falling_500_rho_experimental_data.csv'), delimiter=',')
            data_y_penetration_sun_2018_BEM = np.loadtxt(os.path.join(
                directory, 'sun_2018_falling_500_rho_BEM_data.csv'), delimiter=',')
            # This is 200 resolution D / dx = 200
            data_y_penetration_sun_2018_SPH = np.loadtxt(os.path.join(
                directory, 'sun_2018_falling_500_rho_SPH_data.csv'), delimiter=',')
        if self.rigid_body_rho == 1000.:
            data_y_penetration_sun_2018_exp = np.loadtxt(os.path.join(
                directory, 'sun_2018_falling_1000_rho_experimental_data.csv'), delimiter=',')
            data_y_penetration_sun_2018_BEM = np.loadtxt(os.path.join(
                directory, 'sun_2018_falling_1000_rho_BEM_data.csv'), delimiter=',')
            # This is 200 resolution D / dx = 200
            data_y_penetration_sun_2018_SPH = np.loadtxt(os.path.join(
                directory, 'sun_2018_falling_1000_rho_SPH_data.csv'), delimiter=',')

        t_exp, penetration_exp = data_y_penetration_sun_2018_exp[:, 0], data_y_penetration_sun_2018_exp[:, 1]
        t_BEM, penetration_BEM = data_y_penetration_sun_2018_BEM[:, 0], data_y_penetration_sun_2018_BEM[:, 1]
        t_SPH, penetration_SPH = data_y_penetration_sun_2018_SPH[:, 0], data_y_penetration_sun_2018_SPH[:, 1]
        # =================
        # sort webplot data
        p = t_SPH.argsort()
        t_SPH = t_SPH[p]
        penetration_SPH = penetration_SPH[p]
        # t_SPH = np.delete(t_SPH, np.where(t_SPH > 1.65 and t_SPH < 1.75))
        # penetration_SPH = np.delete(penetration_SPH,  np.where(t_SPH > 1.65 and t_SPH < 1.75))
        t_SPH = np.delete(t_SPH, -4)
        penetration_SPH = np.delete(penetration_SPH, -4)

        p = t_BEM.argsort()
        t_BEM = t_BEM[p]
        penetration_BEM = penetration_BEM[p]
        # sort webplot data
        # =================

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 t_exp=t_exp,
                 penetration_exp=penetration_exp,
                 t_BEM=t_BEM,
                 penetration_BEM=penetration_BEM,
                 t_SPH=t_SPH,
                 penetration_SPH=penetration_SPH,
                 t_current=t_current,
                 penetration_current=-penetration_current)
        data = np.load(res)

        # ========================
        # Variation of y penetration
        # ========================
        plt.clf()
        plt.plot(t_exp, penetration_exp, "^", label='Experimental')
        plt.plot(t_SPH, penetration_SPH, "-+", label='SPH')
        plt.plot(t_BEM, penetration_BEM, "--", label='BEM')
        plt.plot(t_current, -penetration_current, "-", label='Current')

        plt.title('Variation in y-penetration')
        plt.xlabel('t (g / D)^{1/2}')
        plt.ylabel('Penetration (y - y_0)/D')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "penetration_vs_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
