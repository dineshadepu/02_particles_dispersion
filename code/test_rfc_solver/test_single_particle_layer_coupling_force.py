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
import sys
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
    if t < 2.0:
        return True
    else:
        return False


def add_single_particle_layer_rfc_properties(pa):
    # for normals
    pa.add_property('closest_rb_particle_dist')
    pa.add_property('closest_rb_particle_idx', type='int')
    pa.add_property('closest_rb_particle_nx')
    pa.add_property('closest_rb_particle_ny')
    pa.add_property('closest_rb_particle_nz')
    pa.add_property('closest_rb_particle_x')
    pa.add_property('closest_rb_particle_y')
    pa.add_property('closest_rb_particle_z')

    pa.closest_rb_particle_idx[:] = -1

    pa.add_property('fx_rfc')
    pa.add_property('fy_rfc')
    pa.add_property('fz_rfc')


class FindClosestRigidBodyParticle(Equation):
    def initialize(self, d_idx, d_h,
                   d_closest_rb_particle_dist,
                   d_closest_rb_particle_idx,
                   d_closest_rb_particle_nx,
                   d_closest_rb_particle_ny,
                   d_closest_rb_particle_nz):
        d_closest_rb_particle_dist[d_idx] = 1. * d_h[d_idx]
        d_closest_rb_particle_idx[d_idx] = -1
        d_closest_rb_particle_nx[d_idx] = 0.
        d_closest_rb_particle_ny[d_idx] = 0.
        d_closest_rb_particle_nz[d_idx] = 0.

    def loop(self, d_idx, RIJ, s_idx, s_x, s_y, s_z,
             d_closest_rb_particle_dist,
             d_closest_rb_particle_idx,
             d_closest_rb_particle_nx,
             d_closest_rb_particle_ny,
             d_closest_rb_particle_nz,
             d_closest_rb_particle_x,
             d_closest_rb_particle_y,
             d_closest_rb_particle_z,
             s_normal):
        sidx3 = declare('int')
        sidx3 = 3 * s_idx

        if RIJ < d_closest_rb_particle_dist[d_idx]:
            d_closest_rb_particle_dist[d_idx] = RIJ
            d_closest_rb_particle_idx[d_idx] = s_idx
            d_closest_rb_particle_nx[d_idx] = s_normal[sidx3]
            d_closest_rb_particle_ny[d_idx] = s_normal[sidx3+1]
            d_closest_rb_particle_nz[d_idx] = s_normal[sidx3+2]
            d_closest_rb_particle_x[d_idx] = s_x[s_idx]
            d_closest_rb_particle_y[d_idx] = s_y[s_idx]
            d_closest_rb_particle_z[d_idx] = s_z[s_idx]


class SingleParticleLayerCouplingForce(Equation):
    """This is acted on the fluid particles
    """
    def __init__(self, dest, sources, spacing, rfc_eta):
        self.spacing = spacing
        self.rfc_eta = rfc_eta
        super(SingleParticleLayerCouplingForce, self).__init__(dest, sources)

    def post_loop(self, d_idx, d_fx_rfc, d_fy_rfc, d_fz_rfc,
                  d_x, d_y, d_z, d_m, d_au, d_av, d_aw,
                  d_closest_rb_particle_dist,
                  d_closest_rb_particle_idx,
                  d_closest_rb_particle_nx,
                  d_closest_rb_particle_ny,
                  d_closest_rb_particle_nz,
                  d_closest_rb_particle_x,
                  d_closest_rb_particle_y,
                  d_closest_rb_particle_z,
                  dt):
        if d_closest_rb_particle_idx[d_idx] == -1:
            d_fx_rfc[d_idx] = 0.
            d_fy_rfc[d_idx] = 0.
            d_fz_rfc[d_idx] = 0.
        else:
            # eq 28 of Peng
            dx = d_closest_rb_particle_x[d_idx] - d_x[d_idx]
            dy = d_closest_rb_particle_y[d_idx] - d_y[d_idx]
            dz = d_closest_rb_particle_z[d_idx] - d_z[d_idx]

            d = (dx * d_closest_rb_particle_nx[d_idx] + dy *
                 d_closest_rb_particle_ny[d_idx] + dz *
                 d_closest_rb_particle_nz[d_idx])

            if d < self.spacing / 2.:
                tmp = self.rfc_eta * d_m[d_idx] / (dt * dt)
                dp = self.spacing / 2. - d
                d_fx_rfc[d_idx] = tmp * dp * d_closest_rb_particle_nx[d_idx]
                d_fy_rfc[d_idx] = tmp * dp * d_closest_rb_particle_ny[d_idx]
                d_fz_rfc[d_idx] = tmp * dp * d_closest_rb_particle_nz[d_idx]

                d_au[d_idx] += tmp * dp * d_closest_rb_particle_nx[d_idx] / d_m[d_idx]
                d_av[d_idx] += tmp * dp * d_closest_rb_particle_ny[d_idx] / d_m[d_idx]
                d_aw[d_idx] += tmp * dp * d_closest_rb_particle_nz[d_idx] / d_m[d_idx]
            else:
                d_fx_rfc[d_idx] = 0.
                d_fy_rfc[d_idx] = 0.
                d_fz_rfc[d_idx] = 0.


class MakeForcesZeroOnRigidBody(Equation):
    # def initialize(self, d_idx, d_fx, d_fy, d_fz):
    #     d_fx[d_idx] = 0.
    #     d_fy[d_idx] = 0.
    #     d_fz[d_idx] = 0.

    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')

        frc = dst.force
        trq = dst.torque

        frc[:] = 0
        trq[:] = 0


class ComputeTotalForceTorque(Equation):
    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        dx0 = declare('object')
        dy0 = declare('object')
        dz0 = declare('object')
        xcm = declare('object')
        R = declare('object')
        total_mass = declare('object')
        body_id = declare('object')
        j = declare('int')
        i = declare('int')
        i3 = declare('int')
        i9 = declare('int')

        frc = dst.force_test
        trq = dst.torque_test
        fx = dst.fx
        fy = dst.fy
        fz = dst.fz
        x = dst.x
        y = dst.y
        z = dst.z
        dx0 = dst.dx0
        dy0 = dst.dy0
        dz0 = dst.dz0
        xcm = dst.xcm
        R = dst.R
        total_mass = dst.total_mass
        body_id = dst.body_id

        frc[:] = 0
        trq[:] = 0

        for j in range(len(x)):
            i = body_id[j]
            i3 = 3 * i
            i9 = 9 * i
            frc[i3] += fx[j]
            frc[i3 + 1] += fy[j]
            frc[i3 + 2] += fz[j]

            # torque due to force on particle i
            # (r_i - com) \cross f_i

            # get the local vector from particle to center of mass
            dx = (R[i9 + 0] * dx0[j] + R[i9 + 1] * dy0[j] +
                  R[i9 + 2] * dz0[j])
            dy = (R[i9 + 3] * dx0[j] + R[i9 + 4] * dy0[j] +
                  R[i9 + 5] * dz0[j])
            dz = (R[i9 + 6] * dx0[j] + R[i9 + 7] * dy0[j] +
                  R[i9 + 8] * dz0[j])

            # dx = x[j] - xcm[i3]
            # dy = y[j] - xcm[i3 + 1]
            # dz = z[j] - xcm[i3 + 2]

            # torque due to force on particle i
            # dri \cross fi
            trq[i3] += (dy * fz[j] - dz * fy[j])
            trq[i3 + 1] += (dz * fx[j] - dx * fz[j])
            trq[i3 + 2] += (dx * fy[j] - dy * fx[j])

        # # add body force
        # for i in range(max(body_id) + 1):
        #     i3 = 3 * i
        #     frc[i3] += total_mass[i] * self.gx
        #     frc[i3 + 1] += total_mass[i] * self.gy
        #     frc[i3 + 2] += total_mass[i] * self.gz


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
        self.fluid_height = 0.03
        # z - axis
        self.fluid_depth = 0.2

        # x - axis
        self.tank_length = 0.2
        # y - axis
        self.tank_height = 0.04
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
        self.rigid_body_rho = 500.

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
        self.tf = 0.1
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = self.options.alpha
        self.rfc_eta = 0.01

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

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_rho * self.gy * (max(fluid.y) - fluid.y[:])
        fluid.c0_ref[0] = self.c0

        # add single_particle_layer rfc properties
        add_single_particle_layer_rfc_properties(fluid)

        # =========================
        # create rigid body
        # =========================
        x, y = create_circle_1(self.rigid_body_diameter,
                               self.dx)
        center = [0.01, 0.01, 0.]
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
                                                   rho=self.fluid_rho,
                                                   m=self.fluid_rho * self.dx**self.dim,
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
        rigid_body.add_constant('force_test', [0., 0., 0.])
        rigid_body.add_constant('torque_test', [0., 0., 0.])

        # copy the normal
        rigid_body.normal0[:] = rigid_body.normal[:]
        rigid_body.add_output_arrays(['p'])
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
        eqns = self.scheme.get_equations()

        # compute total torque for post processing
        compute_frc = []
        compute_frc.append(
            ComputeTotalForceTorque("rigid_body", sources=None))

        eqns.groups[-1].append(Group(equations=compute_frc))

        # Make forces zero on rb
        zero_frc = []
        zero_frc.append(
            MakeForcesZeroOnRigidBody("rigid_body", sources=None))

        eqns.groups[-1].append(Group(equations=zero_frc,
                                         condition=check_time_make_zero))

        # Force on fluid due to single particle layer solid body
        tmp = []
        tmp.append(FindClosestRigidBodyParticle("fluid",
                                                sources=["rigid_body"]))
        tmp.append(SingleParticleLayerCouplingForce("fluid",
                                                    sources=None,
                                                    rfc_eta=self.rfc_eta,
                                                    spacing=self.dx))

        eqns.groups[-1].append(Group(equations=tmp))
        # remove the rigid body in pressure gradient equation as we are
        # applying the force here
        eqns.groups[-1][4].equations[1].sources = ['fluid', 'tank']

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
        frc_x = []
        frc_y = []
        gravity_force = []
        # torque_x = []
        # torque_y = []
        torque_z = []
        min_pressure = []
        max_pressure = []

        for sd, rigid_body in iter_output(files, 'rigid_body'):
            _t = sd['t']
            print(_t)
            # y.append(rigid_body.xcm[1])
            # u.append(rigid_body.vcm[0])
            # v.append(rigid_body.vcm[1])
            frc_x.append(rigid_body.force_test[0])
            frc_y.append(rigid_body.force_test[1])
            torque_z.append(rigid_body.torque_test[2])
            min_pressure.append(min(rigid_body.p))
            max_pressure.append(max(rigid_body.p))
            gravity_force.append(rigid_body.total_mass[0] * 9.81)

            t.append(_t)

        # # Data from literature
        # path = os.path.abspath(__file__)
        # directory = os.path.dirname(path)

        # # load the data
        # data_y_vel_hashemi_2012_fem = np.loadtxt(os.path.join(
        #     directory, 'hashemi_2012_falling_cylinder_fem_data.csv'), delimiter=',')

        # t_has, vel_has = data_y_vel_hashemi_2012_fem[:, 0], data_y_vel_hashemi_2012_fem[:, 1]

        # res = os.path.join(self.output_dir, "results.npz")
        # np.savez(res, t_has=t_has, vel_has=vel_has)

        # ========================
        # Variation of y velocity
        # ========================
        plt.clf()
        # plt.plot(t_has, vel_has, "-", label='FEM')
        plt.plot(t, frc_x, "-", label='Frc x')
        plt.plot(t, gravity_force, "-", label='Gravity')
        # plt.title('Variation in y-velocity')
        plt.xlabel('t')
        plt.ylabel('Force-x')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "x_force_vs_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # x amplitude figure
        # ========================

        plt.clf()
        # plt.plot(t_has, vel_has, "-", label='FEM')
        plt.plot(t, frc_y, "-", label='Frc y')
        plt.plot(t, gravity_force, "-", label='Gravity')
        # plt.title('Variation in y-velocity')
        plt.xlabel('t')
        plt.ylabel('Force-y')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "y_force_vs_t.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        # plt.plot(t_has, vel_has, "-", label='FEM')
        plt.plot(t, min_pressure, "-", label='Min pressure')
        plt.plot(t, max_pressure, "-", label='Max pressure')
        # plt.title('Variation in y-velocity')
        plt.xlabel('t')
        plt.ylabel('Pressure variation')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "pressure_vs_t.png")
        plt.savefig(fig, dpi=300)



if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
# fx_rfc, fy_rfc, fz_rfc

# normal[::3], normal[1::3], normal[2::3]
