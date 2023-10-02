"""A neutrally buoyand circular cylinder in a shear flow

Section 7.1 in Hashemi 2012 paper

python hashemi_2012_neutrally_buoyant_circular_cylinder_in_shear_flow.py --openmp --no-use-edac --nu 1e-6 --alpha 0.02 --artificial-viscosity-with-boundary --pfreq 300 --internal-flow --pst

"""
import os

# numpy
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.solver.utils import load
import pysph.tools.geometry as G
import sys
sys.path.insert(0, "./../")
from fluids import (get_particle_array_fluid,
                    get_particle_array_boundary,
                    FluidsScheme)
from pysph.sph.scheme import SchemeChooser
from pysph.sph.equation import Group, MultiStageEquations

from geometry import hydrostatic_tank_2d, create_circle_1
from pysph.sph.equation import Equation, Group
from rigid_body import (get_particle_array_rigid_body,
                        set_linear_velocity_of_rigid_body,
                        set_angular_velocity,
                        move_body_to_new_center,
                        get_center_of_mass,
                        color_diagonal_of_rb,
                        AdjustRigidBodyPositionInPipe)
from rigid_fluid_coupling import (ParticlesFluidScheme,
                                  add_rigid_fluid_properties_to_rigid_body)


def check_time_make_zero(t, dt):
    if t < 5.0:
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


class PoiseuilleFlow(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=0.625,
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
        H = 0.01
        L = 5 * H
        # x - axis
        self.fluid_length = L
        # y - axis
        self.fluid_height = H
        # z - axis
        self.fluid_depth = 0.0

        # x - axis
        self.tank_length = 0.0
        # y - axis
        self.tank_height = 0.0
        # z - axis
        self.tank_depth = 0.0
        self.tank_layers = 4

        R = 1. / 8. * H
        self.rigid_body_diameter = R * 2.
        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Velocity
        # ======================
        self.Umax = 0.02
        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.rigid_body_rho = 1000

        self.gx = 0.
        self.gy = 0.
        self.gz = 0.
        self.dim = 2
        # ======================
        # Physical properties and consants ends
        # ======================

        # ======================
        # Numerical properties
        # ======================
        self.hdx = 1.
        # self.dx = 1.0/60.0
        self.dx = self.rigid_body_diameter / 10
        self.h = self.hdx * self.dx
        self.vref = self.Umax
        self.c0 = 10.0 * self.Umax
        self.mach_no = self.vref / self.c0
        # set the viscosity based on the particle reynolds no
        self.nu = self.Umax * self.rigid_body_diameter**2. / (self.fluid_height * self.re)
        self.mu = self.nu * self.fluid_rho
        print("Kinematic viscosity is: ", self.nu)
        self.tf = 83.
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.02

        # Setup default parameters.
        # dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        dt_viscous = 0.125 * self.h**2/self.nu
        # dt_force = 0.25 * np.sqrt(self.h/(self.gy))
        dt_force = 0.25 * np.sqrt(self.h/10)
        print("dt viscous is: ", dt_viscous)
        print("dt force is: ", dt_force)
        self.dt = min(dt_viscous, dt_force)
        # self.dt = 1e-4
        # self.dt = dt_viscous
        print("dt is: ", self.dt)

        # ==========================
        # Numerical properties ends
        # ==========================

        # ====================================================
        # Start: properties to be used while adjusting the equations
        # ====================================================
        Lx = self.fluid_length
        Ly = self.fluid_height
        _x = np.arange(self.dx/2, self.fluid_length, self.dx)

        # create the fluid particles
        _y = np.arange(self.dx/2, self.fluid_height, self.dx)

        x, y = np.meshgrid(_x, _y); fx = x.ravel(); fy = y.ravel()
        # self.x_min = min(fx)
        # self.x_max = max(fx)
        self.x_min = 0.0
        self.x_max = self.fluid_length
        print("x min is ", self.x_min)
        print("x max is ", self.x_max)
        # ====================================================
        # end: properties to be used while adjusting the equations
        # ====================================================

    def create_particles(self):
        Lx = self.fluid_length
        Ly = self.fluid_height
        _x = np.arange(self.dx/2, self.fluid_length, self.dx)

        # create the fluid particles
        _y = np.arange(self.dx/2, self.fluid_height, self.dx)

        x, y = np.meshgrid(_x, _y); fx = x.ravel(); fy = y.ravel()

        # create the channel particles at the top
        _y = np.arange(self.fluid_height+self.dx/2,
                       self.fluid_height+self.dx/2+self.tank_layers*self.dx,
                       self.dx)
        x, y = np.meshgrid(_x, _y); tx = x.ravel(); ty = y.ravel()

        # create the channel particles at the bottom
        _y = np.arange(-self.dx/2, -self.dx/2-self.tank_layers*self.dx, -self.dx)
        x, y = np.meshgrid(_x, _y); bx = x.ravel(); by = y.ravel()

        # concatenate the top and bottom arrays
        cx = np.concatenate((tx, bx))
        cy = np.concatenate((ty, by))

        # create the arrays
        channel = get_particle_array_boundary(name='channel', x=cx, y=cy)
        fluid = get_particle_array_fluid(name='fluid', x=fx, y=fy)

        # set velocities of the top particles of the channel
        channel.u[channel.y > self.fluid_height / 2.] = self.Umax / 2.
        channel.u[channel.y < self.fluid_height / 2.] = - self.Umax / 2.

        print("Poiseuille flow :: Re = %g, nfluid = %d, nchannel=%d"%(
            self.re, fluid.get_number_of_particles(),
            channel.get_number_of_particles()))

        # add requisite properties to the arrays:
        # self.scheme.setup_properties([fluid, channel])

        # setup the particle properties
        volume = self.dx * self.dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.fluid_rho
        channel.m[:] = volume * self.fluid_rho

        # Set the default rho.
        fluid.rho[:] = self.fluid_rho
        channel.rho[:] = self.fluid_rho

        # # volume is set as dx^2
        # fluid.V[:] = 1./volume
        # channel.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = self.h
        channel.h[:] = self.h

        # =========================
        # create rigid body
        # =========================
        x, y = create_circle_1(self.rigid_body_diameter, self.dx)
        z = np.zeros_like(x)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        dem_id = body_id
        m = self.rigid_body_rho * self.dx**self.dim * np.ones_like(x)
        h = self.h
        rad_s = self.dx / 2.
        # y[:] += 2. * dx
        # y[:] -= 2. * self.rigid_body_length
        # x[:] = min(fluid.x) - min(x) + self.rigid_body_length
        # y[:] += 0.3
        # x[:] += 0.3
        center = [self.fluid_length/3., 0.75 * self.fluid_height, 0.]
        # center = [self.rigid_body_diameter, 0.75 * self.fluid_height, 0.]
        # center = [self.fluid_length - 10. * self.dx, 0.75 * self.fluid_height, 0.]
        # center = [self.rigid_body_diameter, 2. * self.fluid_height, 0.]
        xcm = get_center_of_mass(x, y, z, m)
        move_body_to_new_center(xcm, x, y, z, center)

        rigid_body = get_particle_array_rigid_body(name='rigid_body',
                                                   x=x,
                                                   y=y,
                                                   z=z,
                                                   h=h,
                                                   m_rb=m,
                                                   dem_id=dem_id,
                                                   body_id=body_id)

        color_diagonal_of_rb(rigid_body)
        rigid_body.add_output_arrays(['color_diagonal'])

        # print("rigid body total mass: ", rigid_body.total_mass)
        rigid_body.rho[:] = self.fluid_rho
        G.remove_overlap_particles(
            fluid, rigid_body, self.dx, dim=self.dim
        )
        add_rigid_fluid_properties_to_rigid_body(rigid_body)
        # set_linear_velocity_of_rigid_body(rigid_body, [0.1, 0., 0.])
        rigid_body.m[:] = self.fluid_rho * self.dx**self.dim
        # =========================
        # create rigid body ends
        # =========================

        # return the particle list
        return [fluid, channel, rigid_body]

    def create_domain(self):
        return DomainManager(xmin=0, xmax=self.fluid_length, periodic_in_x=True)

    def create_scheme(self):
        wcsph = ParticlesFluidScheme(
            fluids=['fluid'],
            boundaries=['channel'],
            rigid_bodies=["rigid_body"],
            dim=0.,
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

    def create_equations(self):
        # print("inside equations")
        eqns = self.scheme.get_equations()

        # Apply external force
        adjust_eqs = []
        adjust_eqs.append(
            AdjustRigidBodyPositionInPipe(
                "rigid_body", sources=None, x_min=self.x_min, x_max=self.x_max))

        eqns.groups[0].append(Group(adjust_eqs))

        # Apply external force
        zero_frc = []
        zero_frc.append(
            MakeForcesZeroOnRigidBody("rigid_body", sources=None))

        # print(eqns.groups)
        eqns.groups[-1].append(Group(equations=zero_frc,
                                         condition=check_time_make_zero))

        return eqns

    def create_tools(self):
        tools = []
        if self.options.remesh > 0:
            from pysph.solver.tools import SimpleRemesher
            remesher = SimpleRemesher(
                self, 'fluid', props=['u', 'v', 'uhat', 'vhat'],
                freq=self.options.remesh
            )
            tools.append(remesher)
        return tools

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load, get_files

        output_files = get_files(os.path.dirname(fname))

        from pysph.solver.utils import iter_output

        files = output_files

        t_current = []

        vertical_position_current = []
        u_cm = []

        step = 1
        for sd, rigid_body in iter_output(files[::step], 'rigid_body'):
            _t = sd['t']
            if _t > 3:
                print(_t)

                vertical_position_current.append(rigid_body.xcm[1])
                u_cm.append(rigid_body.vcm[0])
                t_current.append(_t)
        t_current = np.asarray(t_current)
        t_current -= 3.
        vertical_position_current = np.asarray(vertical_position_current)

        # Data from literature
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        # We use Sun 2018 accurate and efficient water entry paper data for validation
        data_vertical_postion_feng_2018_exp = np.loadtxt(os.path.join(
            directory, 'hashemi_2012_neutrally_inertial_migration_Z_G_Feng_2002_vertical_position_data.csv'), delimiter=',')

        t_feng, vertical_position_feng = data_vertical_postion_feng_2018_exp[:, 0], data_vertical_postion_feng_2018_exp[:, 1]

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 t_feng, vertical_position_feng,
                 t_current, vertical_position_current)

        plt.plot(t_current, vertical_position_current, label="Current SPH")
        plt.plot(t_feng, vertical_position_feng, label="Feng 2002")
        plt.ylabel("Vertical position (m)")
        plt.xlabel("Time (seconds)")
        plt.legend()
        fig = os.path.join(self.output_dir, "t_vs_y_cm.png")
        plt.savefig(fig, dpi=300)
        plt.clf()


if __name__ == '__main__':
    app = PoiseuilleFlow()
    app.run()
    app.post_process(app.info_filename)
