"""A neutrally buoyand circular cylinder in a shear flow

Section 7.1 in Hashemi 2012 paper
"""
import os

import matplotlib.pyplot as plt
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
from rigid_body import (get_particle_array_rigid_body,
                        set_linear_velocity_of_rigid_body,
                        set_angular_velocity,
                        move_body_to_new_center,
                        get_center_of_mass,
                        color_diagonal_of_rb,
                        AdjustRigidBodyPositionInPipe)
from rigid_fluid_coupling import (ParticlesFluidScheme,
                                  add_rigid_fluid_properties_to_rigid_body)


class PoiseuilleFlow(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=0.625,
            help="Reynolds number of flow."
        )
        group.add_argument(
            "--pipe-diameter", action="store", type=float, dest="pipe_diameter",
            default=5.017, help="Diameter of the pipe (mm)."
        )

        group.add_argument(
            "--pipe-length", action="store", type=float, dest="pipe_length",
            default=20.000, help="Length of the pipe (mm)."
        )

        group.add_argument(
            "--sphere-diameter", action="store", type=float, dest="sphere_diameter",
            default=0.931, help="Diameter of the sphere (mm)."
        )

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        self.pipe_diameter = self.options.pipe_diameter * 1e-3
        self.pipe_length = self.options.pipe_length * 1e-3
        # this is a variable
        self.sphere_diameter = self.options.sphere_diameter * 1e-3
        # ======================
        # ======================

        # ======================
        # Dimensions
        # ======================
        self.rigid_body_diameter = self.sphere_diameter
        self.pipe_bdry_layers = 3
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
        self.dim = 3
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
        self.nu = 1e-6
        self.mu = self.nu * self.fluid_rho
        print("Kinematic viscosity is: ", self.nu)
        self.tf = 80.
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.02

        # Setup default parameters.
        # dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        dt_viscous = 0.125 * self.h**2/self.nu
        # dt_force = 0.25 * np.sqrt(self.h/(self.gy))
        # dt_force = 0.25 * np.sqrt(self.h/10)
        # print("dt viscous is: ", dt_viscous)
        # print("dt force is: ", dt_force)
        # self.dt = min(dt_viscous, dt_force)
        self.dt = 1e-4
        # self.dt = dt_viscous
        print("dt is: ", self.dt)

        # ==========================
        # Numerical properties ends
        # ==========================

        # ====================================================
        # Start: properties to be used while adjusting the equations
        # ====================================================
        self.x_min = 0.0
        self.x_max = self.pipe_length
        # print("x min is ", self.x_min)
        # print("x max is ", self.x_max)
        # ====================================================
        # end: properties to be used while adjusting the equations
        # ====================================================

    def create_particles(self):
        # Length wise axis is x, and at a cross section it is y and z.
        # We create particles at a given cross section.
        # diameter of the pipe including the boundaries
        dia_new = self.pipe_diameter + (self.pipe_bdry_layers + 3) * self.dx
        _y, _z = create_circle_1(dia_new, self.dx)
        # Now create the pipe by stacking the particles together
        _x = np.arange(0., self.pipe_length, self.dx)
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for i in range(len(_x)):
            x_local = np.ones_like(_y) * _x[i]
            x = np.concatenate((x, x_local))
            y = np.concatenate((y, _y))
            z = np.concatenate((z, _z))

        # find the indices which are not fluid
        indices = np.where(np.sqrt(z[i]*z[i] + y[i]*y[i]) - self.pipe_diameter > 1e-10)[0]

        # remove particles outside the circle
        indices_fluid = []
        indices_channel = []
        for i in range(len(x)):
            if np.sqrt(z[i]*z[i] + y[i]*y[i]) - self.pipe_diameter/2. > 1e-10:
                indices_channel.append(i)
            else:
                indices_fluid.append(i)

        xt, yt, zt = x[indices_channel], y[indices_channel], z[indices_channel]
        xf, yf, zf = x[indices_fluid], y[indices_fluid], z[indices_fluid]

        m = self.dx**self.dim * self.fluid_rho
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=zf, h=self.h, m=m, rho=self.fluid_rho)
        channel = get_particle_array_boundary(name='channel', x=xt, y=yt, z=zt, h=self.h, m=m, rho=self.fluid_rho)
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
        # center = [self.fluid_length/3., 0.75 * self.fluid_height, 0.]
        # center = [self.rigid_body_diameter, 0.75 * self.fluid_height, 0.]
        # center = [self.fluid_length - 10. * self.dx, 0.75 * self.fluid_height, 0.]
        # center = [self.rigid_body_diameter, 2. * self.fluid_height, 0.]
        # xcm = get_center_of_mass(x, y, z, m)
        # move_body_to_new_center(xcm, x, y, z, center)

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

    # def create_domain(self):
    #     return DomainManager(xmin=0, xmax=self.fluid_length, periodic_in_x=True)

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

        return eqns

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
            print(_t)

            vertical_position_current.append(rigid_body.xcm[1])
            u_cm.append(rigid_body.vcm[0])
            t_current.append(_t)
        t_currrent = np.asarray(t_current)
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
