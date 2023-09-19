"""Poiseuille flow using the transport velocity formulation (5 minutes).
"""
import os

# numpy
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.solver.utils import load
from fluids import (get_particle_array_fluid,
                    get_particle_array_boundary,
                    FluidsScheme)
from pysph.sph.scheme import SchemeChooser
import pysph.tools.geometry as G
from pysph.tools.geometry import get_2d_tank, get_2d_block
from rigid_body import (get_particle_array_rigid_body,
                        set_linear_velocity_of_rigid_body,
                        set_angular_velocity)
from rigid_fluid_coupling import (ParticlesFluidScheme,
                                  add_rigid_fluid_properties_to_rigid_body)


# Numerical setup
dx = 1.0/80.0
ghost_extent = 5 * dx
hdx = 1.0

# adaptive time steps
h0 = hdx * dx


class PoiseuilleFlow(Application):
    def initialize(self):
        self.d = 0.5
        self.Ly = 2*self.d
        self.Lx = 2 * self.Ly
        self.rho0 = 1.0
        self.nu = 0.01

        # rigid body properties
        self.rigid_body_length = 0.2
        self.rigid_body_height = 0.2
        self.rigid_body_rho = 0.1

        self.dim = 2

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
        self.re = self.options.re
        self.Vmax = self.nu*self.re/(2*self.d)
        self.c0 = 10*self.Vmax
        self.p0 = self.c0**2*self.rho0

        # The body force is adjusted to give the Required Reynold's number
        # based on the steady state maximum velocity Vmax:
        # Vmax = fx/(2*nu)*(d^2) at the centerline

        self.fx = self.Vmax * 2*self.nu/(self.d**2)
        # Setup default parameters.
        dt_cfl = 0.25 * h0/( self.c0 + self.Vmax )
        dt_viscous = 0.125 * h0**2/self.nu
        dt_force = 0.25 * np.sqrt(h0/self.fx)

        self.dt = min(dt_cfl, dt_viscous, dt_force)

    def create_domain(self):
        return DomainManager(xmin=0, xmax=self.Lx, periodic_in_x=True)

    def create_particles(self):
        Lx = self.Lx
        Ly = self.Ly
        _x = np.arange( dx/2, Lx, dx )

        # create the fluid particles
        _y = np.arange( dx/2, Ly, dx )

        x, y = np.meshgrid(_x, _y); fx = x.ravel(); fy = y.ravel()

        # create the channel particles at the top
        _y = np.arange(Ly+dx/2, Ly+dx/2+ghost_extent, dx)
        x, y = np.meshgrid(_x, _y); tx = x.ravel(); ty = y.ravel()

        # create the channel particles at the bottom
        _y = np.arange(-dx/2, -dx/2-ghost_extent, -dx)
        x, y = np.meshgrid(_x, _y); bx = x.ravel(); by = y.ravel()

        # concatenate the top and bottom arrays
        cx = np.concatenate( (tx, bx) )
        cy = np.concatenate( (ty, by) )

        # create the arrays
        channel = get_particle_array_boundary(name='channel', x=cx, y=cy)
        fluid = get_particle_array_fluid(name='fluid', x=fx, y=fy)

        print("Poiseuille flow :: Re = %g, nfluid = %d, nchannel=%d"%(
            self.re, fluid.get_number_of_particles(),
            channel.get_number_of_particles()))

        # add requisite properties to the arrays:
        # self.scheme.setup_properties([fluid, channel])

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.rho0
        channel.m[:] = volume * self.rho0

        # Set the default rho.
        fluid.rho[:] = self.rho0
        channel.rho[:] = self.rho0

        # # volume is set as dx^2
        # fluid.V[:] = 1./volume
        # channel.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = hdx * dx
        channel.h[:] = hdx * dx
        # create rigid body
        x, y = get_2d_block(dx=dx,
                            length=self.rigid_body_length,
                            height=self.rigid_body_height)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        dem_id = body_id
        m = self.rigid_body_rho * dx**2
        h = hdx * dx
        rad_s = dx / 2.
        # y[:] += 2. * dx
        # y[:] -= 2. * self.rigid_body_length
        # x[:] = min(fluid.x) - min(x) + self.rigid_body_length
        y[:] += 0.3
        x[:] += 0.3
        rigid_body = get_particle_array_rigid_body(name='rigid_body',
                                                   x=x,
                                                   y=y,
                                                   z=0,
                                                   h=h,
                                                   m_rb=m,
                                                   dem_id=dem_id,
                                                   body_id=body_id)

        # print("rigid body total mass: ", rigid_body.total_mass)
        rigid_body.rho[:] = self.rho0
        G.remove_overlap_particles(
            fluid, rigid_body, dx, dim=self.dim
        )
        add_rigid_fluid_properties_to_rigid_body(rigid_body)
        # set_linear_velocity_of_rigid_body(rigid_body, [1., 1., 0.])
        rigid_body.m[:] = self.rho0 * dx**self.dim

        # return the particle list
        return [fluid, channel, rigid_body]

    def create_scheme(self):
        wcsph = ParticlesFluidScheme(
            fluids=['fluid'],
            boundaries=['channel'],
            rigid_bodies=["rigid_body"],
            dim=2, rho0=self.rho0, c0=0., pb=0., nu=self.nu,
            gy=0., alpha=0.05)

        s = SchemeChooser(default='wcsph', wcsph=wcsph)
        return s

    def configure_scheme(self):
        tf = 100.0
        scheme = self.scheme
        scheme.configure(c0=self.c0, pb=self.p0, gx=self.fx)
        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=1000)
        print("dt = %g"%self.dt)


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


if __name__ == '__main__':
    app = PoiseuilleFlow()
    app.run()
    app.post_process(app.info_filename)
