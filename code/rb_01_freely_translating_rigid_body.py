from __future__ import print_function
import numpy as np

from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from pysph.base.utils import (get_particle_array)

from rigid_body import (RigidBody3DScheme,
                        get_particle_array_rigid_body,
                        set_linear_velocity_of_rigid_body,
                        set_angular_velocity)
from pysph.sph.equation import Equation, Group
import os

from pysph.tools.geometry import get_2d_block, get_2d_tank, rotate


class FreelyTranslatingRigidBody(Application):
    def initialize(self):
        # Parameters specific to the world
        self.dim = 2
        spacing = 1 * 1e-1
        self.hdx = 1.5
        self.alpha = 0.1
        self.gx = 0.
        self.gy = - 9.81
        self.h = self.hdx * spacing

        # Physical parameters of the rigid body
        self.rigid_body_length = 1.
        self.rigid_body_height = 1.
        self.rigid_body_spacing = spacing
        self.rigid_body_rho = 2700

        # solver data
        self.tf = 1.
        self.dt = 1e-4

        self.seval = None

    def create_rigid_body(self):
        x = np.array([])
        y = np.array([])

        x, y = get_2d_block(dx=self.rigid_body_spacing,
                            length=self.rigid_body_length,
                            height=self.rigid_body_height)

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        dem_id = body_id
        m = self.rigid_body_rho * self.rigid_body_spacing**2
        h = self.h
        rad_s = self.rigid_body_spacing / 2.
        rigid_body = get_particle_array_rigid_body(name='rigid_body',
                                                   x=x,
                                                   y=y,
                                                   z=0,
                                                   h=h,
                                                   m_rb=m,
                                                   dem_id=dem_id,
                                                   body_id=body_id)
        # set_linear_velocity_of_rigid_body(rigid_body, [1., 1., 0.])
        # set_angular_velocity(rigid_body, [0., 0., 2 * np.pi])
        rigid_body.h[-1] = 1.5
        return rigid_body

    def create_particles(self):
        rigid_body = self.create_rigid_body()

        return [rigid_body]

    def create_scheme(self):
        rb3d = RigidBody3DScheme(rigid_bodies=['rigid_body'],
                                 boundaries=['wall'],
                                 gx=1e-5,
                                 gy=0.,
                                 gz=0.,
                                 dim=2,
                                 )
        s = SchemeChooser(default='rb3d', rb3d=rb3d)
        return s

    def configure_scheme(self):
        tf = self.tf

        output_at_times = np.array([0., 0.5, 1.0])
        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=500,
                                     output_at_times=output_at_times)


if __name__ == '__main__':
    app = FreelyTranslatingRigidBody()
    app.run()
    app.post_process(app.info_filename)
