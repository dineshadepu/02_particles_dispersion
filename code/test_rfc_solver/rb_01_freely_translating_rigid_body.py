from __future__ import print_function
import numpy as np

import sys
sys.path.insert(0, "./../")
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
        self.dt = 1e-3

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
        set_linear_velocity_of_rigid_body(rigid_body, [1., 1., 0.])
        set_angular_velocity(rigid_body, [0., 0., 2 * np.pi])
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
        self.scheme.configure_solver(dt=self.dt, tf=tf, pfreq=100,
                                     output_at_times=output_at_times)

    def post_process(self, fname):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        files = files[:]
        t, total_energy = [], []
        x, y = [], []
        R = []
        ang_mom = []
        for sd, body in iter_output(files, 'rigid_body'):
            _t = sd['t']
            # print(_t)
            t.append(_t)
            total_energy.append(0.5 * np.sum(body.m[:] * (body.u[:]**2. +
                                                          body.v[:]**2.)))
            print("R is", body.R)
            print("ang_mom is", body.ang_mom)
            print("omega x is", body.omega)
            print("moi global master ", body.inertia_tensor_inverse_global_frame)
            print("moi body master ", body.inertia_tensor_inverse_body_frame)
            print("moi global master ", body.inertia_tensor_global_frame)
            print("moi body master ", body.inertia_tensor_body_frame)
            # x.append(body.xcm[0])
            # y.append(body.xcm[1])
            # print(body.ang_mom_z[0])
            ang_mom.append(body.ang_mom[2])
            R.append(body.R[0])
        # print(ang_mom)

        import matplotlib
        import os
        # matplotlib.use('Agg')

        from matplotlib import pyplot as plt

        # res = os.path.join(self.output_dir, "results.npz")
        # np.savez(res, t=t, amplitude=amplitude)

        # gtvf data
        # data = np.loadtxt('./oscillating_plate.csv', delimiter=',')
        # t_gtvf, amplitude_gtvf = data[:, 0], data[:, 1]

        plt.clf()

        # plt.plot(t_gtvf, amplitude_gtvf, "s-", label='GTVF Paper')
        # plt.plot(t, total_energy, "-", label='Simulated')
        # plt.plot(t, ang_mom, "-", label='Angular momentum')
        plt.plot(t, R, "-", label='R[0]')

        plt.xlabel('t')
        plt.ylabel('ang energy')
        plt.legend()
        fig = os.path.join(self.output_dir, "ang_mom_vs_t.png")
        plt.savefig(fig, dpi=300)
        plt.show()

        # plt.plot(x, y, label='Simulated')
        # plt.show()


if __name__ == '__main__':
    app = FreelyTranslatingRigidBody()
    app.run()
    app.post_process(app.info_filename)
