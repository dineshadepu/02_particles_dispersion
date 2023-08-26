import numpy
import numpy as np

from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator_step import IntegratorStep
from pysph.sph.scheme import Scheme
from pysph.sph.scheme import add_bool_argument
from pysph.base.utils import get_particle_array
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.base.kernels import (QuinticSpline)
from pysph.examples.solid_mech.impact import add_properties

from pysph.sph.integrator import Integrator

from fluids import (ContinuityEquation, StateEquation, MomentumEquationPressureGradient, FluidStep,
                    SolidWallNoSlipBC, EDACEquation, DeltaSPHCorrection)
from rigid_body import (SumUpExternalForces)


def add_rigid_fluid_properties_to_rigid_body(pa):
    add_properties(pa, 'arho')
    add_properties(pa, 'm_fluid')
    add_properties(pa, 'm_frac')
    add_properties(pa, 'wij')
    add_properties(pa, 'ug', 'vf', 'uf', 'wf', 'vg', 'wg')


class EvaluateSlipWallVelocity(Equation):
    def __init__(self, dest, sources, rho0, c0):
        self.rho0 = rho0
        self.c0 = c0
        super(EvaluateSlipWallVelocity, self).__init__(dest, sources)

    def initialize(self, d_v, d_j2v, d_idx, d_vta, d_u, d_uta, d_j2, d_j3, d_j3v, d_normal):
        tmp = 1 / (2 * self.rho0 * self.c0)
        if (d_normal[3*d_idx+1] < -1e-14):
            d_v[d_idx] = d_vta[d_idx] + d_j2v[d_idx] * tmp
        elif (d_normal[3*d_idx+1] > 1e-14):
            d_v[d_idx] = d_vta[d_idx] - d_j3v[d_idx] * tmp

        if (d_normal[3*d_idx] < -1e-14):
            d_u[d_idx] = d_uta[d_idx] + d_j2[d_idx] * tmp
        elif (d_normal[3*d_idx] > 1e-14):
            d_u[d_idx] = d_uta[d_idx] - d_j3[d_idx] * tmp


class EvaluateSlipWallPressure(Equation):
    def initialize(self, d_p, d_j2v, d_idx, d_pta, d_j3v, d_y, d_normal, d_j2, d_j3):
        if (d_normal[3*d_idx+1] < -1e-14):
            d_p[d_idx] = d_pta[d_idx] + d_j2v[d_idx] / 2
        elif (d_normal[3*d_idx+1] > 1e-14):
            d_p[d_idx] = d_pta[d_idx] + d_j3v[d_idx] / 2

        if (d_normal[3*d_idx] < -1e-14):
            d_p[d_idx] = d_pta[d_idx] + d_j2[d_idx] / 2
        elif (d_normal[3*d_idx] > 1e-14):
            d_p[d_idx] = d_pta[d_idx] + d_j3[d_idx] / 2


class EvaluateCharacteristics(Equation):
    def __init__(self, dest, sources, rho0, c0):
        self.rho0 = rho0
        self.c0 = c0
        super(EvaluateCharacteristics, self).__init__(dest, sources)

    def initialize(self, d_idx, d_j1, d_j2, d_j3, d_u, d_p,
                   d_uta, d_pta, d_v, d_j2v, d_j3v, d_vta):
        co = self.c0
        rho0 = self.rho0
        d_j1[d_idx] = (d_p[d_idx] - d_pta[d_idx])
        d_j2[d_idx] = rho0 * co * (d_u[d_idx] - d_uta[d_idx]) +\
             (d_p[d_idx] - d_pta[d_idx])
        d_j3[d_idx] = -rho0 * co * (d_u[d_idx] - d_uta[d_idx]) +\
             (d_p[d_idx] - d_pta[d_idx])

        d_j2v[d_idx] = rho0 * co * (d_v[d_idx] - d_vta[d_idx]) +\
             (d_p[d_idx] - d_pta[d_idx])
        d_j3v[d_idx] = -rho0 * co * (d_v[d_idx] - d_vta[d_idx]) +\
             (d_p[d_idx] - d_pta[d_idx])


class ExtrapolateCharacteristics(Equation):
    def initialize(self, d_idx, d_j1, d_j2, d_j3, d_wij, d_j2v, d_j3v):
        d_j1[d_idx] = 0.0
        d_j2[d_idx] = 0.0
        d_j3[d_idx] = 0.0
        d_j2v[d_idx] = 0.0
        d_j3v[d_idx] = 0.0
        d_wij[d_idx] = 0.0

    def loop(self, d_idx, d_j1, d_j2, d_j3, s_j1, s_j2, s_j3, WIJ, s_idx, d_j2v, d_j3v,
             s_j2v, s_j3v, d_wij):
        d_wij[d_idx] += WIJ
        d_j1[d_idx] += s_j1[s_idx] * WIJ
        d_j2[d_idx] += s_j2[s_idx] * WIJ
        d_j3[d_idx] += s_j3[s_idx] * WIJ
        d_j2v[d_idx] += s_j2v[s_idx] * WIJ
        d_j3v[d_idx] += s_j3v[s_idx] * WIJ

    def post_loop(self, d_idx, d_j1, d_j2, d_j3, d_wij, d_j2v, d_j3v):
        if d_wij[d_idx] > 1e-14:
            d_j1[d_idx] /= d_wij[d_idx]
            d_j2[d_idx] /= d_wij[d_idx]
            d_j3[d_idx] /= d_wij[d_idx]
            d_j2v[d_idx] /= d_wij[d_idx]
            d_j3v[d_idx] /= d_wij[d_idx]


class RigidFluidForce(Equation):
    """Force on rigid body due to the interaction with fluid.
    The force equation is taken from SPH-DCDEM paper by Canelas

    nu: dynamics viscosity of the fluid

    """
    def __init__(self, dest, sources, nu=0):
        self.nu = nu
        super(RigidFluidForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_m, d_fx, d_fy,
             d_fz, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ, R2IJ, EPS, VIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        # pressure forces
        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = - d_m[d_idx] * s_m[s_idx] * pij

        d_fx[d_idx] += tmp * DWIJ[0]
        d_fy[d_idx] += tmp * DWIJ[1]
        d_fz[d_idx] += tmp * DWIJ[2]

        # viscous forces
        xdotdij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]
        tmp_1 = s_m[s_idx] * 4 * self.nu * xdotdij
        tmp_2 = (d_rho[d_idx] + s_rho[s_idx]) * (R2IJ + EPS)
        fac = tmp_1 / tmp_2

        d_fx[d_idx] += d_m[d_idx] * fac * VIJ[0]
        d_fy[d_idx] += d_m[d_idx] * fac * VIJ[1]
        d_fz[d_idx] += d_m[d_idx] * fac * VIJ[2]


class ParticlesFluidScheme(Scheme):
    def __init__(self, fluids, boundaries, rigid_bodies, dim, c0, nu, rho0, h, pb=0.0,
                 gx=0.0, gy=0.0, gz=0.0, alpha=0.0):
        self.c0 = c0
        self.nu = nu
        self.h = h
        self.rho0 = rho0
        self.pb = pb
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.dim = dim
        self.alpha = alpha
        self.fluids = fluids
        self.boundaries = boundaries
        self.rigid_bodies = rigid_bodies
        self.solver = None

        self.artificial_viscosity_with_boundary = False
        self.use_edac = False
        self.use_delta_sph = False

    def add_user_options(self, group):
        group.add_argument("--alpha", action="store", type=float, dest="alpha",
                           default=0.02,
                           help="Alpha for the artificial viscosity.")

        group.add_argument("--nu", action="store", type=float, dest="nu",
                           default=0.00,
                           help="Dynamics viscosity.")

        add_bool_argument(group, 'use-edac',
                          dest='use_edac',
                          default=False,
                          help='Use edac for pressure evolution')

        add_bool_argument(group, 'use-delta-sph',
                          dest='use_delta_sph',
                          default=False,
                          help='Use Delta SPH for density evolution')

        add_bool_argument(group, 'artificial-viscosity-with-boundary',
                          dest='artificial_viscosity_with_boundary',
                          default=False,
                          help='Use boundary in artificial viscosity computation')

    def consume_user_options(self, options):
        vars = [
            'alpha', 'artificial_viscosity_with_boundary', 'use_edac',
            'use_delta_sph', 'nu'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

    def attributes_changed(self):
        self.edac_alpha = self._get_edac_nu()
        self.edac_nu = self.edac_alpha * self.h * self.c0 / 8.

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.gtvf import GTVFIntegrator
        from pysph.solver.solver import Solver
        from rigid_body import GTVFRigidBody3DStep

        kernel = QuinticSpline(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = FluidStep
        cls = (integrator_cls
               if integrator_cls is not None else GTVFIntegrator)

        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        bodystep = GTVFRigidBody3DStep()
        integrator_cls = GTVFIntegrator

        for body in self.rigid_bodies:
            if body not in steppers:
                steppers[body] = bodystep

        integrator = cls(**steppers)

        self.solver = Solver(dim=self.dim, integrator=integrator,
                             kernel=kernel, **kw)
        # print("configure solver")

    def get_equations(self):
        from pysph.sph.wc.gtvf import (MomentumEquationViscosity)
        from pysph.sph.basic_equations import (IsothermalEOS)
        from pysph.sph.wc.edac import (SolidWallPressureBC,
                                       ClampWallPressure,
                                       SourceNumberDensity)

        from pysph.sph.wc.transport_velocity import (
            MomentumEquationArtificialViscosity
        )
        from pysph.sph.wc.transport_velocity import (SetWallVelocity)


        # all = self.fluids + self.boundaries
        all = self.fluids + self.boundaries + self.rigid_bodies
        # =========================#
        # stage 1 equations start
        # =========================#
        stage1 = []

        eqs = []
        for fluid in self.fluids:
            eqs.append(ContinuityEquation(dest=fluid,
                                          sources=all))
            if self.use_delta_sph == True:
                eqs.append(DeltaSPHCorrection(dest=fluid,
                                              sources=all,
                                              c0=self.c0))

            if self.use_edac == True:
                for fluid in self.fluids:
                    eqs.append(
                        EDACEquation(dest=fluid,
                                     sources=all,
                                     nu=self.edac_nu))

        stage1.append(Group(equations=eqs, real=False))

        # =========================#
        # stage 2 equations start
        # =========================#
        stage2 = []

        if self.use_edac == False:
            tmp = []
            for fluid in self.fluids:
                tmp.append(
                    StateEquation(dest=fluid,
                                  sources=None,
                                  rho0=self.rho0,
                                  p0=self.pb))

            stage2.append(Group(equations=tmp, real=False))

        tmp = []
        for fluid in self.fluids:
            tmp.append(
                EvaluateCharacteristics(dest=fluid,
                                sources=None,
                                rho0=self.rho0,
                                c0=self.c0))

        stage2.append(Group(equations=tmp, real=False))

        if len(self.boundaries) > 0:
            eqs = []
            for boundary in self.boundaries:
                # eqs.append(SetWallVelocity(dest=boundary, sources=self.fluids))
                eqs.append(ExtrapolateCharacteristics(dest=boundary, sources=self.fluids))
            stage2.append(Group(equations=eqs, real=False))

        if len(self.boundaries) > 0:
            eqs = []
            for boundary in self.boundaries:
                # eqs.append(
                #     SourceNumberDensity(dest=boundary, sources=self.fluids))
                # eqs.append(
                #     SolidWallPressureBC(dest=boundary, sources=self.fluids,
                #                         gx=self.gx, gy=self.gy, gz=self.gz))
                # eqs.append(
                #     ClampWallPressure(dest=boundary, sources=None))
                eqs.append(EvaluateSlipWallVelocity(dest=boundary, sources=None, rho0=self.rho0, c0=self.c0))
                eqs.append(EvaluateSlipWallPressure(dest=boundary, sources=None))


            stage2.append(Group(equations=eqs, real=False))

        if len(self.rigid_bodies) > 0:
            eqs = []
            for body in self.rigid_bodies:
                eqs.append(
                    SourceNumberDensity(dest=body, sources=self.fluids))
                eqs.append(
                    SolidWallPressureBC(dest=body, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    ClampWallPressure(dest=body, sources=None))

            stage2.append(Group(equations=eqs, real=False))

        eqs = []
        for fluid in self.fluids:
            if self.alpha > 0.:
                if self.artificial_viscosity_with_boundary == True:
                    eqs.append(
                        MomentumEquationArtificialViscosity(
                            dest=fluid, sources=self.fluids+self.boundaries, c0=self.c0,
                            alpha=self.alpha
                        )
                    )
                else:
                    eqs.append(
                        MomentumEquationArtificialViscosity(
                            dest=fluid, sources=self.fluids, c0=self.c0,
                            alpha=self.alpha
                        )
                    )

            if self.nu > 0.0:
                eqs.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=self.fluids+self.rigid_bodies,
                        nu=self.nu)
                )
                if len(self.boundaries) > 0:
                    eqs.append(
                        SolidWallNoSlipBC(
                            dest=fluid, sources=self.boundaries, nu=self.nu
                        )
                    )
            eqs.append(
                MomentumEquationPressureGradient(dest=fluid, sources=all,
                                                 gx=self.gx, gy=self.gy,
                                                 gz=self.gz), )

        stage2.append(Group(equations=eqs, real=True))

        # ========================
        # ========================
        # Forces on the rigid body
        # ========================
        # ========================
        if len(self.rigid_bodies) > 0:
            eqs = []
            for body in self.rigid_bodies:
                eqs.append(
                    RigidFluidForce(dest=body, sources=self.fluids, nu=self.nu))

            stage2.append(Group(equations=eqs, real=True))

            # computation of total force and torque at center of mass
            g6 = []
            for name in self.rigid_bodies:
                g6.append(SumUpExternalForces(dest=name,
                                              sources=None,
                                              gx=self.gx,
                                              gy=self.gy,
                                              gz=self.gz))

            stage2.append(Group(equations=g6, real=False))

        return MultiStageEquations([stage1, stage2])

    def _get_edac_nu(self):
        if self.alpha > 0:
            nu = self.alpha
            # print(self.alpha)
            # print("Using artificial viscosity for EDAC with nu = %s" % nu)
        else:
            nu = self.nu
            # print("Using real viscosity for EDAC with nu = %s" % self.nu)
        return nu
