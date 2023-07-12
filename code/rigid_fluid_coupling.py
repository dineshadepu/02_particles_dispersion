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
                    SolidWallNoSlipBC
                    )
from rigid_body import (SumUpExternalForces)


def add_rigid_fluid_properties_to_rigid_body(pa):
    add_properties(pa, 'arho')
    add_properties(pa, 'm_fluid')
    add_properties(pa, 'm_frac')
    add_properties(pa, 'wij')
    add_properties(pa, 'ug', 'vf', 'uf', 'wf', 'vg', 'wg')


class RigidFluidForce(Equation):
    def __init__(self, dest, sources):
        super(RigidFluidForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_m, d_fx, d_fy,
             d_fz, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = - d_m[d_idx] * s_m[s_idx] * pij

        d_fx[d_idx] += tmp * DWIJ[0]
        d_fy[d_idx] += tmp * DWIJ[1]
        d_fz[d_idx] += tmp * DWIJ[2]


class ParticlesFluidScheme(Scheme):
    def __init__(self, fluids, boundaries, rigid_bodies, dim, c0, nu, rho0, pb=0.0,
                 gx=0.0, gy=0.0, gz=0.0, alpha=0.0):
        self.c0 = c0
        self.nu = nu
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

        self.attributes_changed()

    def add_user_options(self, group):
        group.add_argument("--alpha", action="store", type=float, dest="alpha",
                           default=None,
                           help="Alpha for the artificial viscosity.")

    def consume_user_options(self, options):
        vars = [
            'alpha'
        ]
        data = dict((var, self._smart_getattr(options, var)) for var in vars)
        self.configure(**data)

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
                                          sources=all), )
        stage1.append(Group(equations=eqs, real=False))

        # =========================#
        # stage 2 equations start
        # =========================#
        stage2 = []

        tmp = []
        for fluid in self.fluids:
            tmp.append(
                StateEquation(dest=fluid,
                              sources=None,
                              rho0=self.rho0,
                              p0=self.pb))

        stage2.append(Group(equations=tmp, real=False))

        if len(self.boundaries) > 0:
            eqs = []
            for boundary in self.boundaries:
                eqs.append(SetWallVelocity(dest=boundary, sources=self.fluids))
            stage2.append(Group(equations=eqs, real=False))

        if len(self.boundaries) > 0:
            eqs = []
            for boundary in self.boundaries:
                eqs.append(
                    SourceNumberDensity(dest=boundary, sources=self.fluids))
                eqs.append(
                    SolidWallPressureBC(dest=boundary, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    ClampWallPressure(dest=boundary, sources=None))

            stage2.append(Group(equations=eqs, real=False))

        if len(self.rigid_bodies) > 0:
            eqs = []
            for body in self.rigid_bodies:
                eqs.append(SetWallVelocity(dest=body, sources=self.fluids))
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
            # FIXME: Change alpha to variable
            if self.alpha > 0.:
                eqs.append(
                    MomentumEquationArtificialViscosity(
                        dest=fluid, sources=self.fluids, c0=self.c0,
                        alpha=self.alpha
                    )
                )
            if self.nu > 0.0:
                eqs.append(
                    MomentumEquationViscosity(
                        dest=fluid, sources=self.fluids, nu=self.nu
                    )
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
                    RigidFluidForce(dest=body, sources=self.fluids))

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
