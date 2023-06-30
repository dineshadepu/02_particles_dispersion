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


def get_particle_array_fluid(x, y, z, m, h, rho, name):
    pa = get_particle_array(name=name,
                            x=x,
                            y=y,
                            z=z,
                            h=h,
                            rho=rho,
                            m=m)
    add_properties(pa, 'arho', 'aconcentration', 'concentration', 'diff_coeff',
                   'is_static')
    add_properties(pa, 'm_frac')
    pa.m_frac[:] = 1.
    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    pa.add_output_arrays(['concentration', 'is_static', 'diff_coeff'])

    return pa


def get_particle_array_boundary(constants=None, **props):
    solids_props = [
        'wij', 'm_frac'
    ]

    # set wdeltap to -1. Which defaults to no self correction
    consts = {

    }
    if constants:
        consts.update(constants)

    pa = get_particle_array(constants=consts, additional_props=solids_props,
                            **props)
    pa.m_frac[:] = 1.

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'm', 'h', 'pid', 'gid', 'tag', 'p'
    ])

    return pa


class ContinuityEquation(Equation):
    r"""Density rate:

    :math:`\frac{d\rho_a}{dt} = \sum_b m_b \boldsymbol{v}_{ab}\cdot
    \nabla_a W_{ab}`

    """
    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_arho, s_idx, s_m, DWIJ, VIJ):
        vijdotdwij = DWIJ[0]*VIJ[0] + DWIJ[1]*VIJ[1] + DWIJ[2]*VIJ[2]
        d_arho[d_idx] += s_m[s_idx]*vijdotdwij


class StateEquation(Equation):
    def __init__(self, dest, sources, p0, rho0, b=1.0):
        self.b = b
        self.p0 = p0
        self.rho0 = rho0
        super(StateEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_rho):
        d_p[d_idx] = self.p0 * (d_rho[d_idx] / self.rho0 - self.b)


class DiffusionEquation(Equation):
    """
    Rate of change of concentration of a particle from Fick's law.
    Equation 20 (without advection) of reference [1].

    [1]
    """
    def initialize(self, d_idx, d_aconcentration):
        d_aconcentration[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, d_m,
             s_m,
             d_diff_coeff,
             s_diff_coeff,
             d_concentration,
             s_concentration,
             d_aconcentration,
             R2IJ, EPS, DWIJ, VIJ, XIJ):

        # averaged shear viscosity Eq. (6)
        tmp_i = s_m[s_idx] * d_rho[d_idx] * d_diff_coeff[d_idx]
        tmp_j = d_m[d_idx] * s_rho[s_idx] * s_diff_coeff[s_idx]

        rho_i = d_rho[d_idx]
        rho_j = s_rho[s_idx]
        rho_ij = (rho_i * rho_j)

        conc_ij = d_concentration[d_idx] - s_concentration[s_idx]

        # scalar part of the kernel gradient
        Fij = DWIJ[0]*XIJ[0] + DWIJ[1]*XIJ[1] + DWIJ[2]*XIJ[2]

        # accelerations 3rd term in Eq. (8)
        tmp = (tmp_i + tmp_j) * conc_ij * Fij/(R2IJ + EPS)

        d_aconcentration[d_idx] += 1. / rho_ij * tmp


class MomentumEquationPressureGradient(Equation):
    def __init__(self, dest, sources, gx=0.0, gy=0.0, gz=0.0):
        self.gx = gx
        self.gy = gy
        self.gz = gz
        super(MomentumEquationPressureGradient, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = self.gx
        d_av[d_idx] = self.gy
        d_aw[d_idx] = self.gz

    def loop(self, d_rho, s_rho, d_idx, s_idx, d_p, s_p, s_m, d_au,
             d_av, d_aw, DWIJ, XIJ, RIJ, SPH_KERNEL, HIJ):
        rhoi2 = d_rho[d_idx] * d_rho[d_idx]
        rhoj2 = s_rho[s_idx] * s_rho[s_idx]

        pij = d_p[d_idx]/rhoi2 + s_p[s_idx]/rhoj2

        tmp = -s_m[s_idx] * pij

        d_au[d_idx] += tmp * DWIJ[0]
        d_av[d_idx] += tmp * DWIJ[1]
        d_aw[d_idx] += tmp * DWIJ[2]


class FluidStep(IntegratorStep):
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_is_static, dt):
        dtb2 = 0.5 * dt
        if d_is_static[d_idx] == 0.:
            d_u[d_idx] += dtb2 * d_au[d_idx]
            d_v[d_idx] += dtb2 * d_av[d_idx]
            d_w[d_idx] += dtb2 * d_aw[d_idx]

    def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_rho,
               d_arho, d_concentration, d_aconcentration, dt):
        d_rho[d_idx] += dt * d_arho[d_idx]
        d_concentration[d_idx] += dt * d_aconcentration[d_idx]

        d_x[d_idx] += dt * d_u[d_idx]
        d_y[d_idx] += dt * d_v[d_idx]
        d_z[d_idx] += dt * d_w[d_idx]

    def stage3(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_is_static, dt):
        dtb2 = 0.5 * dt
        if d_is_static[d_idx] == 0.:
            d_u[d_idx] += dtb2 * d_au[d_idx]
            d_v[d_idx] += dtb2 * d_av[d_idx]
            d_w[d_idx] += dtb2 * d_aw[d_idx]


class FluidsScheme(Scheme):
    def __init__(self, fluids, boundaries, dim, c0, nu, rho0, pb=0.0,
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

        integrator = cls(**steppers)

        from pysph.solver.solver import Solver
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


        all = self.fluids + self.boundaries
        # =========================#
        # stage 1 equations start
        # =========================#
        stage1 = []

        eqs = []
        for fluid in self.fluids:
            eqs.append(ContinuityEquation(dest=fluid,
                                          sources=all), )

            eqs.append(DiffusionEquation(dest=fluid,
                                         sources=self.fluids), )
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
                eqs.append(
                    SourceNumberDensity(dest=boundary, sources=self.fluids))
                eqs.append(
                    SolidWallPressureBC(dest=boundary, sources=self.fluids,
                                        gx=self.gx, gy=self.gy, gz=self.gz))
                eqs.append(
                    ClampWallPressure(dest=boundary, sources=None))

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
            eqs.append(
                MomentumEquationPressureGradient(dest=fluid, sources=all,
                                                 gx=self.gx, gy=self.gy,
                                                 gz=self.gz), )

        stage2.append(Group(equations=eqs, real=True))

        return MultiStageEquations([stage1, stage2])
