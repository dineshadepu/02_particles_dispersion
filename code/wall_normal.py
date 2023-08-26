from math import sqrt
from re import X
from tkinter import Y
from compyle.api import declare
import numpy

from pysph.base.utils import get_particle_array
from pysph.sph.equation import Equation, Group


def find_nearest_point(solid, xn, yn, xb, yb):

    from pysph.tools.sph_evaluator import SPHEvaluator
    m = solid.m[0]
    h = 3*solid.h[0]
    boundary = get_particle_array(name='boundary', x=xb, y=yb, xn=xn, yn=yn, m=m, h=h)

    solid_copy = solid.extract_particles(numpy.arange(len(solid.x)))
    solid_copy.h[:] = h
    # pa.add_property('normal', stride=3)
    # pa.add_property('normal_tmp', stride=3)

    # name = pa.name
    props = ['xn', 'yn', 'zn', 'hard', 'neartag', 'nearest', 'proj']
    for pa in [solid_copy, boundary]:
        for prop in props:
            pa.add_property(prop)

    seval = SPHEvaluator(
        arrays=[boundary, solid_copy], equations=[
            Group(equations=[
                FindNearestNode(dest='solid', sources=['boundary'])
            ]),
        ],
        dim=2
    )
    seval.evaluate()

    solid.normal[0::3] = solid_copy.xn.copy()
    solid.normal[1::3] = solid_copy.yn.copy()
    return abs(solid_copy.proj.copy())


class FindNearestNode(Equation):
    """Find nearest boundary node

    Parameters:

    dest: pysph.base.utils.ParticleArray
        Free or boundary particle

    sources: list of pysph.base.utils.ParticleArray
        Boundary nodes array
    """
    def __init__(self, dest, sources, fac=1.0):
        self.fac = fac
        super(FindNearestNode, self).__init__(dest, sources)

    def initialize(self, d_idx, d_nearest, d_xn, d_yn, d_zn):
        d_nearest[d_idx] = 10000.0
        d_xn[d_idx] = 0.0
        d_yn[d_idx] = 0.0
        d_zn[d_idx] = 0.0

    def loop(self, d_idx, s_idx, RIJ, d_nearest,
             d_xn, d_yn, d_zn, s_xn, s_yn, s_zn, XIJ, d_proj):
        if (RIJ < d_nearest[d_idx]):
            d_nearest[d_idx] = RIJ
            d_xn[d_idx] = s_xn[s_idx]
            d_yn[d_idx] = s_yn[s_idx]
            d_zn[d_idx] = s_zn[s_idx]
            d_proj[d_idx] = XIJ[0] * d_xn[d_idx] + XIJ[1] * d_yn[d_idx]


class ComputeNormals(Equation):
    """Compute normals using a simple approach

    .. math::

       -\frac{m_j}{\rho_j} DW_{ij}

    First compute the normals, then average them and finally normalize them.

    """

    def initialize(self, d_idx, d_normal_tmp, d_normal):
        idx = declare('int')
        idx = 3*d_idx
        d_normal_tmp[idx] = 0.0
        d_normal_tmp[idx + 1] = 0.0
        d_normal_tmp[idx + 2] = 0.0
        d_normal[idx] = 0.0
        d_normal[idx + 1] = 0.0
        d_normal[idx + 2] = 0.0

    def loop(self, d_idx, d_normal_tmp, s_idx, s_m, s_rho, DWIJ):
        idx = declare('int')
        idx = 3*d_idx
        fac = -s_m[s_idx]/s_rho[s_idx]
        d_normal_tmp[idx] += fac*DWIJ[0]
        d_normal_tmp[idx + 1] += fac*DWIJ[1]
        d_normal_tmp[idx + 2] += fac*DWIJ[2]

    def post_loop(self, d_idx, d_normal_tmp, d_h):
        idx = declare('int')
        idx = 3*d_idx
        mag = sqrt(d_normal_tmp[idx]**2 + d_normal_tmp[idx + 1]**2 +
                   d_normal_tmp[idx + 2]**2)
        if mag > 0.05/d_h[d_idx]:
            d_normal_tmp[idx] /= mag
            d_normal_tmp[idx + 1] /= mag
            d_normal_tmp[idx + 2] /= mag
        else:
            d_normal_tmp[idx] = 0.0
            d_normal_tmp[idx + 1] = 0.0
            d_normal_tmp[idx + 2] = 0.0


class SmoothNormals(Equation):
    def loop(self, d_idx, d_normal, s_normal_tmp, s_idx, s_m, s_rho, WIJ):
        idx = declare('int')
        idx = 3*d_idx
        fac = s_m[s_idx]/s_rho[s_idx]*WIJ
        d_normal[idx] += fac*s_normal_tmp[3*s_idx]
        d_normal[idx + 1] += fac*s_normal_tmp[3*s_idx + 1]
        d_normal[idx + 2] += fac*s_normal_tmp[3*s_idx + 2]

    def post_loop(self, d_idx, d_normal, d_h):
        idx = declare('int')
        idx = 3*d_idx
        mag = sqrt(d_normal[idx]**2 + d_normal[idx + 1]**2 +
                   d_normal[idx + 2]**2)
        if mag > 1e-6:
            d_normal[idx] /= mag
            d_normal[idx + 1] /= mag
            d_normal[idx + 2] /= mag
        else:
            d_normal[idx] = 0.0
            d_normal[idx + 1] = 0.0
            d_normal[idx + 2] = 0.0


class InvertNormal(Equation):
    def __init__(self, dest, sources, x0=0.5123, y0=0.5123):
        self.x0 = x0
        self.y0= y0
        super(InvertNormal, self).__init__(dest, sources)

    def initialize(self, d_idx, d_normal_tmp, d_x, d_y, d_z):
        dx = d_x[d_idx] - self.x0
        dy = d_y[d_idx] - self.y0

        dot = dx * d_normal_tmp[3*d_idx] + dy*d_normal_tmp[3*d_idx+1]
        # print(dot, dx, dy, d_normal[3*d_idx], d_normal[3*d_idx+1])

        if (dot > 1e-14):
            # print(dot)
            d_normal_tmp[3*d_idx] = -d_normal_tmp[3*d_idx]
            d_normal_tmp[3*d_idx+1] = -d_normal_tmp[3*d_idx+1]


class SetWallVelocityNew(Equation):
    r"""Modified SetWall velocity which sets a suitable normal velocity.

    This requires that the destination array has a 3-strided "normal"
    property.
    """
    def initialize(self, d_idx, d_uf, d_vf, d_wf, d_wij):
        d_uf[d_idx] = 0.0
        d_vf[d_idx] = 0.0
        d_wf[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_uf, d_vf, d_wf,
             s_u, s_v, s_w, d_wij, d_wij2, XIJ, RIJ, HIJ, SPH_KERNEL):
        wij = SPH_KERNEL.kernel(XIJ, RIJ, HIJ)
        wij2 = SPH_KERNEL.kernel(XIJ, RIJ, 0.5*HIJ)

        if d_wij2[d_idx] > 1e-14:
            d_uf[d_idx] += s_u[s_idx] * wij2
            d_vf[d_idx] += s_v[s_idx] * wij2
            d_wf[d_idx] += s_w[s_idx] * wij2
        else:
            d_uf[d_idx] += s_u[s_idx] * wij
            d_vf[d_idx] += s_v[s_idx] * wij
            d_wf[d_idx] += s_w[s_idx] * wij

    def post_loop(self, d_uf, d_vf, d_wf, d_wij, d_wij2, d_idx,
                  d_ug, d_vg, d_wg, d_u, d_v, d_w, d_normal):
        idx = declare('int')
        idx = 3*d_idx
        # calculation is done only for the relevant boundary particles.
        # d_wij (and d_uf) is 0 for particles sufficiently away from the
        # solid-fluid interface
        if d_wij2[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij2[d_idx]
            d_vf[d_idx] /= d_wij2[d_idx]
            d_wf[d_idx] /= d_wij2[d_idx]
        elif d_wij[d_idx] > 1e-12:
            d_uf[d_idx] /= d_wij[d_idx]
            d_vf[d_idx] /= d_wij[d_idx]
            d_wf[d_idx] /= d_wij[d_idx]

        # Dummy velocities at the ghost points using Eq. (23),
        # d_u, d_v, d_w are the prescribed wall velocities.
        d_ug[d_idx] = 2*d_u[d_idx] - d_uf[d_idx]
        d_vg[d_idx] = 2*d_v[d_idx] - d_vf[d_idx]
        d_wg[d_idx] = 2*d_w[d_idx] - d_wf[d_idx]

        vn = (d_ug[d_idx]*d_normal[idx] + d_vg[d_idx]*d_normal[idx+1]
              + d_wg[d_idx]*d_normal[idx+2])
        if vn < 0:
            d_ug[d_idx] -= vn*d_normal[idx]
            d_vg[d_idx] -= vn*d_normal[idx+1]
            d_wg[d_idx] -= vn*d_normal[idx+2]


def get_normals(pa, dim, domain, x0=0.5, y0=0.5):
    from pysph.tools.sph_evaluator import SPHEvaluator

    pa.add_property('normal', stride=3)
    pa.add_property('normal_tmp', stride=3)

    name = pa.name

    props = ['m', 'rho', 'h']
    for p in props:
        x = pa.get(p)
        if numpy.all(x < 1e-12):
            msg = f'WARNING: cannot compute normals "{p}" is zero'
            print(msg)

    seval = SPHEvaluator(
        arrays=[pa], equations=[
            Group(equations=[
                ComputeNormals(dest=name, sources=[name])
            ]),
            Group(equations=[
                InvertNormal(dest=name, sources=None, x0=x0, y0=y0)
            ]),
            Group(equations=[
                SmoothNormals(dest=name, sources=[name])
            ]),
        ],
        dim=dim, domain_manager=domain
    )
    seval.evaluate()
