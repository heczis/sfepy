r"""
Self-contact of an elastic body with a penalty function for enforcing the contact
constraints.

Find :math:`\ul{u}` such that:

.. math::
    \int_{\Omega} D_{ijkl}\ e_{ij}(\ul{v}) e_{kl}(\ul{u})
    + \int_{\Gamma_{c}} \varepsilon_N \langle g_N(\ul{u}) \rangle \ul{n} \ul{v}
    = 0
    \;, \quad \forall \ul{v} \;,

where :math:`\varepsilon_N \langle g_N(\ul{u}) \rangle` is the penalty
function, :math:`\varepsilon_N` is the normal penalty parameter, :math:`\langle
g_N(\ul{u}) \rangle` are the Macaulay's brackets of the gap function
:math:`g_N(\ul{u})` and

.. math::
    D_{ijkl} = \mu (\delta_{ik} \delta_{jl}+\delta_{il} \delta_{jk}) +
    \lambda \ \delta_{ij} \delta_{kl}
    \;.

Usage examples::

  ./simple.py cell_self_contact.py --save-regions-as-groups --save-ebc-nodes

Post-processing::

  ./postproc.py output/circle_in_square_w_cavity.*.vtk --wire -b \
  -d "u,plot_displacements,rel_scaling=1,color_kind='tensors',"\
  "color_name='cauchy_stress'" --only-names=u

Plot the nonlinear solver convergence::

  ./script/plot_logs.py log.txt
"""

import numpy as nm

from sfepy import data_dir
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.discrete.fem import Mesh

def stress_strain(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    from sfepy.base.base import Struct

    ev = pb.evaluate
    strain = ev('ev_cauchy_strain.2.Omega(u)', mode='el_avg')
    stress = ev('ev_cauchy_stress.2.Omega(solid.D, u)', mode='el_avg',
                copy_materials=False)

    out['cauchy_strain'] = Struct(name='output_data', mode='cell',
                                  data=strain, dofs=None)
    out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                  data=stress, dofs=None)

    return out

filename_mesh = data_dir + '/meshes/2d/special/circle_in_square_w_cavity.vtk'

options = {
    'ts' : 'ts',
    'nls' : 'newton',
    'ls' : 'ls',
    'output_dir' : 'output',
    'post_process_hook' : 'stress_strain'
}

fields = {
    'displacement': ('real', 2, 'Omega', 1),
}

materials = {
    'solid' : ({'D': stiffness_from_youngpoisson(
        2, young=1.0, poisson=0.3)},),
    'contact' : ({'.epss' : 1e+2},),
}

variables = {
    'u' : ('unknown field', 'displacement', 0),
    'v' : ('test field', 'displacement', 'u'),
}

def get_ebc(ts, coors, bc, problem, **kwargs):
    """
    Return the prescribed displacements
    """
    val = nm.empty_like(coors[:, 1])
    val[:] = -ts.time * 0.09
    return val

functions = {
    'get_ebc' : (get_ebc,),
}

regions = {
    'Omega' : 'all',
    'Bottom' : ('vertices in (y < %f)' % 1e-5, 'facet'),
    'Top' : ('vertices in (y > %f)' % (1.-1e-5), 'facet'),
    'Contact0' : ('(vertices of group 5) +v (vertices of group 6) '
                  '+v (vertices of group 10)',
                  'facet'),
    'Contact1' : ('(vertices of group 7) +v (vertices of group 8) '
                  '+v (vertices of group 11)',
                  'facet'),
    'Contact' : ('r.Contact0 +v r.Contact1', 'facet')
}

ebcs = {
    'fixb' : ('Bottom', {'u.all' : 0.0}),
    'fixt' : ('Top', {'u.0' : 0.0, 'u.1' : 'get_ebc'}),
}

integrals = {
    'i' : 20,
}

equations = {
    'elasticity' :
    """dw_lin_elastic.2.Omega(solid.D, v, u)
     + dw_contact.i.Contact(contact.epss, v, u)
     = 0""",
}

solvers = {
    'ls' : ('ls.scipy_direct', {}),
    'newton' : ('nls.newton', {
        'i_max' : 20,
        'eps_a' : 1e-6,
        'eps_r' : 1e-5,
        'macheps' : 1e-16,
        # Linear system error < (eps_a * lin_red).
        'lin_red' : 1e-2,
        'ls_red' : 0.1,
        'ls_red_warp' : 0.001,
        'ls_on' : 1.1,
        'ls_min' : 1e-5,
        'check' : 0,
        'delta' : 1e-8,
        'log' : {'text' : 'log.txt', 'plot' : None},
    }),
    'ts' : ('ts.simple', {
        't0' : 0.0,
        't1' : 1.0,
        'n_step' : 6,
    }),
}
