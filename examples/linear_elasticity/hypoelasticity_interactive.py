#!/usr/bin/env python
"""
Uniaxial tension of a block of hypoelastic material, i.e. the stress-strain
relationship is given as:

.. math::
    \dot\sigma = f(\dot\epsilon)


"""
from __future__ import print_function, absolute_import
import os
import sys

sys.path.append('.')

import numpy as np

from sfepy.base.base import IndexedStruct, Struct
from sfepy.discrete import (FieldVariable, Material, Integral, Equation,
                            Equations, Function, Problem)
from sfepy.discrete.conditions import Conditions, EssentialBC, InitialCondition
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.homogenization.utils import define_box_regions
from sfepy.mechanics.matcoefs import stiffness_from_lame_mixed
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.solvers.ls import ScipyDirect, ScipyIterative
from sfepy.solvers.nls import Newton
from sfepy.solvers.ts_solvers import SimpleTimeSteppingSolver
from sfepy.terms import Term

DIMENSION = 2
ORDER = 3
MAX_DISPLACEMENT = .2
MU = 1.0
LBD = 1.0
RAMP_TIME = 4
HOLD_TIME = 2

def get_displacement(ts, coors, bc=None, problem=None):
    """
    Define the time-dependent displacement
    """
    if ts.time <= RAMP_TIME:
        val = MAX_DISPLACEMENT * ts.time / RAMP_TIME * np.ones(coors.shape[0])
    elif ts.time <= RAMP_TIME + HOLD_TIME:
        val = MAX_DISPLACEMENT * np.ones(coors.shape[0])
    else:
        val = MAX_DISPLACEMENT * (2*RAMP_TIME + HOLD_TIME - ts.time) \
              / RAMP_TIME * np.ones(coors.shape[0])
    return val

stresses = [None,]
strains = [None,]

def get_modulus(ts, coors, problem, mode=None, **kwargs):
    """
    Returns the time-dependent modulus
    """
    if mode != 'qp': return

    if ts.step == 0:
        state = problem.create_state()
        problem.setup_ics()
        state.apply_ic()
        problem.equations.variables.set_data(state())
        m1 = Material('m1', D=stiffness_from_lame_mixed(DIMENSION, 1., 1.))
        stresses[0] = problem.evaluate(
            'ev_cauchy_stress.%d.Omega(m1.D, u)' % ORDER,
            mode='el_avg', m1=m1, copy_materials=False)

    strain = problem.evaluate(
        'ev_cauchy_strain.%d.Omega(u)' % (ORDER+1), mode='el_avg')
    if ts.step == 0:
        strains[0] = strain

    strain_rate = (strain-strains[0]) / ts.dt

    D_mat = stiffness_from_lame_mixed(DIMENSION, 1., np.exp(-ts.time/10.))
    d_D_mat = stiffness_from_lame_mixed(DIMENSION, 0., -.1*np.exp(-ts.time/10.))

    D = np.array([D_mat for ii in range(coors.shape[0])])
    d_D = np.array([d_D_mat for ii in range(coors.shape[0])])

    stress_hypo_rate = np.zeros_like(stresses[0])
    for ielm in range(strain_rate.shape[0]):
        for iqp in range(strain_rate.shape[1]):
            stress_hypo_rate[ielm, iqp] = np.dot(D_mat, strain_rate[ielm, iqp])
    stresses[0] += ts.dt * stress_hypo_rate

    return {'D' : D, 'd_D' : d_D}

def main():
    mesh = gen_block_mesh([1,]*DIMENSION, [9,]*DIMENSION, [.5,]*DIMENSION)
    domain = FEDomain('domain', mesh)

    ### geometry ###
    omega = domain.create_region('Omega', 'all')
    lbn, rtf = domain.get_mesh_bounding_box()
    box_regions = define_box_regions(DIMENSION, lbn, rtf)
    regions = dict([
        [r, domain.create_region(r, box_regions[r][0], box_regions[r][1])]
        for r in box_regions])

    ### fields and variables ###
    field = Field.from_args(
        'fu', np.float64, 'vector', omega, approx_order=ORDER-1)
    field_p = Field.from_args(
        'fp', np.float64, 'scalar', omega, approx_order=ORDER-2)

    u = FieldVariable('u', 'unknown', field, history=1)
    v = FieldVariable('v', 'test', field, primary_var_name='u')
    p = FieldVariable('p', 'unknown', field_p, history=1)
    q = FieldVariable('q', 'test', field_p, primary_var_name='p')

    ### material ###
    m = Material('m', D=stiffness_from_lame_mixed(DIMENSION, 1.0, 1.0))
    m2_fun = Function('m2_fun', get_modulus)
    m2 = Material('m2', function=m2_fun)

    ### equations ###
    integral = Integral('i', order=ORDER)

    term_elast = Term.new('dw_lin_elastic(m2.D, v, du/dt)',
                          integral, omega, m2=m2, v=v, u=u)
    term_d_elast = Term.new('dw_lin_elastic(m2.d_D, v, u)',
                            integral, omega, m2=m2, v=v, u=u)
    term_vol_1 = Term.new('dw_stokes(v, dp/dt)',
                          integral, omega, v=v, p=p)
    term_hypo = Term.new('dw_lin_elastic(m.D, v, du/dt)',
                         integral, omega, m=m, v=v, u=u)
    eq_balance = Equation('balance', term_elast + term_d_elast + term_hypo
                          - term_vol_1)

    term_vol_2 = Term.new('dw_stokes(u, q)',
                          integral, omega, u=u, q=q)
    eq_constraint = Equation('constraint', term_vol_2)
    eqs = Equations([eq_balance, eq_constraint])

    ### boundary and initial conditions ###
    bc_fun = Function('bc_fun', get_displacement)

    ebc_fix_x = EssentialBC('ebc_fix_x', regions['Left'], {'u.0' : 0.0})
    ebc_fix_y = EssentialBC('ebc_fix_y', regions['Left'], {'u.1' : 0.0})
    ebc_move = EssentialBC('ebc_move', regions['Right'],
                           {'u.1' : bc_fun, 'u.0' : 0.0})
    ebcs = Conditions([ebc_fix_x, ebc_fix_y, ebc_move])

    ic_u = InitialCondition('ic', omega, {'u.all' : 0.0, 'p.0' : 0.0})
    ics = Conditions([ic_u,])

    ### solution ###
    ls = ScipyDirect({})
    nls_status = IndexedStruct()
    nls = Newton({'is_linear' : False}, lin_solver=ls, status=nls_status)

    pb = Problem('hypoelasticity', equations=eqs, nls=nls, ls=ls)
    pb.set_bcs(ebcs=ebcs)
    pb.set_ics(ics)

    tss = SimpleTimeSteppingSolver({'t0' : 0.0, 't1' : 10., 'n_step' : 21},
                                   problem=pb)
    tss.init_time()

    for step, time, state in tss():
        out = state.create_output_dict()
        out['stress_hypo'] = Struct(name='output_data', mode='cell',
                                    data=stresses[0], dofs=None)
        p_vals = state.get_parts()['p']
        p_data = np.zeros_like(stresses[0])
        for iel in range(p_data.shape[0]):
            p_data[iel,0] = np.array([[1, 1, 0]]).T * p_vals[iel]
        out['stress_total'] = Struct(name='output_data', mode='cell',
                                     data=stresses[0]-p_data, dofs=None)

        strain = pb.evaluate(
            'ev_cauchy_strain.%d.Omega(u)' % ORDER,
            mode='el_avg')
        out['strain'] = Struct(name='output_data', mode='cell',
                               data=strain, dofs=None)
        pb.save_state('hypoelastic.%d.vtk' % step, out=out)

if __name__ == '__main__':
    main()
