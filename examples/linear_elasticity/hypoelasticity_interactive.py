"""
Compute diffusion-limited oxidation and ageing in rigid medium.
"""
from __future__ import print_function, absolute_import
import os
import sys

DATA_DIR = '/home/jan/Documents/PYTHON/sfepy3'
sys.path.append(DATA_DIR)

import numpy as np

from sfepy.base.base import IndexedStruct, Struct
from sfepy.discrete import (FieldVariable, Material, Integral, Equation,
                            Equations, Function, Problem)
from sfepy.discrete.conditions import Conditions, EssentialBC, InitialCondition
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.homogenization.utils import define_box_regions
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.mesh.mesh_generators import gen_block_mesh
from sfepy.solvers.ls import ScipyDirect, ScipyIterative
from sfepy.solvers.nls import Newton
from sfepy.solvers.ts_solvers import SimpleTimeSteppingSolver
from sfepy.terms import Term

DIMENSION = 2
ORDER = 2
MAX_DISPLACEMENT = .1
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

def get_modulus(ts, coors, problem, mode=None, **kwargs):
    """
    Returns the time-dependent modulus
    """
    if mode == 'qp':
        mu = MU * (1-np.exp(-ts.time/10.)) \
             * np.ones((coors.shape[0]*coors.shape[1], 1, 1))
        lbd = LBD * np.ones((coors.shape[0]*coors.shape[1], 1, 1))
        return {'mu' : mu, 'lbd' : lbd}

def main():
    mesh = gen_block_mesh([1,]*DIMENSION, [4,]*DIMENSION, [.5,]*DIMENSION)
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

    u = FieldVariable('u', 'unknown', field, history=1)
    v = FieldVariable('v', 'test', field, primary_var_name='u')

    ### material ###
    m2_fun = Function('m2_fun', get_modulus)
    m2 = Material('m2', function=m2_fun)

    ### equations ###
    integral = Integral('i', order=ORDER)

    term_hypo = Term.new('dw_lin_elastic_iso(m2.mu, m2.lbd, v, du/dt)',
                         integral, omega, m2=m2, v=v, u=u)
    eq_balance = Equation('balance', term_hypo)
    eqs = Equations([eq_balance,])

    ### boundary and initial conditions ###
    bc_fun = Function('bc_fun', get_displacement)

    ebc_fix_x = EssentialBC('ebc_fix_x', regions['Left'], {'u.0' : 0.0})
    ebc_fix_y = EssentialBC('ebc_fix_y', regions['Bottom'], {'u.1' : 0.0})
    ebc_move = EssentialBC('ebc_move', regions['Right'],
                           {'u.0' : bc_fun})
    ebcs = Conditions([ebc_fix_x, ebc_fix_y, ebc_move])

    ic_u = InitialCondition('ic', omega, {'u.all' : 0.0})
    ics = Conditions([ic_u,])

    ### solution ###
    ls = ScipyDirect({})
    nls_status = IndexedStruct()
    nls = Newton({'is_linear' : False}, lin_solver=ls, status=nls_status)

    pb = Problem('hypoelasticity', equations=eqs, nls=nls, ls=ls)
    pb.set_bcs(ebcs=ebcs)
    pb.set_ics(ics)

    tss = SimpleTimeSteppingSolver({'t0' : 0.0, 't1' : 10, 'n_step' : 11},
                                   problem=pb)
    tss.init_time()

    for step, time, state in tss(save_results=False):
        strain = pb.evaluate(
            'ev_cauchy_strain.%d.Omega(u)' % ORDER, mode='el_avg')

        out = state.create_output_dict()
        out['strain'] = Struct(name='output_data', mode='cell',
                               data=strain, dofs=None)
        pb.save_state('hypoelastic.%d.vtk' % step, out=out)

if __name__ == '__main__':
    main()
