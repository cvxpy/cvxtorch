from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import DivExpression, MulExpression, multiply
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.cumsum import cumsum
from cvxpy.atoms.affine.diag import diag_mat, diag_vec
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.atoms.affine.imag import imag
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.kron import kron
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.real import real
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.atoms.affine.vstack import Vstack
from cvxpy.atoms.cummax import cummax
from cvxpy.atoms.dist_ratio import dist_ratio
from cvxpy.atoms.dotsort import dotsort
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.ceil import ceil
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.elementwise.logistic import logistic
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.power import power
from cvxpy.atoms.elementwise.xexp import xexp
from cvxpy.atoms.eye_minus_inv import eye_minus_inv
from cvxpy.atoms.geo_mean import geo_mean
from cvxpy.atoms.gmatmul import gmatmul
from cvxpy.atoms.length import length
from cvxpy.atoms.log_det import log_det
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.matrix_frac import MatrixFrac
from cvxpy.atoms.max import max
from cvxpy.atoms.min import min
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.one_minus_pos import one_minus_pos
from cvxpy.atoms.pf_eigenvalue import pf_eigenvalue
from cvxpy.atoms.pnorm import Pnorm
from cvxpy.atoms.prod import Prod
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.atoms.sigma_max import sigma_max
from cvxpy.atoms.sign import sign
from cvxpy.atoms.tr_inv import tr_inv

import cvxtorch.torch_numerics.affine.add_expr as add_expr_tch
import cvxtorch.torch_numerics.affine.conj as conj_tch
import cvxtorch.torch_numerics.affine.cumsum as cumsum_tch
import cvxtorch.torch_numerics.affine.diag_mat as diag_mat_tch
import cvxtorch.torch_numerics.affine.diag_vec as diag_vec_tch
import cvxtorch.torch_numerics.affine.div_expression as div_expression_tch
import cvxtorch.torch_numerics.affine.hstack as hstack_tch
import cvxtorch.torch_numerics.affine.imag as imag_tch
import cvxtorch.torch_numerics.affine.index as index_tch
import cvxtorch.torch_numerics.affine.kron as kron_tch
import cvxtorch.torch_numerics.affine.mul_expression as mul_expression_tch
import cvxtorch.torch_numerics.affine.multiply as multiply_tch
import cvxtorch.torch_numerics.affine.promote as promote_tch
import cvxtorch.torch_numerics.affine.real as real_tch
import cvxtorch.torch_numerics.affine.reshape as reshape_tch
import cvxtorch.torch_numerics.affine.sum as sum_tch
import cvxtorch.torch_numerics.affine.trace as trace_tch
import cvxtorch.torch_numerics.affine.transpose as transpose_tch
import cvxtorch.torch_numerics.affine.unary_operators as unary_operators_tch
import cvxtorch.torch_numerics.affine.upper_tri as upper_tri_tch
import cvxtorch.torch_numerics.affine.vstack as vstack_tch
import cvxtorch.torch_numerics.cummax as cummax_tch
import cvxtorch.torch_numerics.dist_ratio as dist_ratio_tch
import cvxtorch.torch_numerics.dotsort as dotsort_tch
import cvxtorch.torch_numerics.elementwise.abs as abs_tch
import cvxtorch.torch_numerics.elementwise.ceil as ceil_tch
import cvxtorch.torch_numerics.elementwise.exp as exp_tch
import cvxtorch.torch_numerics.elementwise.log as log_tch
import cvxtorch.torch_numerics.elementwise.logistic as logistic_tch
import cvxtorch.torch_numerics.elementwise.maximum as maximum_tch
import cvxtorch.torch_numerics.elementwise.minimum as minimum_tch
import cvxtorch.torch_numerics.elementwise.power as power_tch
import cvxtorch.torch_numerics.elementwise.xexp as xexp_tch
import cvxtorch.torch_numerics.eye_minus_inv as eye_minus_inv_tch
import cvxtorch.torch_numerics.geo_mean as geo_mean_tch
import cvxtorch.torch_numerics.gmatmul as gmatmul_tch
import cvxtorch.torch_numerics.length as length_tch
import cvxtorch.torch_numerics.log_det as log_det_tch
import cvxtorch.torch_numerics.log_sum_exp as log_sum_exp_tch
import cvxtorch.torch_numerics.matrix_frac as matrix_frac_tch
import cvxtorch.torch_numerics.max as max_tch
import cvxtorch.torch_numerics.min as min_tch
import cvxtorch.torch_numerics.norm1 as norm1_tch
import cvxtorch.torch_numerics.norm_inf as norm_inf_tch
import cvxtorch.torch_numerics.norm_nuc as norm_nuc_tch
import cvxtorch.torch_numerics.one_minus_pos as one_minus_pos_tch
import cvxtorch.torch_numerics.pf_eigenvalue as pf_eigenvalue_tch
import cvxtorch.torch_numerics.pnorm as pnorm_tch
import cvxtorch.torch_numerics.prod as prod_tch
import cvxtorch.torch_numerics.quad_form as quad_form_tch
import cvxtorch.torch_numerics.quad_over_lin as quad_over_lin_tch
import cvxtorch.torch_numerics.sigma_max as sigma_max_tch
import cvxtorch.torch_numerics.sign as sign_tch
import cvxtorch.torch_numerics.tr_inv as tr_inv_tch

EXPR2TORCH = {
    cummax: cummax_tch.torch_numeric,
    dist_ratio: dist_ratio_tch.torch_numeric,
    dotsort: dotsort_tch.torch_numeric,
    eye_minus_inv: eye_minus_inv_tch.torch_numeric,
    geo_mean: geo_mean_tch.torch_numeric,
    gmatmul: gmatmul_tch.torch_numeric,
    length: length_tch.torch_numeric,
    log_det: log_det_tch.torch_numeric,
    log_sum_exp: log_sum_exp_tch.torch_numeric,
    MatrixFrac: matrix_frac_tch.torch_numeric,
    max: max_tch.torch_numeric,
    min: min_tch.torch_numeric,
    norm_inf: norm_inf_tch.torch_numeric,
    normNuc: norm_nuc_tch.torch_numeric,
    norm1: norm1_tch.torch_numeric,
    one_minus_pos: one_minus_pos_tch.torch_numeric,
    pf_eigenvalue: pf_eigenvalue_tch.torch_numeric,
    Pnorm: pnorm_tch.torch_numeric,
    Prod: prod_tch.torch_numeric,
    QuadForm: quad_form_tch.torch_numeric,
    quad_over_lin: quad_over_lin_tch.torch_numeric,
    sigma_max: sigma_max_tch.torch_numeric,
    sign: sign_tch.torch_numeric,
    tr_inv: tr_inv_tch.torch_numeric,
    AddExpression: add_expr_tch.torch_numeric,
    MulExpression: mul_expression_tch.torch_numeric,
    multiply: multiply_tch.torch_numeric,
    DivExpression: div_expression_tch.torch_numeric,
    conj: conj_tch.torch_numeric,
    cumsum: cumsum_tch.torch_numeric,
    diag_vec: diag_vec_tch.torch_numeric,
    diag_mat: diag_mat_tch.torch_numeric,
    Hstack: hstack_tch.torch_numeric,
    imag: imag_tch.torch_numeric,
    index: index_tch.torch_numeric,
    kron: kron_tch.torch_numeric,
    NegExpression: unary_operators_tch.torch_numeric,
    Promote: promote_tch.torch_numeric,
    real: real_tch.torch_numeric,
    reshape: reshape_tch.torch_numeric,
    Sum: sum_tch.torch_numeric,
    trace: trace_tch.torch_numeric,
    transpose: transpose_tch.torch_numeric,
    upper_tri: upper_tri_tch.torch_numeric,
    Vstack: vstack_tch.torch_numeric,
    abs: abs_tch.torch_numeric,
    ceil: ceil_tch.torch_numeric,
    exp: exp_tch.torch_numeric,
    log: log_tch.torch_numeric,
    logistic: logistic_tch.torch_numeric,
    maximum: maximum_tch.torch_numeric,
    minimum: minimum_tch.torch_numeric,
    power: power_tch.torch_numeric,
    xexp: xexp_tch.torch_numeric,
}