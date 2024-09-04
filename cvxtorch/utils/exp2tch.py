from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import DivExpression, MulExpression, multiply
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.cumsum import cumsum
from cvxpy.atoms.affine.diag import diag_mat, diag_vec
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.atoms.affine.imag import imag
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
    cummax: cummax_tch,
    dist_ratio: dist_ratio_tch,
    dotsort: dotsort_tch,
    eye_minus_inv: eye_minus_inv_tch,
    geo_mean: geo_mean_tch,
    gmatmul: gmatmul_tch,
    length: length_tch,
    log_det: log_det_tch,
    log_sum_exp: log_sum_exp_tch,
    MatrixFrac: matrix_frac_tch,
    max: max_tch,
    min: min_tch,
    norm_inf: norm_inf_tch,
    normNuc: norm_nuc_tch,
    norm1: norm1_tch,
    one_minus_pos: one_minus_pos_tch,
    pf_eigenvalue: pf_eigenvalue_tch,
    Pnorm: pnorm_tch,
    Prod: prod_tch,
    QuadForm: quad_form_tch,
    quad_over_lin: quad_over_lin_tch,
    sigma_max: sigma_max_tch,
    sign: sign_tch,
    tr_inv: tr_inv_tch,
    AddExpression: add_expr_tch,
    MulExpression: mul_expression_tch,
    multiply: multiply_tch,
    DivExpression: div_expression_tch,
    conj: conj_tch,
    cumsum: cumsum_tch,
    diag_vec: diag_vec_tch,
    diag_mat: diag_mat_tch,
    Hstack: hstack_tch,
    imag: imag_tch,
    kron: kron_tch,
    NegExpression: unary_operators_tch,
    Promote: promote_tch,
    real: real_tch,
    reshape: reshape_tch,
    Sum: sum_tch,
    trace: trace_tch,
    transpose: transpose_tch,
    upper_tri: upper_tri_tch,
    Vstack: vstack_tch,
    abs: abs_tch,
    ceil: ceil_tch,
    exp: exp_tch,
    log: log_tch,
    logistic: logistic_tch,
    maximum: maximum_tch,
    minimum: minimum_tch,
    power: power_tch,
    xexp: xexp_tch,
}