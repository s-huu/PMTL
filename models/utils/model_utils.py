from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from torchvision import datasets, transforms
from collections import defaultdict
import random
import numpy as np
import torch
from .so_tag_utils import *
import math
import sys
import six
from PIL import Image
from scipy import special

VOCAB_DIR = 0
emb_array = 0
vocab = 0
embed_dim = 0
IMAGES_DIR = "../data/celeba/data/raw/img_align_celeba"
IMAGE_SIZE = 84


def batch_data(data, batch_size, rng=None, shuffle=True, eval_mode=False, full=False, malicious=False, dataset='femnist'):
    """
    data is a dict := {'x': [list], 'y': [list]} with optional fields 'y_true': [list], 'x_true' : [list]
    If eval_mode, use 'x_true' and 'y_true' instead of 'x' and 'y', if such fields exist
    returns x, y, which are both lists of size-batch_size lists
    """
    x = data['x_true'] if eval_mode and 'x_true' in data else data['x']
    y = data['y_true'] if eval_mode and 'y_true' in data else data['y']
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    # print(len(x))

    if dataset=='celeba':
        x = process_x(x)
        y = process_y(y)
    else:
        x = np.asarray(x)
        y = np.asarray(y)
    # print(x,x.shape,y,y.shape)

    if dataset=='celeba':
        x = torch.tensor((x/1.)).cuda()
        x = torch.transpose(torch.transpose(x,-1,-2),-2,-3)
    else:
        x = torch.tensor(x).cuda()
    y = torch.LongTensor(y).cuda()
    raw_x = x[indices]
    raw_y = y[indices]
    batched_x, batched_y = [], []
    if not full:
        for i in range(0, len(raw_x), batch_size):
            batched_x.append(raw_x[i:i + batch_size])
            batched_y.append(raw_y[i:i + batch_size])
    else:
        batched_x.append(raw_x)
        batched_y.append(raw_y)
    return batched_x, batched_y

def load_image(img_name):
    img = Image.open(os.path.join(IMAGES_DIR, img_name))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
    return np.array(img)

def process_x(raw_x_batch):
    x_batch = [load_image(i) for i in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    return raw_y_batch


def read_so_data():
    groups = []
    train_data, test_data = get_centralized_datasets()
    clients = {
        'train_users': list(train_data.keys()),
        'test_users': list(test_data.keys())
    }
    return clients, groups, train_data, test_data

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(train_data.keys())

    return clients, groups, train_data, test_data

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RDP analysis of the Sampled Gaussian Mechanism.
Functionality for computing Renyi differential privacy (RDP) of an additive
Sampled Gaussian Mechanism (SGM). Its public interface consists of two methods:
  compute_rdp(q, noise_multiplier, T, orders) computes RDP for SGM iterated
                                   T times.
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).
Example use:
Suppose that we have run an SGM applied to a function with l2-sensitivity 1.
Its parameters are given as a list of tuples (q1, sigma1, T1), ...,
(qk, sigma_k, Tk), and we wish to compute eps for a given delta.
The example code would be:
  max_order = 32
  orders = range(2, max_order + 1)
  rdp = np.zeros_like(orders, dtype=float)
  for q, sigma, T in parameters:
   rdp += rdp_accountant.compute_rdp(q, sigma, T, orders)
  eps, _, opt_order = rdp_accountant.get_privacy_spent(rdp, target_delta=delta)
"""



########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx, logy):
  """Add two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
  """Subtract two numbers in the log space. Answer must be non-negative."""
  if logx < logy:
    raise ValueError("The result of subtraction must be non-negative.")
  if logy == -np.inf:  # subtracting 0
    return logx
  if logx == logy:
    return -np.inf  # 0 is represented as -np.inf in the log space.

  try:
    # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
    return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
  except OverflowError:
    return logx


def _log_print(logx):
  """Pretty print."""
  if logx < math.log(sys.float_info.max):
    return "{}".format(math.exp(logx))
  else:
    return "exp({})".format(logx)


def _log_comb(n, k):
  return (special.gammaln(n + 1) -
          special.gammaln(k + 1) - special.gammaln(n - k + 1))


def _compute_log_a_int(q, sigma, alpha):
  """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
  assert isinstance(alpha, six.integer_types)

  # Initialize with 0 in the log space.
  log_a = -np.inf

  for i in range(alpha + 1):
    log_coef_i = (
        _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q))

    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

  return float(log_a)


def _compute_log_a_frac(q, sigma, alpha):
  """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
  # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
  # initialized to 0 in the log space:
  log_a0, log_a1 = -np.inf, -np.inf
  i = 0

  z0 = sigma**2 * math.log(1 / q - 1) + .5

  while True:  # do ... until loop
    coef = special.binom(alpha, i)
    log_coef = math.log(abs(coef))
    j = alpha - i

    log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
    log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

    log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

    log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
    log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

    if coef > 0:
      log_a0 = _log_add(log_a0, log_s0)
      log_a1 = _log_add(log_a1, log_s1)
    else:
      log_a0 = _log_sub(log_a0, log_s0)
      log_a1 = _log_sub(log_a1, log_s1)

    i += 1
    if max(log_s0, log_s1) < -30:
      break

  return _log_add(log_a0, log_a1)


def _compute_log_a(q, sigma, alpha):
  """Compute log(A_alpha) for any positive finite alpha."""
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha))
  else:
    return _compute_log_a_frac(q, sigma, alpha)


def _log_erfc(x):
  """Compute log(erfc(x)) with high accuracy for large x."""
  try:
    return math.log(2) + special.log_ndtr(-x * 2**.5)
  except NameError:
    # If log_ndtr is not available, approximate as follows:
    r = special.erfc(x)
    if r == 0.0:
      # Using the Laurent series at infinity for the tail of the erfc function:
      #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
      # To verify in Mathematica:
      #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
      return (-math.log(math.pi) / 2 - math.log(x) - x**2 - .5 * x**-2 +
              .625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
    else:
      return math.log(r)


def _compute_delta(orders, rdp, eps):
  """Compute delta given a list of RDP values and target epsilon.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    eps: The target epsilon.
  Returns:
    Pair of (delta, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if eps < 0:
    raise ValueError("Value of privacy loss bound epsilon must be >=0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   delta = min( np.exp((rdp_vec - eps) * (orders_vec - 1)) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4):
  logdeltas = []  # work in log space to avoid overflows
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1: raise ValueError("Renyi divergence order must be >=1.")
    if r < 0: raise ValueError("Renyi divergence must be >=0.")
    # For small alpha, we are better of with bound via KL divergence:
    # delta <= sqrt(1-exp(-KL)).
    # Take a min of the two bounds.
    logdelta = 0.5*math.log1p(-math.exp(-r))
    if a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value for alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      rdp_bound = (a - 1) * (r - eps + math.log1p(-1/a)) - math.log(a)
      logdelta = min(logdelta, rdp_bound)

    logdeltas.append(logdelta)

  idx_opt = np.argmin(logdeltas)
  return min(math.exp(logdeltas[idx_opt]), 1.), orders_vec[idx_opt]


def _compute_eps(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
  # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
  eps_vec = []
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1: raise ValueError("Renyi divergence order must be >=1.")
    if r < 0: raise ValueError("Renyi divergence must be >=0.")

    if delta**2 + math.expm1(-r) >= 0:
      # In this case, we can simply bound via KL divergence:
      # delta <= sqrt(1-exp(-KL)).
      eps = 0  # No need to try further computation if we have eps = 0.
    elif a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value of alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      eps = r + math.log1p(-1/a) - math.log(delta * a) / (a - 1)
    else:
      # In this case we can't do anything. E.g., asking for delta = 0.
      eps = np.inf
    eps_vec.append(eps)

  idx_opt = np.argmin(eps_vec)
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def _compute_rdp(q, sigma, alpha):
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.
  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.
  Returns:
    RDP at alpha, can be np.inf.
  """
  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(q, noise_multiplier, steps, orders):
  """Computes RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
        to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array([_compute_rdp(q, noise_multiplier, order)
                    for order in orders])

  return rdp * steps


def compute_heterogenous_rdp(sampling_probabilities, noise_multipliers,
                             steps_list, orders):
  """Computes RDP of Heteregoneous Applications of Sampled Gaussian Mechanisms.
  Args:
    sampling_probabilities: A list containing the sampling rates.
    noise_multipliers: A list containing the noise multipliers: the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of the
      function to which it is added.
    steps_list: A list containing the number of steps at each
      `sampling_probability` and `noise_multiplier`.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  assert len(sampling_probabilities) == len(noise_multipliers)

  rdp = 0
  for q, noise_multiplier, steps in zip(sampling_probabilities,
                                        noise_multipliers, steps_list):
    rdp += compute_rdp(q, noise_multiplier, steps, orders)

  return rdp


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Computes delta (or eps) for given eps (or delta) from RDP values.
  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not `None`, the epsilon for which we compute the
      corresponding delta.
    target_delta: If not `None`, the delta for which we compute the
      corresponding epsilon. Exactly one of `target_eps` and `target_delta`
      must be `None`.
  Returns:
    A tuple of epsilon, delta, and the optimal order.
  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  if target_eps is not None:
    delta, opt_order = _compute_delta(orders, rdp, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(orders, rdp, target_delta)
    return eps, target_delta, opt_order


def compute_rdp_from_ledger(ledger, orders):
  """Computes RDP of Sampled Gaussian Mechanism from ledger.
  Args:
    ledger: A formatted privacy ledger.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    RDP at all orders. Can be `np.inf`.
  """
  total_rdp = np.zeros_like(orders, dtype=float)
  for sample in ledger:
    # Compute equivalent z from l2_clip_bounds and noise stddevs in sample.
    # See https://arxiv.org/pdf/1812.06210.pdf for derivation of this formula.
    effective_z = sum([
        (q.noise_stddev / q.l2_norm_bound)**-2 for q in sample.queries])**-0.5
    total_rdp += compute_rdp(
        sample.selection_probability, effective_z, 1, orders)
  return total_rdp

def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  # compute_rdp requires that sigma be the ratio of the standard deviation of
  # the Gaussian noise to the l2-sensitivity of the function to which it is
  # added. Hence, sigma here corresponds to the `noise_multiplier` parameter
  # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
  rdp = compute_rdp(q, sigma, steps, orders)

  eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

  print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
        ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
  print('differential privacy with eps = {:.3g} and delta = {}.'.format(
      eps, delta))
  print('The optimal RDP order is {}.'.format(opt_order))

  if opt_order == max(orders) or opt_order == min(orders):
    print('The privacy estimate is likely to be improved by expanding '
          'the set of orders.')

  return eps, opt_order


def compute_dp_sgd_privacy(q, noise_multiplier, epochs, delta):
  """Compute epsilon based on the given hyperparameters."""
  if q > 1:
    raise app.UsageError('n must be larger than the batch size.')
  orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
  return apply_dp_sgd_analysis(q, noise_multiplier, epochs, orders, delta)


