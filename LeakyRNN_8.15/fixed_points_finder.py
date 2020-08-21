from gene_seq import *
from torch import optim
from torch.autograd import Variable
import time


def find_fixed_points(rnn_fun, candidates, hps, unique=True ,do_print=True):
    """Top-level routine to find fixed points, keeping only valid fixed points.
    This function will:
      Add noise to the fixed point candidates ('noise_var')
      Optimize to find the closest fixed points / slow points (many hps,
        see optimize_fps)
      Exclude any fixed points whose fixed point loss is above threshold ('fp_tol')
      Exclude any non-unique fixed points according to a tolerance ('unique_tol')
      Exclude any far-away "outlier" fixed points ('outlier_tol')

    This top level function runs at the CPU level, while the actual JAX optimization
    for finding fixed points is dispatched to device.
    Arguments:
      rnn_fun: one-step update function as a function of hidden state
      candidates: torch.tensor with shape npoints x ndims
      hps: dict of hyper parameters for fp optimization, including
        tolerances related to keeping fixed points

    Returns:
      4-tuple of (kept fixed points sorted with slowest points first,
        fixed point losses, indicies of kept fixed points, details of
        optimization)"""
    npoints = candidates.size()[0]
    dim = candidates.size()[1]
    noise_var = hps['noise_var']
    if do_print and noise_var > 0.0:
        print("Adding noise to fixed point candidates.")
        candidates += torch.randn(candidates.size()) * noise_var ** 0.5

    if do_print:
        print("Optimizing to find fixed points.")
    fps, opt_details = optimize_fps(rnn_fun, candidates, hps, do_print)

    if do_print and hps['fp_tol'] < np.inf:
        print("Excluding fixed points with squared speed above tolerance {:0.7f}.".format(hps['fp_tol']))

    fps = fixed_points_with_tolerance(rnn_fun, fps, hps['fp_tol'], do_print)
    # if len(fps) == 0:
    #     return torch.zeros([0, dim]), torch.tensor([0]), [], opt_details

    if do_print and hps['unique_tol'] > 0.0 and unique:
        print("Excluding non-unique fixed points.")
    if unique:
        fps = keep_unique_fixed_points(fps, hps['unique_tol'], do_print)
    # if len(unique_kidxs) == 0:
    #     return np.zeros([0, dim]), np.zeros([0]), [], opt_details

    # if do_print and hps['outlier_tol'] < np.inf:
    #     print("Excluding outliers.")
    # fps, outlier_kidxs = exclude_outliers(fps, hps['outlier_tol'],
    #                                       'euclidean', do_print)  # TODO(sussillo) Make hp?
    # if len(outlier_kidxs) == 0:
    #     return np.zeros([0, dim]), np.zeros([0]), [], opt_details

    # if do_print:
    #     print('Sorting fixed points slowest first.')
    loss_fun = get_fp_loss_fun(rnn_fun)
    losses = []
    for point in fps:
        losses.append(float(loss_fun(point)))

    # try:
    #     keep_idxs = fp_kidxs[unique_kidxs[outlier_kidxs[sort_idxs]]]
    # except:
    #     import pdb
    #     pdb.set_trace()
    fps = fps.detach().numpy()
    return fps, losses, opt_details


def get_fp_loss_fun(rnn):
    """Return the per-example mean-squared-error fixed point loss.
    Arguments:
      rnn_fun : RNN one step update function for a single hidden state vector
        h_t -> h_t+1
    Returns: function that computes the loss for each example
    :type rnn: myRNN
    """

    loss_fun = nn.MSELoss()
    update_fun = rnn.get_one_step_fun()
    return lambda h: loss_fun(h, update_fun(h, input=torch.zeros(rnn.N))[0])


def get_total_fp_loss_fun(rnn_fun):
    """Return the MSE fixed point loss averaged across examples.
    Arguments:
      rnn_fun : RNN one step update function for a single hidden state vector
        h_t -> h_t+1
    Returns: function that computes the average loss over all examples.
    """
    fp_loss_fun = get_fp_loss_fun(rnn_fun)

    def loss(fp_candidates):
        result = 0
        for point in fp_candidates:
            result += fp_loss_fun(point)
        return result / len(fp_candidates)

    return loss


def adjust_learning_rate(optimizer, epoch, hps):
    """Sets the learning rate to the initial LR decayed by decay_rate every period"""
    lr = hps['learning_rate'] * (hps['decay_rate'] ** (epoch // hps['decay_period']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def timeSince(since):
    now = time.time()
    s = now - since
    # m = math.floor(s / 60)
    # s -= m * 60
    return s


def  optimize_fps(rnn_fun, fp_candidates, hps, do_print=True):
    """Find fixed points of the rnn via optimization.
    This loop is at the cpu non-JAX level.
    Arguments:
      rnn_fun : RNN one step update function for a single hidden state vector
        h_t -> h_t+1, for which the fixed point candidates are trained to be
        fixed points
      fp_candidates: torch.tensor with shape (batch size, state dim) of hidden states
        of RNN to start training for fixed points
      hps: fixed point hyperparameters
      do_print: Print useful information?
    Returns:
      numerically optimized fixed points(torch.tensor), total loss of each epoch"""

    batch_size = fp_candidates.size()[0]
    num_batches = hps['num_batches']
    print_every = hps['opt_print_every']
    # num_opt_loops = int(num_batches / print_every)

    fp_candidates = Variable(fp_candidates, requires_grad=True)
    optimizer = optim.Adam([fp_candidates], lr=hps['learning_rate'])
    total_fp_loss_fun = get_total_fp_loss_fun(rnn_fun)
    fp_losses = []
    do_stop = False
    for epoch in range(num_batches):
        if do_stop:
            break

        adjust_learning_rate(optimizer, epoch, hps)
        optimizer.zero_grad()
        loss = total_fp_loss_fun(fp_candidates)
        fp_losses.append(loss)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            start = time.time()
        if do_print and epoch % print_every == 0 and epoch != 0:
            s = "Batches %s - %s in %.2f sec, Step size: %.5f, Training loss %.7f" % (
                epoch - print_every, epoch, timeSince(start), adjust_learning_rate(optimizer, epoch, hps), loss)
            start = time.time()
            print(s)
        if loss < hps['fp_opt_stop_tol']:
            do_stop = True
            if do_print:
                print('Stopping as mean training loss %.7f is below tolerance %.7f.' % (loss, hps[
                    'fp_opt_stop_tol']))
    optimizer_details = {"fp_losses": fp_losses}
    return fp_candidates, optimizer_details


def fixed_points_with_tolerance(rnn_fun, fps, tol=float('inf'), do_print=True):
    """Return fixed points with a fixed point loss under a given tolerance.

    Arguments:
      rnn_fun: one-step update function as a function of hidden state
      fps: torch.tensor with shape npoints x ndims
      tol: loss tolerance over which fixed points are excluded
      do_print: Print useful information?
    Returns:
      kept fixed points
    """
    fps_num = fps.size()[0]
    fp_loss_fun = get_fp_loss_fun(rnn_fun)
    kept_fps = torch.tensor([])
    for fp in fps:
        if float(fp_loss_fun(fp)) < tol:
            kept_fps = torch.cat((kept_fps, fp.unsqueeze(0)), 0)
    if do_print:
        print("Kept %d/%d fixed points with tolerance under %f." %
              (kept_fps.size()[0], fps_num, tol))
    return kept_fps


def keep_unique_fixed_points(fps, identical_tol=0.0, do_print=True):
    """Get unique fixed points by choosing a representative within tolerance.
    Args:
      fps: torch.tensor, FxN tensor of F fixed points of N dimension
      identical_tol: float, tolerance for determination of identical fixed points
      do_print: Print useful information?
    Returns:
        UxN torch.tensor of U unique fixed points and the kept indices
    """
    distance = nn.MSELoss()
    kept_points = torch.tensor([])
    for i in range(fps.size()[0]):
        add_in = True
        point = fps[i]
        if len(kept_points) == 0:
            add_in = True
        else:
            for other in fps:
                if float(distance(point, other)) < identical_tol and other in kept_points:
                    add_in = False
                    break
        if add_in:
            kept_points = torch.cat((kept_points, point.unsqueeze(0)), 0)
    if do_print:
        print("Kept %d/%d fixed points with tolerance under %f." %
              (kept_points.size()[0], fps.size()[0], identical_tol))
    return kept_points


def exclude_outliers(data, outlier_dist=np.inf, metric='euclidean', do_print=True):
    """Exclude points whose closest neighbor is further than threshold.
    Args:
      data: ndarray, matrix holding datapoints (num_points x num_features).
      outlier_dist: float, distance to determine outliers.
      metric: str or function, distance metric passed to scipy.spatial.pdist.
          Defaults to "euclidean"
      do_print: Print useful information?
    Returns:
      2-tuple of (filtered_data: ndarray, matrix holding subset of datapoints,
        keep_idx: ndarray, vector of bools holding indices of kept datapoints).
    """


def compute_jacobians(rnn, points):
    """Compute the jacobians of the rnn_fun at the points.
    Arguments:
      rnn_fun: RNN one step update function for a single hidden state vector
        h_t -> h_t+1
      points: torch.tensor npoints x dim, eval jacobian at this point.
    Returns:
      npoints number of jacobians, torch.tensor with shape npoints x dim x dim
    """
    # rnn_fun = rnn.get_one_step_fun()
    # jacobians = torch.tensor([])
    # dim = points[0].size()[0]
    # for fp in points:
    #     fp.requires_grad = True
    #     fp = fp.unsqueeze(0).repeat(dim, *([1] * dim)).detach().requires_grad_(True)
    #     next_state = rnn_fun(fp, input_t=torch.zeros(rnn_fun.N))
    #     next_state.backward(torch.eye(dim))
    #     jacob = fp.grad.reshape(dim, *dim)
    #     jacobians = torch.cat((jacobians, jacob.unsqueeze(0)), dim=0)
    # return jacobians

def compute_eigenvalue_decomposition(Ms, sort_by='magnitude', do_compute_lefts=True):
    """Compute the eigenvalues of the matrix M. No assumptions are made on M.
    Arguments:
      M: 3D np.array nmatrices x dim x dim matrix
      do_compute_lefts: Compute the left eigenvectors? Requires a pseudo-inverse
        call.
    Returns:
      list of dictionaries with eigenvalues components: sorted
        eigenvalues, sorted right eigenvectors, and sored left eigenvectors
        (as column vectors).
    """
