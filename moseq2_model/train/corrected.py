"""Corrected sampler components for the AR-HMM.

These subclass the installed ``pyhsmm`` / ``pybasicbayes`` / ``autoregressive``
classes and override individual methods, so the compiled dependencies are used
unmodified.

Two of the corrections repair defects in the inherited sampler:

``GapAwareStickyHDPHMMTransitions``
    Draws the Chinese restaurant franchise table counts with the sticky mass
    included on the diagonal, per Fox et al. (2011). The inherited
    implementation calls ``sample_crp_tablecounts(alpha, counts, beta)``, which
    applies ``alpha * beta_k`` to every cell and so omits ``kappa`` where the
    restaurant and dish coincide.

``CorrectedARHMMMixin.resample_trans_distn``
    Counts only transitions between two valid frames. The inherited
    implementation counts every consecutive pair in the state sequence,
    including pairs that straddle a dropped frame, which treats a gap as a real
    frame-to-frame transition.

The third correction aligns a numerical policy rather than repairing a defect:

``BoostedAutoRegression``
    Regularizes the matrix inversions in the matrix-normal inverse-Wishart
    update by adding a small multiple of the identity *before* inverting, rather
    than padding the result afterwards.
"""

import collections
import copy

import numpy as np
from autoregressive.models import (
    ARWeakLimitStickyHDPHMM,
    ARWeakLimitStickyHDPHMMSeparateTrans,
    FastARWeakLimitStickyHDPHMM,
    FastARWeakLimitStickyHDPHMMSeparateTrans,
)
from pybasicbayes.distributions import Multinomial
from pybasicbayes.distributions.regression import AutoRegression
from pybasicbayes.util.general import inv_psd
from pyhsmm.internals.transitions import WeakLimitStickyHDPHMMTransitions

# Matches the boost jax-moseq applies inside its positive-semi-definite solve
# and inverse helpers.
DIAGONAL_BOOST = 1e-6

# Matches the floor jax-moseq applies to a resampled transition row
# (``jax_moseq/utils/transitions.py``).
TRANSITION_FLOOR = float(np.finfo(np.float32).tiny)


def sample_crf_table_counts(customer_counts, concentration, rng=None):
    """Antoniak draw of table counts, with a per-cell concentration.

    For each restaurant/dish pair, the number of occupied tables is the number
    of customers who chose a new table, where the ``k``-th customer does so with
    probability ``c / (k + c)``. Unlike
    ``pybasicbayes.util.general.sample_crp_tablecounts``, the concentration
    ``c`` varies per cell rather than per dish, which is what allows the sticky
    mass to be applied on the diagonal.

    Args:
    customer_counts (np.ndarray): integer counts, shape (N, N).
    concentration (np.ndarray): per-cell concentration, shape (N, N).
    rng (np.random.Generator): optional generator; defaults to global state.

    Returns:
    table_counts (np.ndarray): integer table counts, shape (N, N).
    """
    counts = np.asarray(customer_counts)
    n_max = int(counts.max()) if counts.size else 0
    if n_max == 0:
        return np.zeros_like(counts)

    draw = np.random.random if rng is None else rng.random
    k = np.arange(n_max)
    c = np.asarray(concentration, dtype=np.float64)[..., None]
    p = c / (k + c)
    hit = (draw(p.shape) < p) & (k < counts[..., None])
    return hit.sum(-1).astype(counts.dtype)


class BoundedMultinomial(Multinomial):
    """Multinomial whose Gibbs draw is floored at jax-moseq's epsilon.

    ``pybasicbayes`` clips a resampled row at ``np.spacing(1.)``, 2.2e-16, to
    guard against exact zeros. The guard stops being inert at the stickiness
    these models run at: a weak-limit HDP row is drawn at a per-cell
    concentration near ``alpha / num_states``, about 0.06, which spreads the
    off-diagonal entries from 1e-38 up to 1e-8. The clip therefore binds on
    roughly a quarter of every row and raises those entries by up to twenty
    orders of magnitude, turning an 80-nat penalty for entering a rare state
    into a 36-nat one.

    Flooring at the smallest positive single-precision number instead keeps the
    guard against exact zeros without reshaping the row.
    """

    def resample(self, data=[], counts=None):
        counts = self._get_statistics(data) if counts is None else counts
        weights = np.random.dirichlet(self.alphav_0 + counts)
        weights = weights + TRANSITION_FLOOR
        self.weights = weights / weights.sum()
        # Retained from the inherited implementation so a Gibbs draw can still
        # seed mean field.
        self._alpha_mf = self.weights * self.alphav_0.sum()
        return self


class GapAwareStickyHDPHMMTransitions(WeakLimitStickyHDPHMMTransitions):
    """Sticky HDP-HMM transitions whose table counts include the sticky mass."""

    def __init__(self, *args, **kwargs):
        super(GapAwareStickyHDPHMMTransitions, self).__init__(*args, **kwargs)
        self._bound_rows()

    def _bound_rows(self):
        """Give every transition row the bounded resampling rule.

        The rows are built by ``_HMMTransitionsBase`` and by its
        ``trans_matrix`` setter, neither of which exposes the row class.
        ``BoundedMultinomial`` adds no state and overrides one method, so
        rebinding in place carries every attribute across exactly.
        """
        for distn in self._row_distns:
            distn.__class__ = BoundedMultinomial

    @property
    def trans_matrix(self):
        return WeakLimitStickyHDPHMMTransitions.trans_matrix.fget(self)

    @trans_matrix.setter
    def trans_matrix(self, value):
        WeakLimitStickyHDPHMMTransitions.trans_matrix.fset(self, value)
        self._bound_rows()

    def crf_concentration(self):
        """Per-cell CRP concentration used when sampling the table counts.

        Exposed so a probe can report what this class actually applies rather
        than assuming a formula. The inherited implementation has no equivalent,
        because ``sample_crp_tablecounts`` takes a per-dish weight and cannot
        express a per-cell concentration at all.
        """
        num_states = len(self.beta)
        conc = np.tile(self.alpha * self.beta, (num_states, 1))
        return conc + self.kappa * np.eye(num_states)

    def _get_m(self, trans_counts):
        """Sample the auxiliary table counts and apply the override step.

        The concentration is ``alpha * beta_k + kappa`` on the diagonal and
        ``alpha * beta_k`` elsewhere. The diagonal is then thinned to the share
        attributable to ``beta`` rather than to the sticky component, which is
        unchanged from the inherited behavior.
        """
        counts = np.asarray(trans_counts, dtype=np.int64)
        num_states = counts.shape[0]

        concentration = np.tile(self.alpha * self.beta, (num_states, 1))
        concentration = concentration + self.kappa * np.eye(num_states)

        table_counts = sample_crf_table_counts(counts, concentration)

        diagonal = np.diag(table_counts).copy()
        if diagonal.sum() > 0:
            weights = self.alpha * self.beta
            keep = weights / (weights + self.kappa)
            np.fill_diagonal(
                table_counts, np.random.binomial(diagonal, keep)
            )
        return table_counts.astype(np.int32)


def gap_aware_transition_counts(stateseq, valid, num_states):
    """Count transitions whose destination row is valid.

    The inherited counter walks every consecutive pair of the state sequence,
    so a dropped frame is counted as a real frame-to-frame transition. This
    excludes any transition landing on an invalid row.

    A transition *out of* an invalid row is still counted. Requiring both rows
    would be the stricter reading, but jax-moseq weights each transition by its
    destination frame alone (``jax_moseq/utils/transitions.py:40-47``), and
    matching it exactly is worth more here than the stricter rule: the two
    differ only on rows adjacent to a gap, and an unexplained disagreement
    between the backends costs more than the handful of transitions involved.

    Args:
    stateseq (np.ndarray): state sequence, one entry per autoregressive row.
    valid (np.ndarray): boolean validity, index-aligned with ``stateseq``.
    num_states (int): size of the returned matrix.

    Returns:
    counts (np.ndarray): integer transition counts, shape (num_states, num_states).
    """
    counts = np.zeros((num_states, num_states), dtype=np.int32)
    keep = valid[1:]
    if keep.any():
        np.add.at(counts, (stateseq[:-1][keep], stateseq[1:][keep]), 1)
    return counts


class CorrectedARHMMMixin(object):
    """Model-level corrections shared by the corrected AR-HMM classes."""

    _trans_distn_class = GapAwareStickyHDPHMMTransitions

    def __init__(self, *args, **kwargs):
        super(CorrectedARHMMMixin, self).__init__(*args, **kwargs)
        # WeakLimitStickyHDPHMM constructs its transition distribution directly
        # rather than through a class attribute, so the corrected one replaces
        # it after construction, carrying over the sampled state. The
        # separate-transition variant has already converted that single object
        # into a per-group defaultdict by this point, so both shapes are
        # handled.
        if hasattr(self, "trans_distn"):
            self.trans_distn = self._convert(self.trans_distn)
        else:
            prototype = self._convert(self._trans_distn_prototype)
            self._trans_distn_prototype = prototype
            existing = dict(self.trans_distns)
            self.trans_distns = collections.defaultdict(
                lambda: copy.deepcopy(prototype)
            )
            for group_id, distn in existing.items():
                self.trans_distns[group_id] = self._convert(distn)

    def _convert(self, stock):
        """Rebuild a stock transitions object as the corrected class.

        The transition matrix is redrawn so that it reflects ``kappa``.
        ``WeakLimitStickyHDPHMM.__init__`` writes ``alphav`` directly instead of
        going through the sticky property setter, so a freshly constructed model
        carries no ``kappa`` on its diagonal: at ``kappa = 4.2e8`` the initial
        transition matrix comes out essentially uniform, with self-transitions
        marginally *less* likely than switching. Assigning ``beta`` invokes
        ``_set_alphav``, and resampling against zero counts then draws the
        matrix from the sticky prior the model was asked for.
        """
        if isinstance(stock, self._trans_distn_class):
            corrected = stock
        else:
            corrected = self._trans_distn_class(
                num_states=stock.N,
                alpha=stock.alpha,
                gamma=stock.gamma,
                kappa=stock.kappa,
                beta=stock.beta,
                trans_matrix=stock.trans_matrix,
            )

        corrected.beta = corrected.beta
        corrected.resample(
            trans_counts=np.zeros((corrected.N, corrected.N), dtype=np.int32)
        )
        return corrected

    def _row_validity(self, states):
        """Which autoregressive rows of one session are usable.

        ``states.data`` is the strided autoregressive data, index-aligned with
        ``states.stateseq``, so a single missing frame appears in ``nlags + 1``
        consecutive rows and is excluded from all of them.
        """
        return ~np.isnan(states.data).any(axis=1)

    def _counts_for(self, states_list):
        counts = np.zeros((self.num_states, self.num_states), dtype=np.int32)
        for s in states_list:
            counts += gap_aware_transition_counts(
                np.asarray(s.stateseq), self._row_validity(s), self.num_states
            )
        return counts

    def resample_trans_distn(self):
        if hasattr(self, "trans_distn"):
            self.trans_distn.resample(trans_counts=self._counts_for(self.states_list))
        else:
            for group_id, trans_distn in self.trans_distns.items():
                members = [
                    s for s in self.states_list
                    if hash(s.group_id) == hash(group_id)
                ]
                trans_distn.resample(trans_counts=self._counts_for(members))
        self._clear_caches()


class BoostedAutoRegression(AutoRegression):
    """AutoRegression whose MNIW conversions boost the diagonal before inverting.

    The inherited implementation inverts the natural parameter directly and then
    pads the results by ``1e-8``. Boosting beforehand instead matches the policy
    jax-moseq applies in its positive-semi-definite helpers, and matters most
    for states carrying few frames, where the matrix being inverted is small
    relative to a fixed pad.
    """

    @staticmethod
    def _standard_to_natural(nu, S, M, K):
        Kinv = inv_psd(K + DIAGONAL_BOOST * np.eye(K.shape[0]))
        A = S + M.dot(Kinv).dot(M.T)
        B = M.dot(Kinv)
        C = Kinv
        d = nu
        return np.array([A, B, C, d])

    @staticmethod
    def _natural_to_standard(natparam):
        A, B, C, d = natparam
        nu = d
        Kinv = C + DIAGONAL_BOOST * np.eye(C.shape[0])
        K = inv_psd(Kinv)
        M = np.linalg.solve(Kinv, B.T).T
        S = A - M.dot(B.T)
        S = (S + S.T) / 2.0
        return nu, S, M, K


class CorrectedFastARHMM(CorrectedARHMMMixin, FastARWeakLimitStickyHDPHMM):
    pass


class CorrectedFastARHMMSeparateTrans(
    CorrectedARHMMMixin, FastARWeakLimitStickyHDPHMMSeparateTrans
):
    pass


class CorrectedRobustARHMM(CorrectedARHMMMixin, ARWeakLimitStickyHDPHMM):
    pass


class CorrectedRobustARHMMSeparateTrans(
    CorrectedARHMMMixin, ARWeakLimitStickyHDPHMMSeparateTrans
):
    pass
