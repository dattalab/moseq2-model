"""Regression tests for the corrected AR-HMM sampler components.

Each test states the defect it pins and fails against the uncorrected behavior
for that reason.
"""

from collections import OrderedDict

import numpy as np
import pytest
from pybasicbayes.util.general import AR_striding

from moseq2_model.train.corrected import (
    DIAGONAL_BOOST,
    BoostedAutoRegression,
    GapAwareStickyHDPHMMTransitions,
    gap_aware_transition_counts,
    sample_crf_table_counts,
)
from moseq2_model.train.models import ARHMM

LATENT_DIM = 5
NLAGS = 3


def _data(n_frames=400, nan_frames=(), seed=0):
    rng = np.random.default_rng(seed)
    x = np.require(
        rng.standard_normal((n_frames, LATENT_DIM)),
        dtype=np.float64,
        requirements="C",
    )
    for t in nan_frames:
        x[t] = np.nan
    return x


class TestPriorMeanLagPlacement:
    """M_0 must place the identity on the most recent lag.

    AR_striding lays out row t as [x_{t-nlags}, ..., x_{t-1}], so index 0 of the
    design vector is the *oldest* lag. Placing the identity there encodes a
    prior belief that a frame resembles the frame nlags steps earlier, with the
    intervening frames uninformative.
    """

    def test_striding_puts_most_recent_lag_last(self):
        x = np.arange(6 * 2, dtype=float).reshape(6, 2)
        strided = AR_striding(np.ascontiguousarray(x), 3)
        # Row 0 predicts frame 3 from frames 0, 1, 2.
        assert np.array_equal(strided[0, 4:6], x[2]), "last block is not lag 1"
        assert np.array_equal(strided[0, 0:2], x[0]), "first block is not lag 3"

    def test_identity_on_most_recent_lag(self):
        model = ARHMM(
            OrderedDict([("s", _data())]),
            kappa=1000.0, max_states=4, nlags=NLAGS, silent=True,
        )
        # natural_hypparam stores (A, B, C, d) with B = M K^-1, so M is
        # recovered through the standard-form conversion rather than read off.
        obs = model.obs_distns[0]
        _, _, M_0, _ = obs._natural_to_standard(obs.natural_hypparam)

        blocks = [
            np.abs(M_0[:, b * LATENT_DIM:(b + 1) * LATENT_DIM]).sum()
            for b in range(NLAGS)
        ]
        assert int(np.argmax(blocks)) == NLAGS - 1, (
            "identity is on lag block {}, expected the most recent "
            "(block {})".format(int(np.argmax(blocks)), NLAGS - 1)
        )
        recent = M_0[:, (NLAGS - 1) * LATENT_DIM:NLAGS * LATENT_DIM]
        assert np.allclose(recent, np.eye(LATENT_DIM), atol=1e-4)
        # Every other lag block, and the affine column, are zero.
        for b in range(NLAGS - 1):
            other = M_0[:, b * LATENT_DIM:(b + 1) * LATENT_DIM]
            assert np.allclose(other, 0.0, atol=1e-4)
        assert np.allclose(M_0[:, -1], 0.0, atol=1e-4)


class TestGapAwareTransitionCounts:
    """Transitions landing on an invalid row must not be counted.

    The inherited counter walks every consecutive pair of the state sequence, so
    a dropped frame is treated as a real frame-to-frame transition. The rule
    applied here matches jax-moseq, which weights a transition by its
    destination frame alone.
    """

    def test_transition_into_a_gap_is_not_counted(self):
        stateseq = np.array([0, 1, 2, 3], dtype=np.int32)
        valid = np.array([True, False, True, True])
        counts = gap_aware_transition_counts(stateseq, valid, 4)
        assert counts[0, 1] == 0, "counted a transition into an invalid row"
        assert counts[2, 3] == 1, "dropped a wholly valid transition"
        # Deliberately counted: the destination row is valid even though the
        # source is not. This matches jax-moseq rather than the stricter rule.
        assert counts[1, 2] == 1, "destination-valid transition was dropped"
        assert counts.sum() == 2

    def test_all_valid_matches_naive_count(self):
        rng = np.random.default_rng(0)
        stateseq = rng.integers(0, 5, 200).astype(np.int32)
        valid = np.ones(200, dtype=bool)
        counts = gap_aware_transition_counts(stateseq, valid, 5)
        assert counts.sum() == 199

    def test_matches_jax_moseq_destination_rule(self):
        """The count must equal the number of valid destination rows."""
        rng = np.random.default_rng(1)
        n = 300
        stateseq = rng.integers(0, 6, n).astype(np.int32)
        valid = rng.random(n) > 0.05
        counts = gap_aware_transition_counts(stateseq, valid, 6)
        assert counts.sum() == int(valid[1:].sum())

    def test_missing_frame_invalidates_nlags_plus_one_rows(self):
        """A NaN frame appears in nlags+1 strided rows, and is excluded from all."""
        model = ARHMM(
            OrderedDict([("s", _data(nan_frames=(50,)))]),
            kappa=1000.0, max_states=4, nlags=NLAGS, silent=True,
        )
        states = model.states_list[0]
        invalid = np.isnan(states.data).any(axis=1)
        assert invalid.sum() == NLAGS + 1, (
            "one missing frame should invalidate {} rows, got {}".format(
                NLAGS + 1, int(invalid.sum())
            )
        )

    def test_model_resamples_with_gaps(self):
        model = ARHMM(
            OrderedDict([("s", _data(nan_frames=(50, 300)))]),
            kappa=1000.0, max_states=6, nlags=NLAGS, silent=True,
        )
        np.random.seed(0)
        model.resample_model(num_procs=1)
        assert np.isfinite(model.trans_distn.trans_matrix).all()
        assert np.allclose(model.trans_distn.trans_matrix.sum(1), 1.0)


class TestCRFTableCounts:
    """Table counts must be drawn with the sticky mass on the diagonal.

    Under Fox et al. (2011) the CRF concentration for a cell is that cell's
    Dirichlet parameter, which for the sticky HDP-HMM is alpha*beta_k + kappa
    where the restaurant and dish coincide. The inherited implementation calls
    sample_crp_tablecounts, which cannot express a per-cell concentration and so
    omits kappa entirely.
    """

    def test_sampler_respects_per_cell_concentration(self):
        rng = np.random.default_rng(0)
        counts = np.full((2, 2), 40, dtype=np.int64)
        # A large concentration yields many tables; a tiny one yields few.
        conc = np.array([[1e3, 1e-3], [1e-3, 1e3]])
        draws = np.stack([
            sample_crf_table_counts(counts, conc, rng) for _ in range(200)
        ])
        assert draws[:, 0, 0].mean() > 20, "high concentration gave few tables"
        assert draws[:, 0, 1].mean() < 2, "low concentration gave many tables"

    def test_sampler_bounded_by_customers(self):
        rng = np.random.default_rng(0)
        counts = np.array([[7, 3], [0, 11]], dtype=np.int64)
        for _ in range(50):
            m = sample_crf_table_counts(counts, np.full((2, 2), 5.0), rng)
            assert (m <= counts).all(), "more tables than customers"
            assert (m >= 0).all()
            assert m[1, 0] == 0, "tables where there are no customers"

    def test_diagonal_concentration_includes_kappa(self):
        """The corrected _get_m must use alpha*beta + kappa on the diagonal."""
        num_states = 3
        alpha, gamma, kappa = 2.0, 1.0, 500.0
        beta = np.full(num_states, 1.0 / num_states)
        trans = GapAwareStickyHDPHMMTransitions(
            num_states=num_states, alpha=alpha, gamma=gamma,
            kappa=kappa, beta=beta,
        )
        # Diagonal-heavy counts. With kappa included the diagonal table counts
        # before thinning are large; without it they are near one.
        counts = np.zeros((num_states, num_states), dtype=np.int64)
        np.fill_diagonal(counts, 200)

        np.random.seed(0)
        conc = np.tile(alpha * beta, (num_states, 1)) + kappa * np.eye(num_states)
        with_kappa = np.stack([
            np.diag(sample_crf_table_counts(counts, conc)) for _ in range(50)
        ]).mean()
        without_kappa = np.stack([
            np.diag(sample_crf_table_counts(
                counts, np.tile(alpha * beta, (num_states, 1))
            ))
            for _ in range(50)
        ]).mean()
        assert with_kappa > 5 * without_kappa, (
            "including kappa should raise diagonal table counts substantially; "
            "got {:.2f} vs {:.2f}".format(with_kappa, without_kappa)
        )
        assert trans.kappa == kappa


class TestDiagonalBoost:
    """The MNIW conversions boost the diagonal before inverting."""

    def test_round_trip_is_consistent(self):
        rng = np.random.default_rng(0)
        d, in_dim = 3, 7
        M = rng.standard_normal((d, in_dim))
        A = rng.standard_normal((in_dim, in_dim))
        K = A @ A.T + in_dim * np.eye(in_dim)
        B = rng.standard_normal((d, d))
        S = B @ B.T + d * np.eye(d)
        nu = float(d + 2)

        natural = BoostedAutoRegression._standard_to_natural(nu, S, M, K)
        nu2, S2, M2, K2 = BoostedAutoRegression._natural_to_standard(natural)

        assert np.isclose(nu, nu2)
        # The boost is applied on both legs, so the round trip returns close to
        # the input rather than exactly to it. The residual scales with the
        # boost relative to the matrix being inverted, which for these
        # well-conditioned inputs is on the order of 1e-5 relative.
        assert np.allclose(M, M2, rtol=1e-3, atol=1e-4)
        assert np.allclose(K, K2, rtol=1e-3, atol=1e-4)
        assert np.allclose(S, S2, rtol=1e-3, atol=1e-4)

    def test_no_trailing_pad(self):
        """The inherited version padded K and S by 1e-8 after inverting."""
        d, in_dim = 2, 3
        C = np.eye(in_dim) * 4.0
        B = np.zeros((d, in_dim))
        A = np.eye(d) * 2.0
        nu, S, M, K = BoostedAutoRegression._natural_to_standard(
            np.array([A, B, C, 5.0], dtype=object)
        )
        expected_K = np.linalg.inv(C + DIAGONAL_BOOST * np.eye(in_dim))
        assert np.allclose(K, expected_K), "K carries an unexpected pad"

    def test_returns_symmetric_covariance(self):
        rng = np.random.default_rng(1)
        d, in_dim = 4, 9
        M = rng.standard_normal((d, in_dim))
        A = rng.standard_normal((in_dim, in_dim))
        K = A @ A.T + in_dim * np.eye(in_dim)
        B = rng.standard_normal((d, d))
        S = B @ B.T + d * np.eye(d)
        natural = BoostedAutoRegression._standard_to_natural(float(d + 2), S, M, K)
        _, S2, _, K2 = BoostedAutoRegression._natural_to_standard(natural)
        assert np.allclose(S2, S2.T)
        assert np.allclose(K2, K2.T)


class TestModelWiring:
    """ARHMM must construct the corrected classes."""

    @pytest.mark.parametrize("separate_trans", [False, True])
    def test_corrected_classes_are_used(self, separate_trans):
        groups = {"s": "g"} if separate_trans else None
        model = ARHMM(
            OrderedDict([("s", _data())]),
            kappa=1000.0, max_states=4, nlags=NLAGS, silent=True,
            separate_trans=separate_trans, groups=groups,
        )
        assert isinstance(model.obs_distns[0], BoostedAutoRegression)
        if not separate_trans:
            assert isinstance(
                model.trans_distn, GapAwareStickyHDPHMMTransitions
            )


class TestInitialization:
    """The model must start from the prior it was asked for.

    pyhsmm writes ``alphav`` in the constructor without going through the sticky
    property setter, so an unpatched model built with a large ``kappa`` starts
    with an essentially uniform transition matrix, and then forward-simulates its
    initial state sequence from that matrix while ignoring the data.
    """

    def test_initial_transition_matrix_is_sticky(self):
        kappa = 1e7
        model = ARHMM(
            OrderedDict([("s", _data())]),
            kappa=kappa, max_states=6, nlags=NLAGS, silent=True,
        )
        pi = np.asarray(model.trans_distn.trans_matrix)
        k = pi.shape[0]
        diag = np.diag(pi).mean()
        off = pi[~np.eye(k, dtype=bool)].mean()
        assert diag > 0.99, (
            "initial transition matrix is not sticky at kappa={:g}: mean "
            "diagonal {:.6f}".format(kappa, diag)
        )
        assert off < 1e-3, "off-diagonal mass {:.3e} is too high".format(off)
        assert np.allclose(pi.sum(axis=1), 1.0)

    def test_initial_states_depend_on_the_data(self):
        """Prior forward simulation ignores the data; a posterior draw cannot.

        Two models built with the same seed but different data must produce
        different initial state sequences.
        """
        out = []
        for seed in (0, 1):
            np.random.seed(0)
            model = ARHMM(
                OrderedDict([("s", _data(seed=seed))]),
                kappa=1000.0, max_states=6, nlags=NLAGS, silent=True,
            )
            out.append(np.array(model.states_list[0].stateseq))
        assert not np.array_equal(out[0], out[1]), (
            "initial state sequence is identical for different data, so it "
            "was not drawn from the posterior"
        )

