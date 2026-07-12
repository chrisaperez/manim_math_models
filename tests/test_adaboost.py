"""Demo 9's math: the reweighting identities and the PS3.4 bound chain."""

import numpy as np

from core.sims import adaboost as ab

X, Y = ab.demo_data()
ROUNDS = ab.run(X, Y, T=30)   # training error first hits zero at round 14


def test_weak_learning_every_round():
    for rnd in ROUNDS:
        assert rnd.eps < 0.5


def test_weights_stay_normalized_and_move_the_right_way():
    for rnd in ROUNDS:
        assert abs(rnd.weights_after.sum() - 1.0) < 1e-12
        wrong = rnd.stump.predict(X) != Y
        # every misclassified point gains relative weight, every correct one loses
        ratio = rnd.weights_after / rnd.weights_before
        assert np.all(ratio[wrong] > 1.0)
        assert np.all(ratio[~wrong] < 1.0)


def test_yesterdays_stump_is_worthless_today():
    """Under the new weights, the stump just used has weighted error exactly 1/2."""
    for rnd in ROUNDS:
        wrong = rnd.stump.predict(X) != Y
        assert abs(rnd.weights_after[wrong].sum() - 0.5) < 1e-10


def test_z_equals_two_sqrt_eps_one_minus_eps():
    for rnd in ROUNDS:
        assert abs(rnd.z - 2 * np.sqrt(rnd.eps * (1 - rnd.eps))) < 1e-12


def test_bound_chain_holds_and_drives_error_to_zero():
    hist = ab.bound_history(X, Y, ROUNDS)
    for err, pz, eb in zip(hist["train_error"], hist["prod_z"],
                           hist["exp_bound"]):
        assert err <= pz + 1e-12          # train error <= prod Z_t
        assert pz <= eb + 1e-12           # prod Z_t <= exp(-2 sum gamma^2)
    assert hist["train_error"][-1] == 0.0  # separable demo data: driven to zero
    assert hist["prod_z"][-1] < hist["prod_z"][0]


def test_exponential_loss_equals_prod_z():
    """(1/n) sum exp(-y f(x)) telescopes to prod Z_t exactly."""
    hist = ab.bound_history(X, Y, ROUNDS)
    for t in range(1, len(ROUNDS) + 1):
        f = ab.committee_scores(X, ROUNDS[:t])
        exp_loss = float(np.mean(np.exp(-Y * f)))
        assert abs(exp_loss - hist["prod_z"][t - 1]) < 1e-10
