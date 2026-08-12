import numpy as np
import pytest

from kete import (
    State,
    Time,
    UncertainState,
    Vector,
    constants,
    moid,
    propagate_n_body,
    propagate_two_body,
    spice,
)
from kete.propagation import NonGravModel


@pytest.fixture(scope="session")
def ceres_traj():
    """
    Ceres 2 states 30 days apart.

    This was built from Ceres position data.
    """
    states = [
        State(
            jd=2462583.0,
            desig="Ceres",
            pos=(2.908844981531, -0.007973772593, -0.535807560572),
            vel=(-0.00028988685, 0.009639190713, 0.000360810151),
        ),
        State(
            jd=2462613.0,
            desig="Ceres",
            pos=(2.885148848337, 0.280747114789, -0.522235845766),
            vel=(-0.0012905891515, 0.009592164483, 0.000543572547),
        ),
        State(
            jd=2462883.0,
            desig="Ceres",
            pos=(1.396375741080, 2.3758755439074, -0.1812958200026),
            vel=(-0.00912714993504, 0.0045513730975, 0.00182565391195),
        ),
    ]
    return states


class TestNBodyPropagation:
    def test_propagation_short(self, ceres_traj):
        initial_state = ceres_traj[0]
        final_state = ceres_traj[1]
        calc = propagate_n_body(initial_state, Time(final_state.jd))
        assert np.allclose(calc.pos, final_state.pos)
        assert np.allclose(calc.vel, final_state.vel)

    def test_propagation_long(self, ceres_traj):
        initial_state = ceres_traj[0]
        final_state = ceres_traj[2]
        calc = propagate_n_body([initial_state], final_state.jd)[0]
        assert np.allclose(calc.pos, final_state.pos)
        assert np.allclose(calc.vel, final_state.vel)

    def test_a_terms(self):
        line = State(
            jd=2462583.0,
            desig="Line",
            pos=(10.0, 0.0, 0.0),
            vel=(0.0, 0.01, 0.0),
        )
        model = NonGravModel.new_dust(1.0)
        calc = propagate_n_body([line], line.jd + 10, non_gravs=[model])[0]
        assert np.allclose(calc.pos, line.pos + line.vel * 10, atol=1e-5)


class TestTwoBodyPropagation:
    def test_propagation_single(self):
        """
        Propagate Venus using StateVector and compare the result to using spice.
        This is only over 5 days the 2 body approximation is accurate over that time.
        """
        state = spice.get_state("Venus", 2461161.5)

        for jd in range(-5, 5):
            jd = state.jd + jd
            vec_state = propagate_two_body(state, jd)
            jpl_state = spice.get_state("Venus", jd)
            assert vec_state.jd == jpl_state.jd
            assert np.allclose(vec_state.vel, jpl_state.vel)
            assert np.allclose(vec_state.pos, jpl_state.pos)

    def test_propagation_light_delay(self):
        """
        Propagate Venus using StateVector and compare the result to using spice.
        This is only over 5 days the 2 body approximation is accurate over that time.

        Place an observer X AU away from Venus and ensure that the delay is correct.
        """
        state = spice.get_state("Venus", 2461161.5)

        for au in range(0, 5):
            sun2obs = Vector(state.pos + [au, 0.0, 0.0])
            delay = au / constants.SPEED_OF_LIGHT_AUDAY
            should_be = propagate_two_body(state, state.jd - delay)
            calculated = propagate_two_body(state, state.jd, sun2obs)

            assert np.allclose(calculated.vel, should_be.vel)
            assert np.allclose(calculated.pos, should_be.pos)


class TestFreeParameters:
    """Force parameters left free (NaN) occupy a covariance row of their own."""

    @pytest.fixture
    def state(self):
        return State(
            jd=2460000.5,
            desig="dust",
            pos=(2.0, 0.3, 0.1),
            vel=(-0.002, 0.011, 0.001),
        )

    def test_free_beta_extends_the_covariance(self, state):
        free = NonGravModel.new_dust(beta=float("nan"))
        us = UncertainState.from_state(
            state, 1e-8, 1e-7, non_grav=free,
            free_params=[0.01], param_sigmas=[0.003],
        )
        assert us.param_names == ["p", "f", "g", "h", "k", "L", "beta"]
        cov = np.array(us.cov_matrix)
        assert cov.shape == (7, 7)
        assert np.isclose(cov[6, 6], 0.003**2)
        # The central value is what was asked for, not the zero default.
        assert np.isclose(us.non_grav.beta, 0.01)

    def test_free_params_defaults_to_zero(self, state):
        """The documented default, which is why supplying it matters: a dust
        beta of zero is a grain feeling no radiation pressure."""
        free = NonGravModel.new_dust(beta=float("nan"))
        us = UncertainState.from_state(state, 1e-8, 1e-7, non_grav=free)
        assert us.non_grav.beta == 0.0

    def test_frozen_parameters_take_no_row(self, state):
        """A concrete beta is frozen, not fitted, so the covariance stays 6x6
        and free_params is rejected."""
        fixed = NonGravModel.new_dust(beta=0.01)
        us = UncertainState.from_state(state, 1e-8, 1e-7, non_grav=fixed)
        assert us.param_names == ["p", "f", "g", "h", "k", "L"]
        assert np.array(us.cov_matrix).shape == (6, 6)
        with pytest.raises(ValueError, match="free_params has length"):
            UncertainState.from_state(
                state, 1e-8, 1e-7, non_grav=fixed, free_params=[0.01]
            )

    def test_propagation_builds_state_parameter_covariance(self, state):
        """The point of the augmented row: the STM carries d(state)/d(beta),
        so an initially diagonal covariance develops element-beta
        correlations, and beta itself is preserved (it is constant along a
        trajectory)."""
        free = NonGravModel.new_dust(beta=float("nan"))
        us = UncertainState.from_state(
            state, 1e-8, 1e-7, non_grav=free,
            free_params=[0.01], param_sigmas=[0.003],
        )
        assert np.allclose(np.array(us.cov_matrix)[6, :6], 0.0)
        prop = us.propagate(state.jd + 60.0)
        cov = np.array(prop.cov_matrix)
        assert np.isclose(cov[6, 6], 0.003**2, rtol=1e-9)
        assert not np.allclose(cov[6, :6], 0.0)

    def test_sampling_draws_parameters_with_states(self, state):
        """Samples carry their own beta, jointly with the orbit."""
        free = NonGravModel.new_dust(beta=float("nan"))
        us = UncertainState.from_state(
            state, 1e-8, 1e-7, non_grav=free,
            free_params=[0.02], param_sigmas=[0.004],
        )
        _, non_gravs = us.sample(4000, seed=11)
        betas = np.array([n.beta for n in non_gravs])
        assert np.isclose(betas.mean(), 0.02, atol=5e-4)
        assert np.isclose(betas.std(), 0.004, rtol=0.1)

    def test_from_cartesian_accepts_the_augmented_matrix(self, state):
        free = NonGravModel.new_dust(beta=float("nan"))
        cov = np.diag([1e-16] * 3 + [1e-20] * 3 + [9e-6])
        us = UncertainState.from_cartesian(
            state, cov.tolist(), non_grav=free, free_params=[0.01]
        )
        back = np.array(us.cartesian_cov_matrix)
        assert back.shape == (7, 7)
        assert np.isclose(back[6, 6], 9e-6)

    def test_free_params_validation(self, state):
        free = NonGravModel.new_dust(beta=float("nan"))
        with pytest.raises(ValueError, match="free_params has length"):
            UncertainState.from_state(state, 1e-8, 1e-7, non_grav=free,
                                      free_params=[0.01, 0.02])
        with pytest.raises(ValueError, match="finite"):
            UncertainState.from_state(state, 1e-8, 1e-7, non_grav=free,
                                      free_params=[float("nan")])
        with pytest.raises(ValueError, match="param_sigmas"):
            UncertainState.from_state(state, 1e-8, 1e-7, non_grav=free,
                                      free_params=[0.01], param_sigmas=[1.0, 2.0])


class TestDustBeta:
    """The beta <-> diameter conversion pair."""

    @pytest.mark.parametrize("diameter", [1e-7, 1.19e-6, 1e-5, 1e-3, 0.1])
    def test_diameter_round_trips_through_beta(self, diameter):
        """``diameter`` inverts ``new_dust(diameter=...)``, so the two must
        share their defaults - they disagreed by a factor of 1000 once."""
        model = NonGravModel.new_dust(diameter=diameter)
        assert np.isclose(model.diameter(), diameter, rtol=1e-12)

    @pytest.mark.parametrize("density", [500.0, 1000.0, 3000.0])
    @pytest.mark.parametrize("q_pr", [0.5, 1.0, 2.0])
    def test_round_trip_with_explicit_coefficients(self, density, q_pr):
        """The same holds for any coefficients, as long as both calls agree -
        beta alone is stored, so the conversion inputs are not recovered."""
        model = NonGravModel.new_dust(diameter=2e-6, density=density, q_pr=q_pr)
        assert np.isclose(model.diameter(density=density, q_pr=q_pr), 2e-6, rtol=1e-12)

    def test_beta_one_is_near_micron_scale(self):
        """The Burns, Lamy & Soter scaling puts beta = 1 at ~1.2 um for a
        1000 kg/m^3 grain - the anchor the default c_pr encodes."""
        assert np.isclose(NonGravModel.new_dust(beta=1.0).diameter(), 1.19e-6)

    def test_diameter_is_nan_for_non_dust_models(self):
        assert np.isnan(NonGravModel.new_comet(1e-8, 1e-9, 0.0).diameter())


@pytest.mark.parametrize("planet", [(None, 1.58), ("Earth", 1.58), ("Mercury", 2.18)])
def test_moid(planet, ceres_traj):
    planet, ceres_moid = planet
    if planet is None:
        state = None
        vs = spice.get_state("Earth", 2461161.5)
    else:
        state = spice.get_state(planet, 2461161.5)
        vs = state

    assert np.isclose(moid(vs, state), 0)

    ceres = ceres_traj[0]
    assert np.isclose(moid(ceres, state), ceres_moid, atol=1e-2)
