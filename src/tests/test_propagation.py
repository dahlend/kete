import pytest
import numpy as np
from kete import (
    constants,
    spice,
    propagate_two_body,
    propagate_n_body,
    moid,
    Vector,
    State,
    Time,
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
