import pytest

import kete


def _mixture(pos_sigma=1e-3, include_asteroids=False):
    """A one AU circular orbit wide enough that a 200 day arc splits."""
    state = kete.State(
        "test",
        2451545.0,
        kete.Vector([1.0, 0.0, 0.0]),
        kete.Vector([0.0, 0.0172020, 0.0]),
    )
    uncertain = kete.UncertainState.from_state(
        state, pos_sigma=pos_sigma, vel_sigma=1e-6
    )
    return kete.DiffuseState.from_uncertain(
        uncertain, include_asteroids=include_asteroids
    )


class TestDiffuseMarch:
    def test_unmarched_mixture_reports_nothing(self):
        mixture = _mixture()
        assert mixture.max_eta is None
        assert mixture.component_eta is None
        assert mixture.residual_meters is None
        assert mixture.weight_above_eta(0.1) is None
        assert mixture[0][1].eta is None

    def test_manual_march_matches_a_single_call(self):
        """The property carried probes exist for.

        Each component holds the probes measuring it, so a caller stepping leg by leg
        accumulates what the single call accumulates - same splits, same eta. Probes
        held by the propagation call instead would restart at every step, and this
        loop would silently split less than the one call.
        """
        mixture = _mixture()
        kwargs = {"split_threshold": 1e-4, "max_components": 27}

        single, _ = mixture.propagate(2451545.0 + 200.0, step_days=90.0, **kwargs)

        # The same grid a 90 day step cuts: the arc's remainder is the first leg.
        marched = mixture
        for offset in (20.0, 110.0, 200.0):
            marched, _ = marched.step(2451545.0 + offset, **kwargs)

        assert single.n_components > 1, "the arc has to split or this proves nothing"
        assert marched.n_components == single.n_components
        assert marched.max_eta == pytest.approx(single.max_eta, rel=1e-12)

    def test_step_reports_seeding_and_termination(self):
        mixture = _mixture(pos_sigma=1e-5)
        stepped, report = mixture.step(2451545.0 + 90.0, split_threshold=1.0)
        assert report.seeded == 1, "the input component had no probes yet"
        assert report.termination == "converged"

        # Continuing carries them; nothing is re-seeded.
        _, second = stepped.step(2451545.0 + 180.0, split_threshold=1.0)
        assert second.seeded == 0

    def test_rebuilding_a_component_restarts_the_measurement(self):
        """A mixture taken apart and put back together loses its carried probes.

        That restarts eta from zero, which is reported rather than left to be inferred.
        """
        mixture = _mixture(pos_sigma=1e-4)
        first, _ = mixture.step(2451545.0 + 90.0, split_threshold=1e6)

        rebuilt = kete.DiffuseState.new(list(first.weights), list(first.components))
        _, report = rebuilt.step(2451545.0 + 180.0, split_threshold=1e6)
        assert report.seeded == 1

    def test_force_model_is_fixed_on_the_mixture(self):
        mixture = _mixture(include_asteroids=True)
        assert mixture.include_asteroids is True
        stepped, _ = mixture.step(2451545.0 + 10.0, split_threshold=1.0)
        assert stepped.include_asteroids is True

        # It cannot be varied per call, which is what keeps every leg of a march under
        # the model the carried probes were integrated under.
        with pytest.raises(TypeError):
            mixture.step(2451545.0 + 20.0, include_asteroids=False)

    def test_component_eta_and_residual_are_paired(self):
        mixture = _mixture()
        marched, _ = mixture.propagate(
            2451545.0 + 200.0, split_threshold=1e-4, max_components=27
        )
        etas = marched.component_eta
        residuals = marched.component_residual_meters
        assert len(etas) == marched.n_components
        assert len(residuals) == marched.n_components
        assert marched.max_eta == pytest.approx(max(etas))
        assert marched.residual_meters == pytest.approx(max(residuals))
        # The same numbers are reachable from the components themselves.
        for (_, component), eta, residual in zip(marched, etas, residuals):
            assert component.eta == eta
            assert component.residual_meters == residual

    def test_a_non_finite_target_is_an_error(self):
        with pytest.raises(ValueError):
            _mixture().propagate(float("nan"))


class TestUncertainStateIO:
    """Saving and loading uncertain states.

    A stored component has to come back as the component that was stored, or a
    pipeline that propagates once and reads the result later is not propagating
    what it thinks it is.
    """

    def _marched(self):
        """A component carrying probes, eta, and a whitening reference."""
        marched, _ = _mixture().propagate(
            2451545.0 + 200.0, split_threshold=1e-4, max_components=27
        )
        return marched.components[0]

    def test_round_trip_keeps_the_measurement(self, tmp_path):
        before = self._marched()
        path = str(tmp_path / "one.kete")
        before.save(path)
        after = kete.UncertainState.load(path)

        assert after.epoch.jd == before.epoch.jd
        assert after.cov_matrix == before.cov_matrix
        assert after.param_names == before.param_names
        # The reported numbers survive, which means the probes did.
        assert after.eta == before.eta
        assert after.residual_meters == before.residual_meters

    def test_round_trip_keeps_the_force_model(self, tmp_path):
        state = kete.State(
            "dust",
            2451545.0,
            kete.Vector([1.5, 0.0, 0.0]),
            kete.Vector([0.0, 0.014, 0.0]),
        )
        uncertain = kete.UncertainState.from_state(
            state,
            pos_sigma=1e-6,
            vel_sigma=1e-9,
            non_grav=kete.propagation.NonGravModel.new_dust(beta=float("nan")),
            free_params=[0.01],
            param_sigmas=[0.002],
        )
        path = str(tmp_path / "dust.kete")
        uncertain.save(path)
        after = kete.UncertainState.load(path)

        # Without the model the free parameter value means nothing.
        assert after.non_grav.beta == pytest.approx(0.01)
        assert after.param_names == ["p", "f", "g", "h", "k", "L", "beta"]

    def test_many_states_in_one_file(self, tmp_path):
        marched, _ = _mixture().propagate(
            2451545.0 + 200.0, split_threshold=1e-4, max_components=27
        )
        states = marched.components
        assert len(states) > 1

        path = str(tmp_path / "many.kete")
        kete.UncertainState.save_list(states, path)
        after = kete.UncertainState.load_list(path)

        assert len(after) == len(states)
        for got, want in zip(after, states):
            assert got.eta == want.eta
            assert got.cov_matrix == want.cov_matrix

    def test_a_single_state_file_loads_as_a_list_of_one(self, tmp_path):
        path = str(tmp_path / "one.kete")
        self._marched().save(path)
        assert len(kete.UncertainState.load_list(path)) == 1

    def test_loading_the_wrong_type_is_an_error(self, tmp_path):
        path = str(tmp_path / "one.kete")
        self._marched().save(path)
        with pytest.raises(ValueError):
            kete.SimultaneousStates.load(path)


class TestDiffuseStateIO:
    """Saving and loading mixtures.

    Stage-2-style use: propagate once, write to disk, read it back later and
    render or keep marching.
    """

    def _marched(self):
        marched, _ = _mixture().propagate(
            2451545.0 + 200.0, split_threshold=1e-4, max_components=27
        )
        assert marched.n_components > 1
        return marched

    def test_round_trip_keeps_the_mixture(self, tmp_path):
        before = self._marched()
        path = str(tmp_path / "mix.kete")
        before.save(path)
        after = kete.DiffuseState.load(path)

        assert after.n_components == before.n_components
        assert after.weights == before.weights
        assert after.epoch.jd == before.epoch.jd
        assert after.component_eta == before.component_eta
        assert after.component_residual_meters == before.component_residual_meters
        assert after.include_asteroids == before.include_asteroids

    def test_a_reloaded_mixture_keeps_marching(self, tmp_path):
        """The probes survive, so continuing does not restart ``eta``.

        A reseed would be reported here, and would mean the stored product had
        silently lost the measurement behind its own numbers.
        """
        path = str(tmp_path / "mix.kete")
        self._marched().save(path)
        after = kete.DiffuseState.load(path)
        _, report = after.propagate(
            2451545.0 + 300.0, split_threshold=1e-4, max_components=27
        )
        assert report.seeded == 0

    def test_round_trip_keeps_the_force_model(self, tmp_path):
        state = kete.State(
            "dust",
            2451545.0,
            kete.Vector([1.5, 0.0, 0.0]),
            kete.Vector([0.0, 0.014, 0.0]),
        )
        uncertain = kete.UncertainState.from_state(
            state,
            pos_sigma=1e-6,
            vel_sigma=1e-9,
            non_grav=kete.propagation.NonGravModel.new_dust(beta=float("nan")),
            free_params=[0.01],
            param_sigmas=[0.002],
        )
        before = kete.DiffuseState.from_uncertain(uncertain, include_asteroids=True)
        path = str(tmp_path / "dust.kete")
        before.save(path)
        after = kete.DiffuseState.load(path)

        assert after.non_grav.beta == pytest.approx(0.01)
        assert after.include_asteroids is True
        assert after.param_names == ["p", "f", "g", "h", "k", "L", "beta"]

    def test_many_mixtures_in_one_file(self, tmp_path):
        mixtures = [self._marched(), _mixture()]
        path = str(tmp_path / "many.kete")
        kete.DiffuseState.save_list(mixtures, path)
        after = kete.DiffuseState.load_list(path)

        assert len(after) == 2
        assert after[0].n_components == mixtures[0].n_components
        assert after[1].max_eta is None

    def test_a_single_mixture_file_loads_as_a_list_of_one(self, tmp_path):
        path = str(tmp_path / "mix.kete")
        self._marched().save(path)
        assert len(kete.DiffuseState.load_list(path)) == 1

    def test_loading_the_wrong_type_is_an_error(self, tmp_path):
        path = str(tmp_path / "mix.kete")
        self._marched().save(path)
        with pytest.raises(ValueError):
            kete.UncertainState.load(path)
