import numpy as np
import pandas as pd
import pytest

import prototype_learner as pl


def _proto_components():
    return pl.make_components(retrieve_fn=pl.retrieve_none, store_fn=pl.store_mu_only)


def test_encode_characters_maps_in_first_seen_order():
    s = pd.Series(["b", "a", "b", "c", "a"])
    codes, char_to_idx, idx_to_char = pl.encode_characters(s)

    np.testing.assert_array_equal(codes, np.array([0, 1, 0, 2, 1]))
    assert char_to_idx == {"b": 0, "a": 1, "c": 2}
    assert idx_to_char == {0: "b", 1: "a", 2: "c"}


def test_make_episode_df_normalization_and_cumulative_sums():
    choices = np.array([[3.0, 4.0], [0.0, 0.0], [-3.0, 4.0]])
    chars = np.array([10, 10, 20])

    df = pl.make_episode_df(choices, chars, normalize=True)

    assert list(df["decision_num"]) == [1, 2, 1]
    np.testing.assert_allclose(
        df.loc[0, ["choice_affil_unit", "choice_power_unit"]].to_numpy(float),
        np.array([0.6, 0.8]),
        atol=1e-12,
    )
    assert np.isnan(df.loc[1, "choice_affil_unit"])
    assert np.isnan(df.loc[1, "choice_power_unit"])
    np.testing.assert_allclose(
        df.loc[2, ["choice_affil_unit", "choice_power_unit"]].to_numpy(float),
        np.array([-0.6, 0.8]),
        atol=1e-12,
    )
    np.testing.assert_array_equal(df["pos_affil"].to_numpy(), np.array([3.0, 3.0, -3.0]))
    np.testing.assert_array_equal(df["pos_power"].to_numpy(), np.array([4.0, 4.0, 4.0]))


def test_make_episode_df_validates_input_shapes():
    with pytest.raises(ValueError, match="shape"):
        pl.make_episode_df(np.array([1.0, 2.0]), np.array([0]))

    with pytest.raises(ValueError, match="length"):
        pl.make_episode_df(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0]))


def test_init_proto_state_initializes_shapes_and_types():
    state = pl.init_proto_state(3)
    assert state["n_char"] == 3
    assert state["mu"].shape == (3, 2)
    assert state["n"].shape == (3,)
    assert state["mu"].dtype == float
    assert state["n"].dtype == int
    assert state["mem_vecs"] == []
    assert state["mem_chars"] == []


def test_lr_running_mean_exact_and_leaky_modes():
    state = pl.init_proto_state(2)

    lr1 = pl.lr_running_mean(state, 0, alpha=None)
    lr2 = pl.lr_running_mean(state, 0, alpha=None)
    lr3 = pl.lr_running_mean(state, 1, alpha=0.2)

    assert lr1 == pytest.approx(1.0)
    assert lr2 == pytest.approx(0.5)
    assert lr3 == pytest.approx(0.2)
    np.testing.assert_array_equal(state["n"], np.array([2, 1]))

    with pytest.raises(ValueError, match="alpha must be in"):
        pl.lr_running_mean(state, 0, alpha=0.0)


def test_delta_rule_update_returns_expected_vector_and_info():
    m_new, info = pl.delta_rule_update(np.array([0.0, 0.0]), np.array([1.0, -1.0]), 0.25)

    np.testing.assert_allclose(m_new, np.array([0.25, -0.25]), atol=1e-12)
    assert info["affil_delta"] == pytest.approx(1.0)
    assert info["power_delta"] == pytest.approx(-1.0)
    assert info["affil_update"] == pytest.approx(0.25)
    assert info["power_update"] == pytest.approx(-0.25)
    assert info["update_magnitude"] == pytest.approx(np.sqrt(0.125))
    assert info["surprise"] == pytest.approx(2.0)


def test_retrieve_none_returns_current_prototype_and_default_info():
    state = pl.init_proto_state(1)
    state["mu"][0] = np.array([0.3, -0.4])
    state["mem_vecs"] = [np.array([1.0, 0.0])]
    state["mem_chars"] = [0]

    retrieved, info = pl.retrieve_none(state, 0)

    np.testing.assert_allclose(retrieved, np.array([0.3, -0.4]))
    assert info["retrieved_affil"] == pytest.approx(0.3)
    assert info["retrieved_power"] == pytest.approx(-0.4)
    assert info["mem_n"] == 1
    assert np.isnan(info["retrieval_entropy"])
    assert np.isnan(info["retrieval_neff"])
    assert np.isnan(info["other_mass"])


def test_safe_softmax_normalizes_and_has_nan_fallback():
    w = pl._safe_softmax(np.array([2.0, 0.0]))
    assert w.shape == (2,)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] > w[1]

    w_bad = pl._safe_softmax(np.array([np.nan, np.nan]))
    np.testing.assert_allclose(w_bad, np.array([0.5, 0.5]), atol=1e-12)


def test_retrieve_softmax_episode_with_empty_memory_returns_mu():
    state = pl.init_proto_state(1)
    state["mu"][0] = np.array([0.7, 0.1])

    retrieved, info = pl.retrieve_softmax_episode(state, 0, kappa=3.0)

    np.testing.assert_allclose(retrieved, np.array([0.7, 0.1]), atol=1e-12)
    assert info["mem_n"] == 0
    assert info["retrieval_entropy"] == pytest.approx(0.0)
    assert info["retrieval_neff"] == pytest.approx(0.0)
    assert info["other_mass"] == pytest.approx(0.0)


def test_retrieve_softmax_episode_applies_same_character_bias():
    state = pl.init_proto_state(2)
    state["mu"][0] = np.array([1.0, 0.0])
    state["mem_vecs"] = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    state["mem_chars"] = [0, 1]

    retrieved, info = pl.retrieve_softmax_episode(
        state,
        0,
        kappa=0.0,
        same_char_bias=2.0,
    )

    np.testing.assert_allclose(retrieved, np.array([1.0, 0.0]), atol=1e-12)
    expected_other = 1.0 / (np.exp(2.0) + 1.0)
    assert info["other_mass"] == pytest.approx(expected_other, rel=1e-6)
    assert info["mem_n"] == 2


def test_store_mu_only_updates_mu_without_touching_memory():
    state = pl.init_proto_state(1)
    state["mem_vecs"] = [np.array([1.0, 0.0])]
    state["mem_chars"] = [0]

    out = pl.store_mu_only(state, 0, np.array([0.2, 0.9]))

    assert out is state
    np.testing.assert_allclose(state["mu"][0], np.array([0.2, 0.9]), atol=1e-12)
    assert len(state["mem_vecs"]) == 1
    assert state["mem_chars"] == [0]


def test_store_mu_and_episode_appends_copy_of_new_mean():
    state = pl.init_proto_state(1)
    m = np.array([0.5, -0.1])

    out = pl.store_mu_and_episode(state, 0, m)
    m[:] = 99.0

    assert out is state
    np.testing.assert_allclose(state["mu"][0], np.array([0.5, -0.1]), atol=1e-12)
    np.testing.assert_allclose(state["mem_vecs"][0], np.array([0.5, -0.1]), atol=1e-12)
    assert state["mem_chars"] == [0]


def test_resolve_components_validates_argument_combinations():
    components = _proto_components()
    resolved = pl._resolve_components(
        components=components,
        retrieve_fn=None,
        store_fn=None,
        lr_fn=None,
        update_fn=None,
    )
    assert resolved is components

    with pytest.raises(ValueError, match="either `components` or individual functions"):
        pl._resolve_components(
            components=components,
            retrieve_fn=pl.retrieve_none,
            store_fn=None,
            lr_fn=None,
            update_fn=None,
        )

    with pytest.raises(ValueError, match="Missing required functions"):
        pl._resolve_components(
            components=None,
            retrieve_fn=None,
            store_fn=None,
            lr_fn=None,
            update_fn=None,
        )


def test_fill_retrieval_fields_adds_missing_without_overwriting():
    out = {"retrieved_affil": 1.2, "retrieval_entropy": 0.7}
    filled = pl.fill_retrieval_fields(out)

    assert filled is out
    assert out["retrieved_affil"] == 1.2
    assert out["retrieval_entropy"] == 0.7
    assert "retrieval_neff" in out
    assert "other_mass" in out
    assert "mem_n" in out


def test_online_step_non_finite_input_keeps_state_unchanged():
    state = pl.init_proto_state(1)
    state["mu"][0] = np.array([0.2, -0.1])
    n_before = state["n"].copy()
    mu_before = state["mu"].copy()

    state2, out = pl.online_step(
        state=state,
        char_idx=0,
        x=np.array([np.nan, 1.0]),
        components=_proto_components(),
    )

    assert state2 is state
    np.testing.assert_allclose(state["mu"], mu_before, atol=1e-12)
    np.testing.assert_array_equal(state["n"], n_before)
    assert out["update_magnitude"] == pytest.approx(0.0)
    assert np.isnan(out["alpha"])
    assert np.isnan(out["surprise"])


def test_online_step_updates_state_with_running_mean():
    state = pl.init_proto_state(1)
    components = _proto_components()

    state, out1 = pl.online_step(
        state=state,
        char_idx=0,
        x=np.array([1.0, 0.0]),
        components=components,
        alpha=None,
    )
    state, out2 = pl.online_step(
        state=state,
        char_idx=0,
        x=np.array([0.0, 1.0]),
        components=components,
        alpha=None,
    )

    np.testing.assert_allclose(state["mu"][0], np.array([0.5, 0.5]), atol=1e-12)
    np.testing.assert_array_equal(state["n"], np.array([2]))
    assert out1["alpha"] == pytest.approx(1.0)
    assert out2["alpha"] == pytest.approx(0.5)


def test_online_step_requires_components_or_required_functions():
    state = pl.init_proto_state(1)
    with pytest.raises(ValueError, match="Missing required functions"):
        pl.online_step(
            state=state,
            char_idx=0,
            x=np.array([1.0, 0.0]),
            retrieve_fn=pl.retrieve_none,
        )


def test_run_online_model_functional_sorts_trials_and_returns_outputs():
    episodes = pd.DataFrame(
        {
            "global_t": [2, 1, 3],
            "character_role_num": [1, 1, 2],
            "choice_affil_unit": [1.0, 0.0, 0.0],
            "choice_power_unit": [0.0, 1.0, 1.0],
        }
    )

    df, state = pl.run_online_model_functional(
        episodes,
        components=_proto_components(),
        alpha=None,
    )

    assert list(df["global_t"]) == [1, 2, 3]
    assert {"affil_mean", "power_mean", "retrieved_affil", "retrieval_entropy"}.issubset(
        set(df.columns)
    )
    np.testing.assert_array_equal(state["n"], np.array([2, 1]))


def test_run_online_model_functional_validates_x_columns():
    episodes = pd.DataFrame({"global_t": [1], "character_role_num": [0], "x": [1.0]})
    with pytest.raises(ValueError, match="missing x_cols"):
        pl.run_online_model_functional(episodes, components=_proto_components())


def test_linreg_slope_handles_regular_and_degenerate_inputs():
    assert pl.linreg_slope([0, 1, 2, 3], [1, 3, 5, 7]) == pytest.approx(2.0)
    assert np.isnan(pl.linreg_slope([1, 1, 1], [2, 3, 4]))
    assert np.isnan(pl.linreg_slope([1, 2], [3, 4]))


def test_pearson_r_handles_regular_and_short_inputs():
    assert pl.pearson_r([0, 1, 2, 3], [0, 2, 4, 6]) == pytest.approx(1.0)
    assert np.isnan(pl.pearson_r([1, 2], [3, 4]))


def test_distance_matrix_from_mu_is_pairwise_euclidean():
    state = {"mu": np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])}
    d = pl.distance_matrix_from_mu(state)

    np.testing.assert_allclose(
        d,
        np.array([[0.0, 5.0, 3.0], [5.0, 0.0, 4.0], [3.0, 4.0, 0.0]]),
        atol=1e-12,
    )


def test_retrieval_weights_all_and_same_restrictions():
    state = pl.init_proto_state(2)
    state["mu"][0] = np.array([1.0, 0.0])
    state["mu"][1] = np.array([0.0, 1.0])
    state["mem_vecs"] = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    state["mem_chars"] = [0, 1, 0]

    w_all, idx_all = pl.retrieval_weights(state, cue_char=0, kappa=2.0, restrict="all")
    assert idx_all is not None
    assert w_all is not None
    assert len(idx_all) == 3
    assert np.isclose(w_all.sum(), 1.0)

    w_same, idx_same = pl.retrieval_weights(state, cue_char=0, kappa=2.0, restrict="same")
    assert idx_same is not None
    assert w_same is not None
    mem_chars = np.asarray(state["mem_chars"], int)
    assert np.all(mem_chars[idx_same] == 0)
    assert np.isclose(w_same.sum(), 1.0)


def test_retrieval_weights_returns_none_when_no_matching_same_char():
    state = pl.init_proto_state(2)
    state["mu"][0] = np.array([1.0, 0.0])
    state["mu"][1] = np.array([0.0, 1.0])
    state["mem_vecs"] = [np.array([1.0, 0.0])]
    state["mem_chars"] = [0]

    w, idx = pl.retrieval_weights(state, cue_char=1, kappa=2.0, restrict="same")
    assert w is None
    assert idx is None


def test_predicted_confusability_mass_has_rows_that_sum_to_one_when_memory_exists():
    state = pl.init_proto_state(2)
    state["mu"][0] = np.array([1.0, 0.0])
    state["mu"][1] = np.array([0.0, 1.0])
    state["mem_vecs"] = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    state["mem_chars"] = [0, 1, 0]

    m = pl.predicted_confusability_mass(state, kappa=2.0, restrict="all")
    assert m.shape == (2, 2)
    np.testing.assert_allclose(m.sum(axis=1), np.ones(2), atol=1e-12)


def test_confusion_prob_offdiag_zeroes_diagonal_and_row_normalizes():
    m = np.array([[0.7, 0.3], [0.2, 0.8]])
    c = pl.confusion_prob_offdiag(m)

    np.testing.assert_allclose(c, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)


def test_confusability_vs_distance_slope_matches_manual_pairwise_computation(monkeypatch):
    fake_mass = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.5, 0.1],
            [0.3, 0.2, 0.5],
        ]
    )
    fake_dist = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]
    )

    monkeypatch.setattr(pl, "predicted_confusability_mass", lambda *_a, **_k: fake_mass)
    monkeypatch.setattr(pl, "distance_matrix_from_mu", lambda *_a, **_k: fake_dist)

    slope, r = pl.confusability_vs_distance_slope({"n_char": 3}, kappa=1.0)

    cprob = pl.confusion_prob_offdiag(fake_mass)
    xs = np.array([fake_dist[0, 1], fake_dist[0, 2], fake_dist[1, 2]], dtype=float)
    ys = np.array(
        [
            0.5 * (cprob[0, 1] + cprob[1, 0]),
            0.5 * (cprob[0, 2] + cprob[2, 0]),
            0.5 * (cprob[1, 2] + cprob[2, 1]),
        ],
        dtype=float,
    )
    assert slope == pytest.approx(pl.linreg_slope(xs, ys))
    assert r == pytest.approx(pl.pearson_r(xs, ys))


def test_rt_correlations_with_and_without_required_columns():
    df_ok = pd.DataFrame(
        {
            "reaction_time": [1.0, 2.0, 3.0],
            "retrieval_entropy": [0.0, 0.5, 1.0],
            "retrieval_neff": [3.0, 2.0, 1.0],
        }
    )
    r_ent, r_neff = pl.rt_correlations(df_ok, rt_col="reaction_time")
    assert r_ent == pytest.approx(1.0)
    assert r_neff == pytest.approx(-1.0)

    df_missing = pd.DataFrame({"reaction_time": [1.0, 2.0, 3.0]})
    r_ent2, r_neff2 = pl.rt_correlations(df_missing, rt_col="reaction_time")
    assert np.isnan(r_ent2)
    assert np.isnan(r_neff2)


def test_build_model_spec_sets_defaults_and_step_function_behavior():
    spec = pl.build_model_spec(name="proto", components=_proto_components(), alpha=0.25)
    assert spec["name"] == "proto"
    assert spec["alpha"] == 0.25
    assert callable(spec["retrieve_state_fn"])
    assert callable(spec["weights_fn"])

    state = spec["init_fn"](1)
    retrieved, info = spec["retrieve_state_fn"](state, 0)
    np.testing.assert_allclose(retrieved, np.array([0.0, 0.0]), atol=1e-12)
    assert set(info.keys()) == {"entropy", "n_eff", "other_mass"}

    state, out = spec["step_fn"](state, 0, np.array([1.0, 0.0]))
    assert out["alpha"] == pytest.approx(0.25)
    assert "retrieved_affil" in out
    assert "retrieval_entropy" in out

    w, idx = spec["weights_fn"](state, 0)
    assert w is None
    assert idx is None


def test_compose_model_spec_wraps_components_and_merges_readout_params():
    spec = pl.compose_model_spec(
        name="wrapped",
        retrieve_fn=pl.retrieve_none,
        store_fn=pl.store_mu_only,
        alpha=None,
        readout_params={"foo": 123},
    )

    assert spec["name"] == "wrapped"
    assert isinstance(spec["components"], pl.ModelComponents)
    assert spec["foo"] == 123


def test_make_model_spec_episodic_exposes_weights_fn_and_params():
    spec = pl.make_model_spec_episodic(alpha=0.5, kappa=3.0, same_char_bias=1.0, restrict="same")
    assert spec["name"] == "episodic"
    assert spec["kappa"] == pytest.approx(3.0)
    assert spec["same_char_bias"] == pytest.approx(1.0)
    assert spec["restrict"] == "same"

    state = spec["init_fn"](2)
    state, _ = spec["step_fn"](state, 0, np.array([1.0, 0.0]))
    w, idx = spec["weights_fn"](state, cue_char=0)
    assert w is not None
    assert idx is not None
    assert np.isclose(w.sum(), 1.0)
    mem_chars = np.asarray(state["mem_chars"], dtype=int)
    assert np.all(mem_chars[idx] == 0)


def test_make_model_spec_prototype_uses_non_episodic_defaults():
    spec = pl.make_model_spec_prototype(alpha=None)
    assert spec["name"] == "prototype"
    assert spec["kappa"] is None
    assert spec["same_char_bias"] is None
    assert spec["restrict"] == "all"

    state = spec["init_fn"](1)
    w, idx = spec["weights_fn"](state, cue_char=0)
    assert w is None
    assert idx is None


def test_fit_subject_raises_when_load_behavior_is_missing(monkeypatch):
    monkeypatch.delattr(pl, "load_behavior", raising=False)
    with pytest.raises(NameError, match="load_behavior"):
        pl.fit_subject("sub-1", model_spec=pl.make_model_spec_prototype())


def test_fit_subject_returns_none_triplet_for_empty_behavior(monkeypatch):
    monkeypatch.setattr(pl, "load_behavior", lambda *_a, **_k: pd.DataFrame(), raising=False)
    out = pl.fit_subject("sub-1", model_spec=pl.make_model_spec_prototype())
    assert out == (None, None, None)


def test_fit_subject_runs_with_char_role_num_fallback_and_reaction_time(monkeypatch):
    behavior = pd.DataFrame(
        {
            "char_role_num": [7, 7, 9],
            "affil_decision": [1.0, 0.0, 0.0],
            "power_decision": [0.0, 1.0, 1.0],
            "reaction_time": ["1.0", "2.5", "3.0"],
        }
    )

    monkeypatch.setattr(pl, "load_behavior", lambda *_a, **_k: behavior, raising=False)
    model_spec = {
        "name": "prototype",
        "components": _proto_components(),
        "alpha": 0.3,
        "rt_col": "reaction_time",
    }
    df, state, meta = pl.fit_subject("sub-7", model_spec=model_spec, normalize=True)

    assert df is not None
    assert state is not None
    assert meta == {"sub_id": "sub-7", "model": "prototype"}
    assert len(df) == 3
    assert "reaction_time" in df.columns
    np.testing.assert_allclose(df["reaction_time"].to_numpy(), np.array([1.0, 2.5, 3.0]))
    assert int(state["n"].sum()) == 3


def test_compute_readouts_sets_nan_conf_metrics_when_kappa_missing():
    df = pd.DataFrame(
        {
            "reaction_time": [1.0, 2.0, 3.0],
            "retrieval_entropy": [0.1, 0.2, 0.3],
            "retrieval_neff": [3.0, 2.0, 1.0],
        }
    )
    state = pl.init_proto_state(1)
    meta = {"sub_id": "s1", "model": "prototype"}
    out = pl.compute_readouts(df, state, meta, model_spec={"rt_col": "reaction_time", "kappa": None})

    assert out["n_trials"] == 3
    assert out["rt_r_entropy"] == pytest.approx(1.0)
    assert out["rt_r_neff"] == pytest.approx(-1.0)
    assert np.isnan(out["conf_slope"])
    assert np.isnan(out["conf_r"])


def test_compute_readouts_uses_confusability_when_kappa_present(monkeypatch):
    monkeypatch.setattr(pl, "confusability_vs_distance_slope", lambda *_a, **_k: (-0.2, -0.6))
    df = pd.DataFrame(
        {
            "reaction_time": [1.0, 2.0, 3.0],
            "retrieval_entropy": [0.1, 0.2, 0.3],
            "retrieval_neff": [3.0, 2.0, 1.0],
        }
    )
    state = pl.init_proto_state(1)
    meta = {"sub_id": "s2", "model": "episodic"}
    spec = {"rt_col": "reaction_time", "kappa": 8.0, "same_char_bias": 0.5, "restrict": "all"}

    out = pl.compute_readouts(df, state, meta, model_spec=spec)
    assert out["conf_slope"] == pytest.approx(-0.2)
    assert out["conf_r"] == pytest.approx(-0.6)


def test_summarize_subjects_skips_errors_and_sorts_rows(monkeypatch):
    def fake_fit_subject(sub_id, *, model_spec, **_kwargs):
        if sub_id == "err":
            raise RuntimeError("boom")
        if sub_id == "missing":
            return None, None, None
        return pd.DataFrame({"x": [1, 2]}), {"n_char": 1}, {"sub_id": sub_id, "model": model_spec["name"]}

    def fake_compute_readouts(df, state, meta, *, model_spec):
        return {
            "sub_id": meta["sub_id"],
            "model": meta["model"],
            "n_trials": len(df),
            "rt_r_entropy": np.nan,
            "rt_r_neff": np.nan,
            "conf_slope": 0.0,
            "conf_r": 0.0,
        }

    monkeypatch.setattr(pl, "fit_subject", fake_fit_subject)
    monkeypatch.setattr(pl, "compute_readouts", fake_compute_readouts)

    out = pl.summarize_subjects(
        ["b", "err", "a", "missing"],
        model_spec={"name": "m"},
        verbose=False,
        on_error="skip",
    )
    assert list(out["sub_id"]) == ["a", "b"]
    assert list(out["model"]) == ["m", "m"]


def test_summarize_subjects_can_raise_on_error(monkeypatch):
    def fake_fit_subject(sub_id, *, model_spec, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pl, "fit_subject", fake_fit_subject)
    with pytest.raises(RuntimeError, match="boom"):
        pl.summarize_subjects(["s1"], model_spec={"name": "m"}, verbose=False, on_error="raise")


def test_normalize_handles_zero_and_nonzero_vectors():
    np.testing.assert_allclose(pl.normalize(np.array([0.0, 0.0])), np.array([0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(pl.normalize(np.array([3.0, 4.0])), np.array([0.6, 0.8]), atol=1e-12)


def test_make_true_vectors_are_unit_circle_points():
    v = pl.make_true_vectors(4)
    assert v.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), np.ones(4), atol=1e-12)


def test_make_schedule_blocks_and_interleave_and_validation():
    rng = np.random.default_rng(0)
    block_sched = pl.make_schedule(7, 3, mode="blocks", block_len=2, rng=rng)
    np.testing.assert_array_equal(block_sched, np.array([0, 0, 1, 1, 2, 2, 0]))

    interleave_sched = pl.make_schedule(20, 3, mode="interleave", rng=np.random.default_rng(1))
    assert len(interleave_sched) == 20
    assert interleave_sched.min() >= 0
    assert interleave_sched.max() < 3

    with pytest.raises(ValueError, match="mode must be"):
        pl.make_schedule(5, 2, mode="bad")


def test_choice_continuous_returns_normalized_noisy_vector():
    out = pl.choice_continuous(np.array([3.0, 4.0]), noise_sd=0.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(out, np.array([0.6, 0.8]), atol=1e-12)


def test_choice_discrete_returns_one_of_allowed_directions():
    out = pl.choice_discrete(np.array([1.0, 0.0]), beta=20.0, n_dirs=8, rng=np.random.default_rng(0))
    dirs = np.c_[
        np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)),
        np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False)),
    ]
    assert any(np.allclose(out, d, atol=1e-12) for d in dirs)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-12)


def test_choice_assimilation_respects_lam_endpoints_without_noise():
    true_vec = np.array([1.0, 0.0])
    retrieved = np.array([0.0, 2.0])

    out0 = pl.choice_assimilation(true_vec, retrieved, lam=0.0, noise_sd=0.0, rng=np.random.default_rng(0))
    out1 = pl.choice_assimilation(true_vec, retrieved, lam=1.0, noise_sd=0.0, rng=np.random.default_rng(0))

    np.testing.assert_allclose(out0, pl.normalize(true_vec), atol=1e-12)
    np.testing.assert_allclose(out1, pl.normalize(retrieved), atol=1e-12)


def test_run_simulation_outputs_expected_columns_and_lengths():
    spec = pl.make_model_spec_prototype(alpha=None)
    df, base_true, state = pl.run_simulation(
        regime_name="demo",
        model_spec=spec,
        choice_mode="continuous",
        choice_kwargs={"noise_sd": 0.0},
        n_trials=12,
        n_char=2,
        seed=123,
    )

    assert len(df) == 12
    assert base_true.shape == (2, 2)
    assert state["mu"].shape == (2, 2)
    expected_cols = {"mu_affil", "mu_power", "retr_affil", "retr_power", "err_to_true", "retr_err_to_true"}
    assert expected_cols.issubset(df.columns)
    assert (df["regime"] == "demo").all()
    assert (df["model"] == "prototype").all()
    assert np.isfinite(df["err_to_true"]).all()


def test_run_simulation_validates_choice_mode():
    spec = pl.make_model_spec_prototype(alpha=None)
    with pytest.raises(ValueError, match="choice_mode must be"):
        pl.run_simulation(regime_name="x", model_spec=spec, choice_mode="bad", n_trials=3, n_char=1)


def test_run_simulation_switches_true_vector_after_t_switch():
    spec = pl.make_model_spec_prototype(alpha=None)
    df, base_true, _state = pl.run_simulation(
        regime_name="switch",
        model_spec=spec,
        choice_mode="continuous",
        choice_kwargs={"noise_sd": 0.0},
        n_trials=4,
        n_char=1,
        seed=0,
        t_switch=2,
        char_switch=0,
        new_vec=np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(
        df.loc[df["t"] < 2, ["true_affil", "true_power"]].to_numpy(),
        np.tile(base_true[0], (2, 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        df.loc[df["t"] >= 2, ["true_affil", "true_power"]].to_numpy(),
        np.tile(np.array([0.0, 1.0]), (2, 1)),
        atol=1e-12,
    )


def test_snapshot_attention_reports_no_weights_for_prototype_model(capsys):
    spec = pl.make_model_spec_prototype(alpha=None)
    pl.snapshot_attention(
        model_spec=spec,
        snapshot_t=0,
        cue_char=0,
        seed=0,
        n_char=2,
        n_trials=5,
        noise_sd=0.0,
    )
    captured = capsys.readouterr().out
    assert "Snapshot at t=0" in captured
    assert "No weights available for this model" in captured


def test_snapshot_attention_reports_out_of_range_snapshot(capsys):
    spec = pl.make_model_spec_prototype(alpha=None)
    pl.snapshot_attention(
        model_spec=spec,
        snapshot_t=999,
        cue_char=0,
        seed=0,
        n_char=2,
        n_trials=5,
        noise_sd=0.0,
    )
    captured = capsys.readouterr().out
    assert "No snapshot captured (snapshot_t out of range)." in captured
