import numpy as np
import pandas as pd

from get_choice_embeddings import (
    build_choice_embeddings,
    build_embedding_text_inputs,
    canonicalize_slides,
    clean_text,
    compose_context_text,
    find_project_root,
    generate_and_save_choice_embeddings_from_files,
    read_task_table,
    resolve_task_file,
    save_choice_embeddings,
)


def sample_snt_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_num": 1,
                "slide_type": "Narration",
                "decision_num": np.nan,
                "text": "  Intro   text ",
                "opt1_text": np.nan,
                "opt2_text": np.nan,
                "dimension": np.nan,
            },
            {
                "trial_num": 2,
                "slide_type": "Decision",
                "decision_num": 10,
                "text": np.nan,
                "opt1_text": "  Ask   for help ",
                "opt2_text": "Ignore",
                "dimension": "trust",
            },
            {
                "trial_num": 3,
                "slide_type": "Decision",
                "decision_num": 20,
                "text": np.nan,
                "opt1_text": "Run",
                "opt2_text": "Hide",
                "dimension": "safety",
            },
            {
                "trial_num": 4,
                "slide_type": "Image",
                "decision_num": np.nan,
                "text": "Filtered",
                "opt1_text": np.nan,
                "opt2_text": np.nan,
                "dimension": np.nan,
            },
            {
                "trial_num": 5,
                "slide_type": "Narration",
                "decision_num": np.nan,
                "text": " After   scene ",
                "opt1_text": np.nan,
                "opt2_text": np.nan,
                "dimension": np.nan,
            },
            {
                "trial_num": 6,
                "slide_type": "Decision",
                "decision_num": 30,
                "text": np.nan,
                "opt1_text": "Stay",
                "opt2_text": "Leave",
                "dimension": "agency",
            },
            {
                "trial_num": 7,
                "slide_type": "Game over",
                "decision_num": np.nan,
                "text": "Filtered",
                "opt1_text": np.nan,
                "opt2_text": np.nan,
                "dimension": np.nan,
            },
            {
                "trial_num": np.nan,
                "slide_type": "Decision",
                "decision_num": 999,
                "text": np.nan,
                "opt1_text": "Should",
                "opt2_text": "Drop",
                "dimension": "invalid",
            },
        ]
    )


def fake_embed_fn(texts, normalize=True, model="openai") -> np.ndarray:
    del normalize, model
    rows = []
    for idx, text in enumerate(texts):
        rows.append([float(len(text)), float(idx)])
    return np.asarray(rows, dtype=np.float32)


def test_find_project_root_prefers_env_var(tmp_path, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setenv("SNT_PROJECT_ROOT", str(root))
    start = root / "code" / "analyses" / "semantics" / "get_choice_embeddings.py"
    start.parent.mkdir(parents=True)
    start.touch()

    assert find_project_root(start) == root.resolve()


def test_resolve_task_file_from_data_info(tmp_path) -> None:
    root = tmp_path / "project"
    task_file = root / "data" / "info" / "social-navigation-task.csv"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("trial_num,slide_type\n1,Narration\n", encoding="utf-8")

    resolved = resolve_task_file(project_root=root, task_filename="social-navigation-task.csv")
    assert resolved == task_file.resolve()


def test_read_task_table_csv(tmp_path) -> None:
    file_path = tmp_path / "task.csv"
    file_path.write_text("trial_num,slide_type\n1,Narration\n", encoding="utf-8")

    table = read_task_table(file_path)
    assert table.shape == (1, 2)
    assert table.loc[0, "slide_type"] == "Narration"


def test_clean_and_context_helpers() -> None:
    assert clean_text(" a   b  c ") == "a b c"
    assert clean_text(None) == ""
    assert compose_context_text("   Previous line ", "  Option text ") == "Previous line\n\nOption text"
    assert compose_context_text(None, "  Option text ") == "Option text"


def test_canonicalize_slides_filters_and_sorts() -> None:
    canonical = canonicalize_slides(sample_snt_df())
    assert canonical["trial_num"].tolist() == [1.0, 2.0, 3.0, 5.0, 6.0]
    assert canonical["slide_type"].tolist() == [
        "Narration",
        "Decision",
        "Decision",
        "Narration",
        "Decision",
    ]


def test_build_embedding_text_inputs_content_and_order() -> None:
    inputs = build_embedding_text_inputs(sample_snt_df())

    assert inputs.decision_nums.tolist() == [10, 20, 30]
    assert inputs.no_context_texts == ["Ask for help", "Ignore", "Run", "Hide", "Stay", "Leave"]
    assert len(inputs.contextual_texts) == 12

    key_to_text = dict(zip(inputs.contextual_keys, inputs.contextual_texts))
    assert key_to_text[(10, 0, 0)] == "Intro text\n\nAsk for help"
    assert key_to_text[(10, 1, 1)] == "Intro text\n\nIgnore"
    assert key_to_text[(20, 0, 0)] == "Ask for help\n\nRun"
    assert key_to_text[(20, 1, 0)] == "Ignore\n\nRun"
    assert key_to_text[(30, 1, 1)] == "After scene\n\nLeave"
    assert inputs.metadata["decision_num"].tolist() == [10, 20, 30]


def test_build_choice_embeddings_shape_and_mapping() -> None:
    df = sample_snt_df()
    inputs = build_embedding_text_inputs(df)
    payload = build_choice_embeddings(df, model="mock-model", normalize=False, embed_fn=fake_embed_fn)

    assert payload["decision_nums"].tolist() == [10, 20, 30]
    assert payload["emb_ctx"].shape == (3, 2, 2, 2)
    assert payload["emb_noctx"].shape == (3, 2, 2)
    assert payload["model"] == "mock-model"

    contextual_vectors = fake_embed_fn(inputs.contextual_texts, normalize=False, model="mock-model")
    no_context_vectors = fake_embed_fn(inputs.no_context_texts, normalize=False, model="mock-model")

    ctx_row = inputs.contextual_keys.index((20, 1, 0))
    np.testing.assert_allclose(payload["emb_ctx"][1, 1, 0], contextual_vectors[ctx_row])
    np.testing.assert_allclose(payload["emb_noctx"][2, 1], no_context_vectors[5])


def test_save_choice_embeddings_writes_npz(tmp_path) -> None:
    payload = {
        "decision_nums": np.asarray([1, 2]),
        "emb_ctx": np.zeros((2, 2, 2, 1), dtype=np.float32),
        "emb_noctx": np.zeros((2, 2, 1), dtype=np.float32),
        "meta_json": '[{"decision_num":1,"dimension":"x"}]',
        "model": "mock-model",
    }

    out_file = tmp_path / "choice_embeddings.npz"
    save_choice_embeddings(payload, out_file)
    loaded = np.load(out_file, allow_pickle=True)

    np.testing.assert_array_equal(loaded["decision_nums"], payload["decision_nums"])
    assert loaded["meta_json"].item() == payload["meta_json"]
    assert loaded["model"].item() == "mock-model"


def test_generate_and_save_choice_embeddings_from_files_csv(tmp_path) -> None:
    root = tmp_path / "project"
    task_file = root / "data" / "info" / "social-navigation-task.csv"
    task_file.parent.mkdir(parents=True)
    sample_snt_df().to_csv(task_file, index=False)

    payload, resolved_task_file, output_file = generate_and_save_choice_embeddings_from_files(
        project_root=root,
        task_file=task_file,
        model="mock-model",
        normalize=False,
        embed_fn=fake_embed_fn,
    )

    assert resolved_task_file == task_file.resolve()
    assert output_file == (root / "data" / "narratives" / "choice_embeddings.npz").resolve()
    assert output_file.exists()
    assert payload["model"] == "mock-model"
