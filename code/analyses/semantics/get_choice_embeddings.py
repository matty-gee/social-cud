"""
Build task-level embeddings for Social Navigation Task decision options.

The module is split into pure functions so it can be imported and tested with
pytest. The only side-effecting operation is `save_choice_embeddings`.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import numpy as np
import pandas as pd

DEFAULT_TASK_FILENAME = "social-navigation-task.xlsx"
DEFAULT_OUTPUT_SUBPATH = Path("data/narratives/choice_embeddings.npz")
EXCLUDED_SLIDE_TYPES = {"Game over", "Image"}
EmbeddingFn = Callable[..., np.ndarray]


def find_project_root(start: Path | str) -> Path:
    """
    Find project root using the same heuristic as `shared/main.py`.

    Priority:
    1) `SNT_PROJECT_ROOT` env var
    2) first parent containing `data` and `results`, plus one of
       (`figures`, `masks`, `analyses`, `.git`, `code`)
    3) fallback: two levels up from `start`
    """
    env_root = os.environ.get("SNT_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    start_path = Path(start).expanduser().resolve()
    probe_parents = [start_path.parent, *start_path.parents] if start_path.is_file() else [start_path, *start_path.parents]

    must_exist = ("data", "results")
    any_exist = ("figures", "masks", "analyses", ".git", "code")

    for candidate in probe_parents:
        if all((candidate / name).exists() for name in must_exist) and any(
            (candidate / name).exists() for name in any_exist
        ):
            return candidate.resolve()

    return start_path.parent.parent.resolve()

PROJECT_ROOT = find_project_root(Path(__file__).resolve())
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / DEFAULT_OUTPUT_SUBPATH

def candidate_task_paths(project_root: Path, task_filename: str = DEFAULT_TASK_FILENAME) -> list[Path]:
    """Return candidate locations for the Social Navigation Task table."""
    root = Path(project_root).expanduser().resolve()
    candidates = [
        root / "data" / "info" / task_filename,
        root / "info" / task_filename,
        root / "code" / "data" / "info" / task_filename,
        root / "data" / task_filename,
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped

def resolve_task_file(
    *,
    project_root: Path | str | None = None,
    task_filename: str = DEFAULT_TASK_FILENAME,
) -> Path:
    """Find the first existing task file path."""
    root = PROJECT_ROOT if project_root is None else Path(project_root).expanduser().resolve()
    for candidate in candidate_task_paths(root, task_filename=task_filename):
        if candidate.exists():
            return candidate
    tried = "\n".join(f"- {path}" for path in candidate_task_paths(root, task_filename=task_filename))
    raise FileNotFoundError(
        f"Could not locate task file '{task_filename}'. Tried:\n{tried}"
    )

def read_task_table(task_file: Path | str, *, sheet_name: str | int | None = None) -> pd.DataFrame:
    """Read task table from Excel/CSV/TSV."""
    path = Path(task_file).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        kwargs: dict[str, Any] = {}
        if sheet_name is not None:
            kwargs["sheet_name"] = sheet_name
        try:
            return pd.read_excel(path, **kwargs)
        except ImportError as exc:
            raise ImportError(
                f"Reading Excel file '{path}' requires 'openpyxl'. "
                "Install it or provide a CSV/TSV task file."
            ) from exc
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported task file extension: '{suffix}' for file '{path}'.")

def load_task_slides(
    *,
    project_root: Path | str | None = None,
    task_file: Path | str | None = None,
    sheet_name: str | int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load the Social Navigation Task table and return `(DataFrame, resolved_path)`."""
    if task_file is None:
        resolved_file = resolve_task_file(project_root=project_root)
    else:
        resolved_file = Path(task_file).expanduser().resolve()
        if not resolved_file.exists():
            raise FileNotFoundError(f"Task file does not exist: {resolved_file}")
    return read_task_table(resolved_file, sheet_name=sheet_name), resolved_file

def _load_default_embedding_fn() -> EmbeddingFn | None:
    """Try known embedding loaders without forcing heavy imports at module import time."""
    try:
        from utils_nlp import get_sentence_embeddings

        return get_sentence_embeddings
    except Exception:
        pass

    code_dir = PROJECT_ROOT / "code"
    if code_dir.exists():
        code_dir_str = str(code_dir)
        if code_dir_str not in sys.path:
            sys.path.insert(0, code_dir_str)
        try:
            from shared.nlp import get_sentence_embeddings

            return get_sentence_embeddings
        except Exception:
            pass

    return None

_default_get_sentence_embeddings = _load_default_embedding_fn()

@dataclass(frozen=True)
class EmbeddingTextInputs:
    """All text inputs required to build contextual and no-context embeddings."""

    decision_nums: np.ndarray
    contextual_texts: list[str]
    contextual_keys: list[tuple[int, int, int]]
    no_context_texts: list[str]
    metadata: pd.DataFrame

def clean_text(value: Any) -> str:
    """Normalize whitespace and coerce non-strings to empty text."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())

def compose_context_text(previous_text: Any, option_text: Any) -> str:
    """Attach previous-slide text to the current option text when available."""
    prev_text = clean_text(previous_text)
    curr_text = clean_text(option_text)
    return f"{prev_text}\n\n{curr_text}" if prev_text else curr_text

def canonicalize_slides(snt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and sort slides into canonical task order.

    Keeps rows with finite `trial_num`, removes excluded slide types, and sorts
    by `trial_num`.
    """
    required = {"trial_num", "slide_type"}
    missing = required.difference(snt_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    trial_num = pd.to_numeric(snt_df["trial_num"], errors="coerce")
    finite_trial_mask = np.isfinite(trial_num.to_numpy(dtype=float))

    canonical = snt_df.loc[finite_trial_mask].copy()
    canonical["trial_num"] = trial_num.loc[finite_trial_mask]
    canonical = canonical.loc[~canonical["slide_type"].isin(EXCLUDED_SLIDE_TYPES)]
    return canonical.sort_values("trial_num").reset_index(drop=True)

def previous_context_texts(slides: pd.DataFrame, slide_idx: int) -> tuple[str, str]:
    """Return the two possible previous context strings for a slide index."""
    if slide_idx <= 0:
        return "", ""

    prev_slide = slides.iloc[slide_idx - 1]
    if prev_slide["slide_type"] == "Decision":
        return clean_text(prev_slide.get("opt1_text")), clean_text(prev_slide.get("opt2_text"))

    prev_text = clean_text(prev_slide.get("text"))
    return prev_text, prev_text

def build_embedding_text_inputs(snt_df: pd.DataFrame) -> EmbeddingTextInputs:
    """Build ordered text inputs and keys needed for downstream embedding calls."""
    slides = canonicalize_slides(snt_df)

    decision_nums: list[int] = []
    contextual_texts: list[str] = []
    contextual_keys: list[tuple[int, int, int]] = []
    no_context_texts: list[str] = []
    metadata_rows: list[dict[str, Any]] = []

    for slide_idx, row in slides.iterrows():
        if row["slide_type"] != "Decision":
            continue

        if pd.isna(row.get("decision_num")):
            raise ValueError("Found Decision row with missing decision_num.")

        decision_num = int(row["decision_num"])
        decision_nums.append(decision_num)
        metadata_rows.append(
            {
                "decision_num": decision_num,
                "dimension": row.get("dimension"),
            }
        )

        options = [clean_text(row.get("opt1_text")), clean_text(row.get("opt2_text"))]
        previous_options = previous_context_texts(slides, slide_idx)

        for prev_choice in (0, 1):
            for option_idx in (0, 1):
                contextual_texts.append(compose_context_text(previous_options[prev_choice], options[option_idx]))
                contextual_keys.append((decision_num, prev_choice, option_idx))

        no_context_texts.extend(options)

    if not decision_nums:
        raise ValueError("No Decision slides were found after filtering.")

    metadata = pd.DataFrame(metadata_rows, columns=["decision_num", "dimension"])
    return EmbeddingTextInputs(
        decision_nums=np.asarray(decision_nums, dtype=int),
        contextual_texts=contextual_texts,
        contextual_keys=contextual_keys,
        no_context_texts=no_context_texts,
        metadata=metadata,
    )

def resolve_embedding_fn(embed_fn: EmbeddingFn | None = None) -> EmbeddingFn:
    """Pick an embedding function, preferring explicit dependency injection."""
    if embed_fn is not None:
        return embed_fn
    if _default_get_sentence_embeddings is None:
        raise ImportError(
            "Could not import a default embedding function. "
            "Pass `embed_fn=` explicitly or make `utils_nlp.get_sentence_embeddings` "
            "or `shared.nlp.get_sentence_embeddings` importable."
        )
    return _default_get_sentence_embeddings


def validate_embedding_matrix(vectors: np.ndarray, expected_rows: int, label: str) -> np.ndarray:
    """Ensure embedding output has the expected 2D shape and row count."""
    arr = np.asarray(vectors)
    if arr.ndim != 2:
        raise ValueError(f"{label} embeddings must be 2D, received shape {arr.shape}.")
    if arr.shape[0] != expected_rows:
        raise ValueError(
            f"{label} embeddings row mismatch: expected {expected_rows}, got {arr.shape[0]}."
        )
    return arr.astype(np.float32, copy=False)


def assemble_embedding_arrays(
    text_inputs: EmbeddingTextInputs,
    contextual_vectors: np.ndarray,
    no_context_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble final tensors: contextual `(N,2,2,D)` and no-context `(N,2,D)`."""
    n_decisions = len(text_inputs.decision_nums)
    dnum_to_idx = {decision_num: i for i, decision_num in enumerate(text_inputs.decision_nums)}

    embed_dim = contextual_vectors.shape[1]
    contextual = np.zeros((n_decisions, 2, 2, embed_dim), dtype=np.float32)
    for vec, (decision_num, prev_choice, option_idx) in zip(
        contextual_vectors,
        text_inputs.contextual_keys,
    ):
        contextual[dnum_to_idx[decision_num], prev_choice, option_idx] = vec

    if no_context_vectors.shape[1] != embed_dim:
        raise ValueError(
            "Embedding dimension mismatch between contextual and no-context arrays: "
            f"{embed_dim} != {no_context_vectors.shape[1]}."
        )

    no_context = no_context_vectors.reshape(n_decisions, 2, embed_dim).astype(np.float32, copy=False)
    return contextual, no_context


def build_choice_embeddings(
    snt_df: pd.DataFrame,
    *,
    model: str = "openai",
    normalize: bool = True,
    embed_fn: EmbeddingFn | None = None,
) -> dict[str, Any]:
    """Build all arrays and metadata required for `choice_embeddings.npz`."""
    text_inputs = build_embedding_text_inputs(snt_df)
    embedding_fn = resolve_embedding_fn(embed_fn)

    contextual_vectors = embedding_fn(text_inputs.contextual_texts, normalize=normalize, model=model)
    no_context_vectors = embedding_fn(text_inputs.no_context_texts, normalize=normalize, model=model)

    contextual_vectors = validate_embedding_matrix(
        contextual_vectors,
        expected_rows=len(text_inputs.contextual_texts),
        label="contextual",
    )
    no_context_vectors = validate_embedding_matrix(
        no_context_vectors,
        expected_rows=len(text_inputs.no_context_texts),
        label="no-context",
    )

    emb_ctx, emb_noctx = assemble_embedding_arrays(
        text_inputs=text_inputs,
        contextual_vectors=contextual_vectors,
        no_context_vectors=no_context_vectors,
    )

    return {
        "decision_nums": text_inputs.decision_nums,
        "emb_ctx": emb_ctx,
        "emb_noctx": emb_noctx,
        "meta_json": text_inputs.metadata.to_json(orient="records"),
        "model": model,
    }


def save_choice_embeddings(payload: dict[str, Any], out_path: str | Path) -> Path:
    """Persist embedding outputs to a compressed NPZ file."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, **payload)
    return out_file


def generate_and_save_choice_embeddings(
    snt_df: pd.DataFrame,
    *,
    out_path: str | Path = DEFAULT_OUTPUT_PATH,
    model: str = "openai",
    normalize: bool = True,
    embed_fn: EmbeddingFn | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: build payload and write it to disk."""
    payload = build_choice_embeddings(
        snt_df,
        model=model,
        normalize=normalize,
        embed_fn=embed_fn,
    )
    save_choice_embeddings(payload, out_path)
    return payload


def generate_and_save_choice_embeddings_from_files(
    *,
    project_root: Path | str | None = None,
    task_file: Path | str | None = None,
    sheet_name: str | int | None = None,
    out_path: str | Path | None = None,
    model: str = "openai",
    normalize: bool = True,
    embed_fn: EmbeddingFn | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """
    Discover task file using project layout hints and write embeddings to disk.

    Returns `(payload, task_file_path, output_path)`.
    """
    root = PROJECT_ROOT if project_root is None else Path(project_root).expanduser().resolve()
    snt_df, resolved_task_file = load_task_slides(
        project_root=root,
        task_file=task_file,
        sheet_name=sheet_name,
    )
    resolved_out_path = (
        (root / DEFAULT_OUTPUT_SUBPATH).resolve()
        if out_path is None
        else Path(out_path).expanduser().resolve()
    )
    payload = generate_and_save_choice_embeddings(
        snt_df,
        out_path=resolved_out_path,
        model=model,
        normalize=normalize,
        embed_fn=embed_fn,
    )
    return payload, resolved_task_file, resolved_out_path


def main() -> None:
    """Script entry point for file-based embedding generation."""
    parser = argparse.ArgumentParser(description="Build and save Social Navigation choice embeddings.")
    parser.add_argument("--project-root", type=str, default=None, help="Optional project root override.")
    parser.add_argument("--task-file", type=str, default=None, help="Optional explicit task file path.")
    parser.add_argument(
        "--sheet-name",
        type=str,
        default=None,
        help="Excel sheet name or index (stringified integer). Ignored for CSV/TSV.",
    )
    parser.add_argument("--out-path", type=str, default=None, help="Output .npz path.")
    parser.add_argument("--model", type=str, default="openai", help="Embedding model identifier.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable embedding normalization.")
    args = parser.parse_args()

    sheet_name: str | int | None = args.sheet_name
    if isinstance(sheet_name, str) and sheet_name.isdigit():
        sheet_name = int(sheet_name)

    _, task_file, out_file = generate_and_save_choice_embeddings_from_files(
        project_root=args.project_root,
        task_file=args.task_file,
        sheet_name=sheet_name,
        out_path=args.out_path,
        model=args.model,
        normalize=not args.no_normalize,
    )
    print(f"Task file: {task_file}")
    print(f"Saved embeddings: {out_file}")


if __name__ == "__main__":
    main()
