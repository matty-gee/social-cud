"""
Build task-level embeddings for Social Navigation Task decision options.

This script reads the Social Navigation Task slide table (snt_df), extracts all
Decision slides, and embeds both options (opt1_text, opt2_text) using
`utils_nlp.get_sentence_embeddings`.

Two embedding stores are saved to a single compressed NPZ file:

1) emb_ctx  : (n_decisions, 2, 2, D)
   Contextual embeddings with a 1-slide-back context.
   - Axis 1 (prev_choice): which previous option is assumed if the previous slide
     was a Decision (0=prev opt1, 1=prev opt2). If the previous slide was not a
     Decision (or does not exist), both contexts are identical.
   - Axis 2 (option): the current decision option (0=opt1, 1=opt2).

2) emb_noctx: (n_decisions, 2, D)
   No-context embeddings for each decision option (opt1/opt2).

Also saves `decision_nums` (mapping each row to decision_num), minimal metadata
(`decision_num`, `dimension`) as JSON, and the embedding model identifier.

Outputs are designed to be easily indexed later using a subject’s behavioral
decision_num and selected_option (0/1), with optional conditioning on the previous
decision’s selected_option for context selection.
"""



import os
import numpy as np
import pandas as pd
from utils_nlp import get_sentence_embeddings

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _clean(x):
    return "" if not isinstance(x, str) else " ".join(x.split())

def _ctx(prev_txt, opt_txt):
    prev_txt = _clean(prev_txt)
    opt_txt  = _clean(opt_txt)
    return f"{prev_txt}\n\n{opt_txt}" if prev_txt else opt_txt

# ------------------------------------------------------------
# main builder
# ------------------------------------------------------------

out_path = "../data/narratives/choice_embeddings.npz"
model = "openai"
normalize = True

# ---- canonical slide order ----
snt = (
    snt_df[np.isfinite(snt_df["trial_num"])]
    .sort_values("trial_num")
    .query("slide_type not in ['Game over','Image']")
    .reset_index(drop=True)
)

# decision rows
dec_mask = snt["slide_type"] == "Decision"
dec_rows = snt[dec_mask].copy()
dec_rows["decision_num"] = dec_rows["decision_num"].astype(int)

decision_nums = dec_rows["decision_num"].to_numpy()
n_dec = len(decision_nums)

# mapping decision_num → index
dnum_to_idx = {d: i for i, d in enumerate(decision_nums)}

# ---- collect texts to embed ----
texts = []
keys  = []  # (d_idx, prev_choice, option)

for d_idx, row in dec_rows.iterrows():
    dnum = int(row["decision_num"])
    opt_texts = [_clean(row["opt1_text"]), _clean(row["opt2_text"])]

    # find previous slide
    slide_idx = snt.index[snt["decision_num"] == dnum][0]
    if slide_idx == 0:
        prev_is_dec = False
        prev_texts = ["", ""]
        prev_dnum = np.nan
    else:
        prev = snt.iloc[slide_idx - 1]
        prev_is_dec = prev["slide_type"] == "Decision"
        prev_dnum = int(prev["decision_num"]) if prev_is_dec else np.nan
        if prev_is_dec:
            prev_texts = [_clean(prev["opt1_text"]), _clean(prev["opt2_text"])]
        else:
            prev_texts = [_clean(prev["text"]), _clean(prev["text"])]

    for pc in (0, 1):
        for opt in (0, 1):
            texts.append(_ctx(prev_texts[pc], opt_texts[opt]))
            keys.append((dnum, pc, opt))

# ---- embed ----
E = get_sentence_embeddings(texts, normalize=normalize, model=model)
D = E.shape[1]

emb_ctx = np.zeros((n_dec, 2, 2, D), dtype=np.float32)

for vec, (dnum, pc, opt) in zip(E, keys):
    emb_ctx[dnum_to_idx[dnum], pc, opt] = vec

# ---- also store no-context embeddings ----
opt_texts = []
for _, row in dec_rows.iterrows():
    opt_texts.append(_clean(row["opt1_text"]))
    opt_texts.append(_clean(row["opt2_text"]))

E0 = get_sentence_embeddings(opt_texts, normalize=normalize, model=model)
emb_noctx = E0.reshape(n_dec, 2, D).astype(np.float32)

meta = dec_rows[[
    "decision_num", "dimension"
]].reset_index(drop=True)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(
    out_path,
    decision_nums=decision_nums,
    emb_ctx=emb_ctx,
    emb_noctx=emb_noctx,
    meta_json=meta.to_json(orient="records"),
    model=model,
)
