'''
Computes the choice option embeddings for the social navigation task
Also computes the cosine similarities between them
'''


from utils import *
from utils_nlp import *

narrative_dir = '../data/narratives'


#--------------------- define the text


# snt info
snt_df = pd.read_excel('../data/info/social-navigation-task.xlsx', sheet_name='gender_neutral')
snt_df = snt_df[np.isfinite(snt_df['trial_num'])]
snt_df = snt_df.sort_values(by='trial_num')
snt_df = snt_df[~snt_df["slide_type"].isin(["Game over", "Image"])].reset_index(drop=True)

# local decision slides
decision_df = snt_df[snt_df['slide_type'] == 'Decision'].copy() 
decision_df = decision_df[['decision_num', 'dimension', 
                           'opt1_text', 'opt2_text', 
                           'opt1_affil', 'opt2_affil', 
                           'opt1_power', 'opt2_power']]
decision_df["decision_num"] = decision_df["decision_num"].astype(int)


#--------------------- get embeddings


def save_embds_by_option(npz_path, decision_df, *, model="openai"):
    keys, sents = [], []
    for r in decision_df.itertuples(index=False):
        dnum = int(r.decision_num)
        for which in ("opt1", "opt2"):
            txt = getattr(r, f"{which}_text")
            if isinstance(txt, str) and txt.strip():
                keys.append((dnum, which))
                sents.append(txt)

    E = get_sentence_embeddings(sents, normalize=True, model=model)  # (N, D)

    payload = {"info": np.array([decision_df], dtype=object)}
    for (dnum, which), vec in zip(keys, E):
        payload[f"decision_{dnum}__{which}"] = np.asarray(vec, float)

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(npz_path, **payload)

def load_embds_by_option(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    info = z["info"][0]

    out = {}
    for k in z.files:
        if k == "info":
            continue
        # k: "dnum_<DECISION_NUM>__opt1/opt2"
        left, which = k.split("__", 1)
        dnum = int(left.replace("decision_", ""))
        out.setdefault(dnum, {})[which] = z[k]
    out["info"] = info
    return out

out_fname = f'{narrative_dir}/task-embeddings/decision-options_openai.npz'
if not os.path.isfile(out_fname):
    save_embds_by_option(out_fname, decision_df, model="openai")
embd_dict = load_embds_by_option(out_fname)


#--------------------- analyze choice cosine similarities & create df


rows = []
for r in decision_df.itertuples(index=False):
    dnum = int(r.decision_num)
    if dnum not in embd_dict:
        continue

    for which in ("opt1", "opt2"):
        vec = embd_dict[dnum].get(which, None)
        if vec is None:
            continue

        rows.append({
            "decision_num": dnum,
            "dimension": r.dimension,  # whatever is in the sheet (optional)
            "which": which,
            "opt_text": getattr(r, f"{which}_text"),
            "y_affil": float(getattr(r, f"{which}_affil")),
            "y_power": float(getattr(r, f"{which}_power")),
            "emb": vec,
        })

df_long = pd.DataFrame(rows)
df_long["dim"]  = np.where(df_long["y_affil"].isin([-1, 1]), "affil", np.where(df_long["y_power"].isin([-1, 1]), "power", None))
df_long["sign"] = np.where(df_long["dim"].eq("affil"), df_long["y_affil"], np.where(df_long["dim"].eq("power"), df_long["y_power"], np.nan)).astype("float")


def cosine_sim(a, b, eps=1e-12):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.dot(a, b) / ((np.linalg.norm(a) + eps) * (np.linalg.norm(b) + eps)))

def within_decision_option_cosines(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    df_long columns expected:
      decision_num, which in {"opt1","opt2"}, emb (array), plus any metadata
    Returns one row per decision with cosine(opt1, opt2).
    """
    out_rows = []
    for dnum, g in df_long.groupby("decision_num", sort=True):
        g1 = g[g["which"] == "opt1"]
        g2 = g[g["which"] == "opt2"]
        if (len(g1) != 1) or (len(g2) != 1):
            continue

        v1 = g1["emb"].iloc[0]
        v2 = g2["emb"].iloc[0]

        out_rows.append({
            "decision_num": int(dnum),
            "cos": cosine_sim(v1, v2),
            # optional metadata (take first non-null if present)
            "dimension": g["dimension"].iloc[0] if "dimension" in g.columns else None,
            "opt1_text": g1["opt_text"].iloc[0] if "opt_text" in g1.columns else None,
            "opt2_text": g2["opt_text"].iloc[0] if "opt_text" in g2.columns else None,
        })

    return pd.DataFrame(out_rows).sort_values("decision_num").reset_index(drop=True)

def word_count(x) -> int:
    _WORD_RE = re.compile(r"\b[\w']+\b")  # keeps contractions like "it's"
    if pd.isna(x):
        return 0
    return len(_WORD_RE.findall(str(x)))

df_cos = within_decision_option_cosines(df_long)
df_cos["word_count_opt1"] = df_cos["opt1_text"].map(word_count)
df_cos["word_count_opt2"] = df_cos["opt2_text"].map(word_count)
df_cos["word_count_sum"]  = df_cos["word_count_opt1"] + df_cos["word_count_opt2"]
df_cos["word_count_diff"] = (df_cos["word_count_opt1"] - df_cos["word_count_opt2"]).abs()
df_cos.to_csv(f'{narrative_dir}/task-embeddings/decision-options_cosines.csv', index=False)