from typing import Optional

YES_STR = {"yes", "y", "true", "t", "1"}
NO_STR  = {"no", "n", "false", "f", "0"}

def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _norm_str(s: pd.Series) -> pd.Series:
    """
    Normalize strings for robust comparisons:
      - lowercase
      - strip whitespace
      - convert NaN/None to ""
    """
    s2 = s.astype("string").fillna("").str.strip().str.lower()
    # normalize common NA tokens
    s2 = s2.replace({"n/a": "na", "n\\a": "na"})
    return s2

def _to_bool(s: pd.Series) -> pd.Series:
    """
    Robust boolean coercion:
      - numeric: nonzero => True
      - strings: yes/y/true/1 => True; no/n/false/0 => False
      - otherwise False
    """
    out = pd.Series(False, index=s.index)

    num = _to_numeric(s)
    out.loc[num.notna()] = num.loc[num.notna()] != 0

    st = _norm_str(s)
    out.loc[st.isin(YES_STR)] = True
    out.loc[st.isin(NO_STR)]  = False
    return out

def _cap_0_7(s: pd.Series) -> pd.Series:
    """Enforce SNI-style 0..7 ('7 or more')."""
    return _to_numeric(s).fillna(0).clip(lower=0, upper=7)

def _recode_neither_mother_father_both(s: pd.Series) -> pd.Series:
    """
    Recode either numeric codes or strings into counts:
      neither/none/0 -> 0
      mother/father/1/2 -> 1
      both/3 -> 2
    Works with your strings:
      'neither', 'mother', 'father', 'both', and 'na' (treated as 0).
    """
    # numeric path (if present)
    num = _to_numeric(s)
    mapped_num = num.map({0: 0, 1: 1, 2: 1, 3: 2})
    mapped_num = mapped_num.fillna(num.clip(lower=0, upper=2))

    # string path
    st = _norm_str(s)
    mapped_str = st.map({
        "": 0,
        "na": 0,
        "neither": 0,
        "none": 0,
        "no": 0,
        "mother": 1,
        "mom": 1,
        "mum": 1,
        "father": 1,
        "dad": 1,
        "both": 2,
    })

    out = mapped_num.copy()
    out.loc[num.isna()] = mapped_str.loc[num.isna()]
    return out.fillna(0).astype(int)

def _is_employed(employment_status: pd.Series) -> pd.Series:
    """
    Your employment field is string-coded:
      'employed_others', 'self_employed', 'no'
    We treat anything other than 'no' (and empty/na) as employed.
    Also supports numeric encodings if present (nonzero => employed).
    """
    num = _to_numeric(employment_status)
    employed = pd.Series(False, index=employment_status.index)

    employed.loc[num.notna()] = num.loc[num.notna()] != 0

    st = _norm_str(employment_status)
    not_emp = {"", "na", "no", "none", "unemployed", "not employed"}
    employed.loc[num.isna()] = ~st.loc[num.isna()].isin(not_emp)
    return employed

DEFAULT_SNI_COLMAP = {
    "marital_status": "sni_marital_status",
    "num_children": "sni_numbr_children",
    "children_contact": "sni_numbr_children_contact",

    "parents_living": "sni_parents_living",
    "parents_contact": "sni_parents_contact_biweekly",

    "inlaws_living": "sni_inlaws",
    "inlaws_contact": "sni_in_laws_talk_biweekly",

    "relatives_close": "sni_other_relatives_close_to",
    "relatives_contact": "sni_relatives_talk_to_biweekly",

    "num_close_friends": "sni_num_close_friends",
    "friends_contact": "sni_friends_contact_biweekly",

    "religious_member": "sni_religious_affiliation",
    "religious_contact": "sni_religious_affiliation_contact_biweekly",

    "classes_attend": "sni_attend_classes_regularly",
    "school_contact": "sni_students_teachers_biweekly_contact",

    "employment_status": "sni_employment_status",
    "supervise_n": "sni_ppl_supervise",
    "coworkers_contact": "sni_coworkers_biweekly_contact",

    "neighbors_contact": "sni_neighboors_biweekly_contact",

    "volunteer_yes": "sni_reg_volunteer_work",
    "volunteers_contact": "sni_volunteers_biweekly_contact",

    "group_yes": "sni_other_volunteer_grp_member_biweekly_contact",
    "group_total_contact": "sni_total_nmbr_other_volunteers_biweekly_contact",
}

def score_sni(
    df: pd.DataFrame,
    colmap: Optional[dict] = None,
    *,
    spouse_codes: set[int] = {1},
    spouse_strings_regex: str = r"(?:married|cohab|partner|living with)",
    return_components: bool = True,
) -> pd.DataFrame:
    """
    Scores SNI subscales:
      - sni_network_diversity: # high-contact roles (0..12)
      - sni_network_size: # people contacted >= biweekly (summed, with group cap)
      - sni_embedded_networks: # embedded domains (0..8)

    Updated to handle your string-coded:
      - parents/in-laws items: 'neither/mother/father/both/na'
      - employment_status: 'employed_others/self_employed/no'

    spouse_codes:
      Which numeric marital-status codes imply a spouse/partner role.
      Default {1}. Adjust if your codebook differs.
    """
    cm = dict(DEFAULT_SNI_COLMAP)
    if colmap:
        cm.update(colmap)

    missing = [c for c in cm.values() if c not in df.columns]
    if missing:
        raise KeyError(f"Missing SNI columns in df: {missing}")

    out = pd.DataFrame(index=df.index)

    # --- Spouse (Item 1): numeric codes or string match fallback
    marital = df[cm["marital_status"]]
    marital_num = _to_numeric(marital)

    spouse_present = pd.Series(False, index=df.index)
    spouse_present.loc[marital_num.notna()] = marital_num.loc[marital_num.notna()].astype(int).isin(spouse_codes)

    if spouse_strings_regex:
        st = _norm_str(marital)
        spouse_present.loc[marital_num.isna()] = st.loc[marital_num.isna()].str.contains(
            spouse_strings_regex, regex=True, na=False
        )

    spouse_n = spouse_present.astype(int)

    # --- Children (Items 2/2a)
    n_children = _to_numeric(df[cm["num_children"]]).fillna(0)
    children_contact = _cap_0_7(df[cm["children_contact"]])
    children_n = np.where((n_children > 0) | (children_contact > 0), children_contact, 0).astype(int)

    # --- Parents (Items 3/3a): your living/contact are strings
    parents_living_norm = _norm_str(df[cm["parents_living"]])
    parents_exist = ~parents_living_norm.isin({"", "na", "neither", "none", "no"})
    parents_contact_count = _recode_neither_mother_father_both(df[cm["parents_contact"]])
    parents_n = np.where(parents_exist, parents_contact_count, 0).astype(int)
    # keep positive contact even if existence flag is inconsistent/missing
    parents_n = np.where(parents_contact_count > 0, parents_contact_count, parents_n).astype(int)

    # --- In-laws (Items 4/4a): your inlaws has 'na'
    inlaws_norm = _norm_str(df[cm["inlaws_living"]])
    inlaws_applicable = ~inlaws_norm.isin({"", "na"})
    inlaws_exist = inlaws_applicable & ~inlaws_norm.isin({"neither", "none", "no"})
    inlaws_contact_count = _recode_neither_mother_father_both(df[cm["inlaws_contact"]])

    inlaws_n = np.where(inlaws_exist, inlaws_contact_count, 0).astype(int)
    # keep positive contact if applicable (not explicitly 'na')
    inlaws_n = np.where((inlaws_contact_count > 0) & inlaws_applicable, inlaws_contact_count, inlaws_n).astype(int)

    # --- Other relatives (Items 5/5a)
    relatives_close = _to_numeric(df[cm["relatives_close"]]).fillna(0)
    relatives_contact = _cap_0_7(df[cm["relatives_contact"]])
    relatives_n = np.where((relatives_close > 0) | (relatives_contact > 0), relatives_contact, 0).astype(int)

    # --- Close friends (Items 6/6a)
    n_close_friends = _to_numeric(df[cm["num_close_friends"]]).fillna(0)
    friends_contact = _cap_0_7(df[cm["friends_contact"]])
    friends_n = np.where((n_close_friends > 0) | (friends_contact > 0), friends_contact, 0).astype(int)

    # --- Religious group (Items 7/7a)
    religious_member = _to_bool(df[cm["religious_member"]])
    religious_contact = _cap_0_7(df[cm["religious_contact"]])
    religious_n = np.where(religious_member | (religious_contact > 0), religious_contact, 0).astype(int)

    # --- School/classes (Items 8/8a)
    classes_attend = _to_bool(df[cm["classes_attend"]])
    school_contact = _cap_0_7(df[cm["school_contact"]])
    school_n = np.where(classes_attend | (school_contact > 0), school_contact, 0).astype(int)

    # --- Work (Items 9/9a/9b): your employment_status is strings
    employed_yes = _is_employed(df[cm["employment_status"]])
    supervise_n = _cap_0_7(df[cm["supervise_n"]])
    coworkers_n = _cap_0_7(df[cm["coworkers_contact"]])
    work_contacts = (supervise_n + coworkers_n).astype(int)
    work_n = np.where(employed_yes | (work_contacts > 0), work_contacts, 0).astype(int)

    # --- Neighbors (Item 10)
    neighbors_n = _cap_0_7(df[cm["neighbors_contact"]]).astype(int)

    # --- Volunteering (Items 11/11a)
    volunteer_yes = _to_bool(df[cm["volunteer_yes"]])
    volunteers_contact = _cap_0_7(df[cm["volunteers_contact"]])
    volunteer_n = np.where(volunteer_yes | (volunteers_contact > 0), volunteers_contact, 0).astype(int)

    # --- Other groups (Item 12): cap summed contacts at 7
    group_yes = _to_bool(df[cm["group_yes"]])
    group_total = _to_numeric(df[cm["group_total_contact"]]).fillna(0).clip(lower=0)
    group_total_capped = group_total.clip(upper=7).astype(int)
    group_n = np.where(group_yes | (group_total_capped > 0), group_total_capped, 0).astype(int)

    # -----------------------
    # Role indicators (high-contact roles)
    # -----------------------
    role_spouse = (spouse_n > 0).astype(int)
    role_parent = (children_n > 0).astype(int)
    role_child = (parents_n > 0).astype(int)
    role_child_in_law = (inlaws_n > 0).astype(int)
    role_close_relative = (relatives_n > 0).astype(int)
    role_close_friend = (friends_n > 0).astype(int)
    role_religious = (religious_n > 0).astype(int)
    role_student = (school_n > 0).astype(int)
    role_work = (work_n > 0).astype(int)
    role_neighbor = (neighbors_n > 0).astype(int)
    role_volunteer = (volunteer_n > 0).astype(int)
    role_group = (group_n > 0).astype(int)

    out["sni_network_diversity"] = (
        role_spouse + role_parent + role_child + role_child_in_law + role_close_relative +
        role_close_friend + role_religious + role_student + role_work + role_neighbor +
        role_volunteer + role_group
    ).astype(int)

    # -----------------------
    # Network size (sum of role counts)
    # -----------------------
    out["sni_network_size"] = (
        spouse_n + children_n + parents_n + inlaws_n + relatives_n +
        friends_n + religious_n + school_n + work_n + neighbors_n +
        volunteer_n + group_n
    ).astype(int)

    # -----------------------
    # Embedded networks (8 domains)
    # -----------------------
    family_roles_n = (role_spouse + role_parent + role_child + role_child_in_law + role_close_relative).astype(int)
    family_people_n = (spouse_n + children_n + parents_n + inlaws_n + relatives_n).astype(int)

    embedded_family = ((family_roles_n >= 3) & (family_people_n >= 4)).astype(int)
    embedded_friends = (friends_n >= 4).astype(int)
    embedded_religious = (religious_n >= 4).astype(int)
    embedded_school = (school_n >= 4).astype(int)
    embedded_work = (work_n >= 4).astype(int)
    embedded_neighbors = (neighbors_n >= 4).astype(int)
    embedded_volunteering = (volunteer_n >= 4).astype(int)
    embedded_groups = (group_n >= 4).astype(int)

    out["sni_embedded_networks"] = (
        embedded_family + embedded_friends + embedded_religious + embedded_school +
        embedded_work + embedded_neighbors + embedded_volunteering + embedded_groups
    ).astype(int)

    if return_components:
        # role counts
        out["sni_spouse_n"] = spouse_n
        out["sni_children_n"] = children_n
        out["sni_parents_n"] = parents_n
        out["sni_inlaws_n"] = inlaws_n
        out["sni_relatives_n"] = relatives_n
        out["sni_friends_n"] = friends_n
        out["sni_religious_n"] = religious_n
        out["sni_school_n"] = school_n
        out["sni_work_n"] = work_n
        out["sni_neighbors_n"] = neighbors_n
        out["sni_volunteer_n"] = volunteer_n
        out["sni_group_n"] = group_n

        # embedded domain flags + family intermediates
        out["sni_embedded_family"] = embedded_family
        out["sni_embedded_friends"] = embedded_friends
        out["sni_embedded_religious"] = embedded_religious
        out["sni_embedded_school"] = embedded_school
        out["sni_embedded_work"] = embedded_work
        out["sni_embedded_neighbors"] = embedded_neighbors
        out["sni_embedded_volunteering"] = embedded_volunteering
        out["sni_embedded_groups"] = embedded_groups
        out["sni_family_roles_n"] = family_roles_n
        out["sni_family_people_n"] = family_people_n

    return out
