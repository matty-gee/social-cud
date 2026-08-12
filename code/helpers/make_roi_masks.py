def fetch_ho_atlas(atlas_name: str, thr: int = 50, symmetric_split: bool = True):
    """
    Fetch a Harvard-Oxford atlas and return (img, labels).
    """
    ho = datasets.fetch_atlas_harvard_oxford(
        atlas_name=f"{atlas_name}-thr{thr}-1mm",
        symmetric_split=symmetric_split
    )
    img = nimg.load_img(ho.maps)
    labels = list(ho.labels)
    return img, labels

def label_indices(substrings, labels):
    """
    Return atlas label indices whose names contain ALL substrings
    (case-insensitive).
    """
    out = []
    for i, lab in enumerate(labels):
        name = lab.lower()
        if all(s.lower() in name for s in substrings):
            out.append(i)
    return out

def save_roi_mask(atlas_img, labels, name, substrings, out_dir, thr: int):
    """
    Build and save a binary ROI mask from a Harvard-Oxford atlas.

    Parameters
    ----------
    atlas_img : Nifti1Image
        Atlas image (max-probability labels).
    labels : list of str
        Atlas label names.
    name : str
        Short ROI name prefix for the output file.
    substrings : list of str
        Substrings that must all be present in the label name.
    out_dir : str or Path
        Output directory.
    thr : int
        Threshold used in atlas name (for filename consistency).

    Returns
    -------
    out_path : str
        Path to saved ROI mask.
    """
    idxs = label_indices(substrings, labels)
    if len(idxs) == 0:
        raise ValueError(f"No atlas labels match: {substrings}")

    data = atlas_img.get_fdata().astype(int)
    mask = np.isin(data, idxs).astype(np.uint8)

    roi_img = nib.Nifti1Image(mask, atlas_img.affine, atlas_img.header)
    out_path = os.path.join(out_dir, f"{name}_harvardoxford_maxprob-thr{thr}_1mm.nii.gz")
    nib.save(roi_img, out_path)
    return out_path

def split_hpc_ant_post(mask_path, *, y_split: float = -25.0, thr: int = 50, out_dir: str = "../masks/ROIs"):
    """
    Split a hippocampal mask into anterior vs posterior using MNI Y coordinate.

    Parameters
    ----------
    mask_path : str
        Path to a hippocampal mask NIfTI file (binary or probabilistic).
    y_split : float, default -25.0
        Y (MNI) boundary in mm; Y > y_split is 'anterior'.
    thr : int, default 50
        Threshold label used in filenames (for consistent naming).
    out_dir : str, default "../masks/ROIs"
        Directory to save the split masks.

    Returns
    -------
    anterior_img, posterior_img, anterior_out, posterior_out
    """
    hpc_img = nimg.load_img(mask_path)

    # Binarize (in case it's probabilistic)
    hpc_data = (hpc_img.get_fdata() > 0).astype(np.uint8)
    affine = hpc_img.affine
    shape = hpc_data.shape

    # Build 3D field of world-space Y coordinates
    i, j, k = np.indices(shape)
    vox = np.column_stack([i.ravel(), j.ravel(), k.ravel()])   # (N, 3)
    world = nib.affines.apply_affine(affine, vox)              # (N, 3)
    y_world = world[:, 1].reshape(shape)                       # Y in mm (MNI space)

    anterior_bool = (hpc_data > 0) & (y_world > y_split)
    posterior_bool = (hpc_data > 0) & (y_world <= y_split)

    anterior_img = nimg.new_img_like(hpc_img, anterior_bool.astype(np.uint8), copy_header=True)
    posterior_img = nimg.new_img_like(hpc_img, posterior_bool.astype(np.uint8), copy_header=True)

    # Infer hemisphere from filename
    if "-L_" in mask_path:
        side = "L"
    elif "-R_" in mask_path:
        side = "R"
    else:
        raise ValueError(f"Could not infer hemisphere from path: {mask_path}")

    os.makedirs(out_dir, exist_ok=True)
    anterior_out = os.path.join(out_dir, f"HPC-{side}_anterior_thr{thr}-1mm.nii.gz")
    posterior_out = os.path.join(out_dir, f"HPC-{side}_posterior_thr{thr}-1mm.nii.gz")

    anterior_img.to_filename(anterior_out)
    posterior_img.to_filename(posterior_out)

    return anterior_img, posterior_img, anterior_out, posterior_out

# thr = 50
# out_dir = "../masks/ROIs" 
# os.makedirs(out_dir, exist_ok=True)

# # Harvard-Oxford cortical + subcortical
# atlas_img, labels         = fetch_ho_atlas("cort-maxprob", thr=thr, symmetric_split=True)
# atlas_img_sub, labels_sub = fetch_ho_atlas("sub-maxprob", thr=thr, symmetric_split=True)
# roi_files = {
#     # Primary visual cortex (≈ Intracalcarine)
#     "V1-L": save_roi_mask(atlas_img, labels, "V1-L", ["intracalcarine", "left"], out_dir, thr),
#     "V1-R": save_roi_mask(atlas_img, labels, "V1-R", ["intracalcarine", "right"], out_dir, thr),
#     # "V1-bilat": save_roi_mask(atlas_img, labels, "V1-bilat", ["intracalcarine"], out_dir, thr),

#     # Primary motor cortex (≈ Precentral Gyrus)
#     "M1-L": save_roi_mask(atlas_img, labels, "M1-L", ["precentral gyrus", "left"], out_dir, thr),
#     "M1-R": save_roi_mask(atlas_img, labels, "M1-R", ["precentral gyrus", "right"], out_dir, thr),
#     # "M1-bilat": save_roi_mask(atlas_img, labels, "M1-bilat", ["precentral gyrus"], out_dir, thr),

#     # Hippocampus
#     "HPC-L": save_roi_mask(atlas_img_sub, labels_sub, "HPC-L", ["hippocampus", "left"], out_dir, thr),
#     "HPC-R": save_roi_mask(atlas_img_sub, labels_sub, "HPC-R", ["hippocampus", "right"], out_dir, thr),
#     # "HPC-bilat": save_roi_mask(atlas_img_sub, labels_sub, "HPC-bilat", ["hippocampus"], out_dir, thr),

#     # Primary auditory cortex ≈ Heschl's gyrus
#     "A1-L": save_roi_mask(atlas_img, labels, "A1-L", ["heschl", "left"], out_dir, thr),
#     "A1-R": save_roi_mask(atlas_img, labels, "A1-R", ["heschl", "right"], out_dir, thr),
#     # "A1-bilat": save_roi_mask(atlas_img, labels, "A1-bilat", ["heschl"], out_dir, thr),
# }

# # Add Tavares seed (pre-existing file)
# roi_files["Tavares"] = os.path.join(out_dir, "Tavares_x-33_y-18_z-15_10mm.nii.gz")

# # Split hippocampus into anterior/posterior for L/R
# y_split = -25.0
# for side in ("L", "R"):
#     mask_path = os.path.join(out_dir, f"HPC-{side}_harvardoxford_maxprob-thr{thr}_1mm.nii.gz")
#     ant_img, post_img, ant_out, post_out = split_hpc_ant_post(
#         mask_path,
#         y_split=y_split,
#         thr=thr,
#         out_dir=out_dir,
#     )
#     roi_files[f"HPC-{side}_ant"] = ant_out
#     roi_files[f"HPC-{side}_post"] = post_out

# print("Saved ROI masks:")
# for k, v in roi_files.items():
#     print(f"  {k}: {v}")