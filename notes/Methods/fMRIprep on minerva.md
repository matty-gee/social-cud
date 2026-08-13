**BIDS format**
Need correct BIDS filenames correct

BIDS/
└── sub-01/
    ├── anat/
    │   ├── sub-01_T1w.nii.gz
    │   └── sub-01_T1w.json
    │
    ├── func/
    │   ├── sub-01_task-socialnav_bold.nii.gz
    │   ├── sub-01_task-socialnav_bold.json
    │   ├── sub-01_task-rest_bold.nii.gz
    │   └── sub-01_task-rest_bold.json
    │
    └── fmap/
        ├── sub-01_dir-AP_epi.nii.gz
        ├── sub-01_dir-AP_epi.json
        ├── sub-01_dir-PA_epi.nii.gz
        └── sub-01_dir-PA_epi.json

**External dependencies**
TemplateFlow: make sure this cache exists and is bound 
- TemplateFlow is the template repository fMRIPrep uses to get standard anatomical and spatial reference files
- need a pre-staged TemplateFlow cache directory on the host filesystem (your “templates” folder) that includes the templates fMRIPrep will request
- batch script, you then bind-mount that cache into the container and tell fMRIPrep/TemplateFlow where it is: 
	- Bind: `-B /path/to/templates:/templateflow`
	- Set inside-container env var (because you use `--cleanenv`): `export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow`
FreeSurfer license + atlases
Tracking/version checks 

**Need an fmriprep singularity image** 


**Problems that I have run into**
LSF killed the job for exceeding the hard memory limit; could be a variety of things...
