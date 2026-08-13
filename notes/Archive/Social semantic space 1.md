
*Show both encoding and decoding in task*
Use the semantic embeddings of participant-specific choices (with local context) to learn semantic directions that predict the social locations on each trial (or some other similar social space feature)

Basic approach
1. Extract choices and embed them with LLM
2. Train a ridge model with leave one character out cross-validation to predict locations from embeddings 
	- Control for temporal autocorrelation, somehow 
	- test how much relationship context is optimal for decoding (likely depends on the feature)
		- this may be confounded with SNR (more info about character?) or clustering in time
		- test across embedding models 
	- how well can we decode character identity from the choice embeddings themselves?
		- may need to remove character names etc 
 

*Decode state in resting state volume patterns*
Test for replay-like sequences in resting state
For this we need a set of states, with some sort of valid successors
- NEED TO THINK THIS THROUGH CAREFULLY....

Basic approach:
1. decode some state/feature for each TR in rest
	- probabilities over discrete states (classification)
	- continuous coordinate (regression)
2. test whether those states/features occur in some sequence that we might expect (something like, if state A is active now, is state B (a valid successor) more likely next than invalid successor? eg p(A->B) > p(A->C))
3. do HPC, PCC and PFC all show replay like signatures? are they more likely to show those signatures that they showed in the task (specifically, decision > narrative )

Does decoding in resting state relate to memory or subjective placement accuracy?

*Other features*
Depending on results, follow up with other LLM features: e.g., trial-wise attention patterns, entropy, etc 


**Random**
Is there a way to add the spline-based logic back in? More expressly framed as a pattern similarity idea 



## Social space as a semantic subspace

Semantic space (from embeddings) is a very high-dimensional representation of meaning
Social space (affiliation, power) is a low-dimensional latent structure embedded within that space
If the task truly induces a social space, then variation in affiliation/power should correspond to systematic directions/subspace within the embedding space, not arbitrary drift

### Embedding the options

just embedded the options, individually, without any additional context 
clear affiliation/power structure in 2D with unsupervised PCA

![[Screenshot 2025-12-25 at 12.33.40 AM.png]]

PCs of the option embeddings
Correlate PC scores with different option features

![[76cbbefb-c884-4eb0-a580-6906628f6881.png]]


### Embedding the choices and relationships

*Basic approach*
We represent each decision trial's choice with different amount of context in them
We then embed into semantic space with an embedding LLM in various ways:
- Local context "no-ctxt"
	- Input text includes the current decision plus previous slides that help to interpret it
	- There is no relationship context involved outside of the local situation
	- So this should represent the choice given only the immediate situation
	- These embeddings strongly differentiate affiliation and power trials across subjects 
- Relationship-context "ctxt"
	- Input text includes everything in “no-ctxt” plus all the accumulated interaction history with the current character
	- These should represent the current choice within the context of the evolving relationship narrative with that specific character
	- These embeddings strongly differentiate character (kind of trivial)
- OTHER POSSIBILITIES...
	- ctxt(last 1, 2, 4, 8 interactions)
	- ctxt(first + current)
	- ctxt(first + last k)
	- ctxt(recency-weighted summary)
	- represent choices differently: chosen - unchosen, to get an action direction vector 

We can compute the cosine distance between these choice vectors as a proxy for the semantic distance between them 

*Context - no-context basic effects etc*
- Adding relationship context produces a consistent, moderate semantic shift in trial representations across subjects
- Doesn't seem to add a simple global accumulation artifact
- we can perfectly decode character identity from context embeddings
	- still present in no-ctxt embeddings (~0.75)
- the context produces very smooth semantic trajectories 
	- no-ctxt is much more volatile: w/o accumulated context, trial semantics jump around dramatically, local scenes and choices vary a lot
- can predict social coordinates better, even when demeaning by character 


*Some questions we can ask*
- Context sensitivity: Does adding relationship history measurably change the embedding of the current choice?
- Relational structure emergence: Does adding relationship history induce a geometry with interpretable low-dimensional structure consistent with “social space”-like axes or trajectories?

*Robustness checks*
- remove gender, names, etc 
- ensure the model can represent the amount of context I am asking it to 
- including context will increase the correlation with time 
	- maybe: compress history into a fixed-length summary.

### Learn a map from embeddings to locations

**Using cumulative relationship history embeddings:**
- Projected into 2D with MDS: they cleanly differentiate characters, as expected (descriptive sanity check only; axes are arbitrary and not used for inference).
- Potential issues related to temporal smoothness and cumulative context, which motivate explicit history-destroying nulls
- As a comparison, I have embeddings that are just each choice, without any previous relationship history 

**Is there a mapping from embeddings to locations?**
Do the embeddings contain enough information to (a) predict the current relationship location and (b) preserve the map-like geometry of the relationship trajectory?

Basic idea: If the behavioral latent variables are meaningful, they should be predictable from the semantic content that generated them

*Ridge regression approach* 
- Time-blocked CV
	- to avoid leakage across temporally adjacent trials and to keep the readout honest about generalizing across the narrative sequence
	- Pick alpha by CV
- Fit a linear readout: $L \approx EW + b$, where $L$ is the 2D location implied by the behavioral model, $E$ is the semantic embedding
- Two tests
	- Predict out-of-fold locations (joint R2): do embeddings predict locations  
	- RSA between location and predicted distance matrices: does extracted state preserve the relative structure of social space (map decoding; i.e., whether pairwise distances between states are preserved, not just marginal accuracy on each axis)
- Several subject-specific nulls (all group test against 0)
	- Character-mean baseline (CV): predict each trial from the training-fold mean location for that character, which controls for character identity and any stable character-level bias in location
	- Subject-specific shuffled choice nulls
	- No relationship history embeddings: test whether “memory/context” contributes beyond current-choice semantics
- Results & interpretations
	- Semantic content predicts the latent affiliation, power location 
	- The relational structure in semantic space also mirrors the relational structure in social space: suggests that the decoder is recovering an internal “map” rather than only axis-wise regressions
	- The context embeddings outperform no-context embeddings
	- Axis geometry in the readout weights is close to orthogonal
		- Affiliation and power also seem separable: cross-dimension specificity is high, ie true affiliation correlates strongly with predicted affiliation and true power correlates strongly with predicted power, while off-diagonal correlations are near zero
	- The model also seems to implicitly learn directionality of the change: Beyond recovering locations, the predicted step vectors align with true step vectors: angular error is well below 90° chance (median ≈ mid-40° range) and is significantly better than a within-subject shuffled temporal null.


**Do these mappings generalize across characters?**
Does the embedding → location mapping learned from other characters generalize to a held-out character?
Ridge regression:
- LOCO CV
	- For each held-out character:
		- tune alpha on training characters only
		- fit ridge on training characters
		- predict held-out character
		- score with average Pearson correlation across affiliation and power
- Subject-specific shuffled choice nulls & z-scores, test sample against 0


**Are the learned affiliation and power directions orthogonal?** 
Are the learned readout directions for affiliation and power close to orthogonal (two distinct factors) versus collapsing onto one “good–bad” axis?

The embeddings have weights that map each dimension to affiliation and to power
Do these weight vectors correlate or are they orthogonal?
I think I need some kind of targeted null here, to really make sure I am not being confused 
In HD space, random vectors are often orthogonal: how can I be sure that this isn't the case? 
- The model predicts the outputs, so it has learned something at least 
Maybe I can test these directions by projecting new statements onto them or something?




### Do semantic embeddings improve neural fit?

*We fit 2 RSA models per subject*: 
Within-subject RSA models that predict trialwise neural pattern (correlation) distance from trialwise semantic (cosine) distance, while controlling for temporal distance, temporal distance squared, and character-identity structure (five within-character RDVs)
We fit two kinds of semantic embedding models:
- Contextualized embeddings (`ctxt`): choice semantics embedded with accumulated relationship context
- Decontextualized embeddings (`noctxt`): local choice semantics without any extra relationship context
We then computed the subject-specific ROI-by-ROI difference of ctxt beta - noctxt beta, where a positive difference means the region is sensitive to contextualized neural geometry

*And then tested how this differs (ROI by ROI) for healthy controls (HCs) v cocaine users (CDs), controlling for CTQ, fd, memory, and sex* 
HCs are more sensitive to context in DMN regions (e.g. PCC) as well as right hippocampus (HPC)
- Contextual embeddings predicting neural geometry better than just the semantics of the choice
- DMN is a plausible region to do the kind of context accumulation/narrative integration function
	- Do we also see impairments in DMN functional correlation to HPC?

Then I did a within- and between-character style analysis, to differentiate within relationship versus between information...

HCs use contextual information to organize representations across characters more than CDs do

**Show robustness**
- Show that this is not explained by temporal autocorrelation or character identity at all: more control models, show correlations between residuals and time or something, etc
- replicate in Tavares data
- multiple embedding models

**Make additional neural connections**
- Test differing amounts of context: various amounts of local context, various amounts of previous interactions, just current and first, etc... 
- See if we can isolate which features in particular predict this: e.g., use the ridge regression approach to predict this with social locations built in somehow 

**Need to make some non-neural connection**
- maybe small betas means worse correspondence to dots...
	- try different ways of representing the dots-behavior correspondence
- to real world behaviors, esp ones that might require integration over time 
	- SNI
	- mentalizing
- to cocaine use variables
	- might be related to age at first use negatively, years of using plausibly...?
	- fraction of years using positively correlates, esp with PCC, slightly weaker in L HPC
		- even when controlling for age, asi coc pasmonth, cocaine screening, sex, memory, motion
		- maybe theyre just less impaired? have adapted? less severe users at this point?
		- they maybe are trending to having more accurate dots too... 

### Can we use this approach to build an encoding model for fMRI data?

Treat brain as a function that transforms features into activity 

Voxelwise encoding model
- Y: trial-wise betas
- X: (n_trials, n_features)
	- semantic embeddings
	- social locations
	- feature space projection (hybrid semantic-social approach): this just rotates the LLM space to align with the behavioral history in some sense 
- Model to map X to Y: learn a map of weights W for every voxel so that Y ~= XW 
- Cross-validation: no runs in the task so have to get creative 
	- chop up the task into fake runs
	- leave one character out (LOCO)

RSA approach
- fit model to map X to Y for training set 
	- use regularization to handle high-dimensionality 
- generate synthetic brain activity for the held-out set 
- compute correlations between RDMs for real and synthetic brain 

Might be able to use this model fMRI activity, e.g. with some kind of encoding model... and then can use what I learn from encoding models for common cause too 






