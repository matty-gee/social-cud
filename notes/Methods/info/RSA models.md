## multiple regression RSA

**Set of choices that will be same across different models**
- input features 
	- distance metrics
		- neural: correlation distance
		- continuous features: absolute difference of euclidean distance
		- categorical: mismatch (0 same v. 1 different)
	- z-score continuous features
	- drop neutral and missed trials 
- model: standard OLS regression (ridge if the number of predictors grows big)
- inference: use circular-shift or block-shift permutations of residuals (Freedman–Lane style)
	- do not want to rely on i.i.d. assumption over trial pairs, given that we have a single-run 

**Candidate features at the trial-level**
- character identity as categorical 
- current options + choice
	- chosen option coordinates
	- unchosen option coordinates
	- option-set geometry: e.g. distance between options
	- action vector (chosen − current position, or chosen − unchosen)
	- can think of extending this to outcome-like signals later, by looking at neural response as a function of semantic distance of next trial after decision 
- relationship history
	- cumulative mean of past chosen coordinates
- semantic content
	- option embeddings
	- word count
	- sentiment 

**Model likely confounds**
- time/drift 
- character block structure 
- low-level text features

**Check separability before fitting RSAs**
- Compute pairwise correlations among all candidate predictor RDVs (including nuisance RDVs)
- Identify near-redundant predictors
- Sanity-check distributions

**Fit models in a stepwise, interpretable sequence**
- Nuisance only
- Identity baseline
- Social space
- Semantics
- Combined 

**Stress test significant results**
- temporal robustness: exclude adjacent trial pairs
- leave-one-character-out generalization 
- negative controls


### results

maybe right hippocampus is the main region?
character binary vs. dummy doesnt seem to be make a big difference

## other pattern similarity analyses

These should also adhere to the general same principles
- simple, defensible choices shared across as many analyses as possible
- control for time especially
- stress-test w.r.t. other trivial variables 

**Dimension similarity** 

**Character self-similarity**
Show that this isn't just scene

**Character identity (boundary analysis)**
Show that this isn't just scene

**Character convergence**

**Character subspace overlap**
