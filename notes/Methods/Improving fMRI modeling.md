
We have to be careful in interpreting effects: we only want to interpret positive RSA effects (more similar/dissimilar in behavior, more similar/dissimilar in brain), so we should want either 1) a positive mean effect across all subjects, adjusting for covariates or 2) a positive effect in HCs, adjusting for covariates, to be able to interpret any differences of CUD or varying with CTQ
- Simplest way to do it is use code HC as 0 and CUD as 1 in dx, then intercept is HC effect and CUD-HC is the beta on the dx variable
	- HC shows positive effect: test the intercept beta as > 0
	- CUD is reduced v. HC: then test the dx beta < 0 

We need to show that these effects aren't explainable by time alone
- should start from the robust control, to establish that an effect will more or less survive this correction

**Addressing issues with confounding**

For any trajectory-like analysis, it's important to show that history matters in a space-like way more than a time-like way: it's not just time since last interaction, or the number of interactions, but instead the specific interactions and how they change location in abstract social space, that determines the representation of the character
At minimum, always test against a within-character shuffled choice null: this is fairly rigorous I think, controlling for most possible confounds 
We can also get around some of the time-related issues include character-wise comparisons and inter-subject analyses (like RSA) that control for task structure 

At every step, I want to be rigorous and ensure that any result that I think I get is robust to obvious confounds 
So, on subject-level should always control for time at minimum
On group-level, always control for sex, memory_mean and fd_mean

*Nulls to use across fMRI analyses*
some sort of mean baseline 

*Shuffled-choice (history-destroying) null* rules out explanations based on time, drift, or static character effects
This permutes decisions within character and recomputes cumulative social trajectories, which preserves:
- character identity
- character-specific marginal distributions of choices
- end location and overall trajectory length per character
	- test how much the average location changes, and whether this matters...
while destroying ordered interaction history



**SNT framework**

*Does correlations between affiliation and power choices imply non-orthogonality?*
Not necessarily
It could be a preference or with some other dimension (e.g., approach/avoid) that influences both kinds of decisions: it's only necessarily about the trajectory the participant takes 
Non-orthogonality is a statement about neural basis vectors of the representation in latent state space

**Subjective placements**
We see reasonable correlations between the behavioral locations and the subjective post-task placements 
But there are also noticeable differences
Can we improve the correspondence?

*Shared translation*
1. estimated a 2D offset vector by matching the overall mean location of the subjective map to the overall mean location of the behavioral map
2. applied that same offset to all subjective coordinates and re-scored correspondence
We tested whether it improved the correspondence e.g. using a split-half cross-validation
the correction actually worsened the correspondence!

Tried subject-specific translations too... and it didnt seem to help either...
not necessarily any easy transforms that can be done here, given the small number of points

### SNR

tSNR must be computed from the preprocessed BOLD volumes, before any voxelwise standardization

### 2D egocentric distance is essentially just affiliation

![[4b18e760-31f8-4abe-9803-053a66de81be.png]]
### Temporal confounding

*Problem: temporal correlation*
Temporal confounding is a major problem
- various kinds: not just linear or non-linear progression of time, but scene structure, etc 

**theres at least a few notions of temporal structure that could confound what we want to estimate**
- local autocorrelation/lag: nearby trials could be more similar due to temporal dependence in fMRI signals and behavioral trajectories
	- trial-wise:  functions of pairwise lag, e.g. with low-order polynomials
	- character-wise: summarize how interleaved two characters are in time, e.g. mean cross-character difference in time across their trials
- absolute time structure/midpoint: distances can vary systematically as a function of where in the task the trials are 
	- trial-wise: functions of pairwise midpoint time
	- character-wise: include differences in mean onset
- clustering/boundary: discrete segments can create step-like context shifts 
	- trial-wise: include within-scene indicators
	- character-wise: include overlap metrics e.g. the proportion of cross-character trial pairs that fall within the same scene
- character identity + time (even something maybe theoretically interesting like familiarity) could create the false impression of a character trajectory effect where there is none

*Solution: relax some social space assumptions and build up from the ground*

I tried to improve the modeling to address some of this but the original findings did not survive these controls
We should relax the assumptions of the original social-space model: choices are represented as language-model embeddings and accumulated into semantic trajectories, providing a richer and less theory-constrained description of social interaction


### Ensuring across-subject variation

*Problem: minimal across-subject variation*
Raw chosen-option embeddings should results across participants but are almost completely shared across participants
This is likely to be task structure - which may be interesting but doesn't make use of the choice-related variation

*Solution: choice contrasts*
But choice contrast representations contain substantial individual variation superimposed on a moderate common task component (r ~= 0.06)
- adding local context or prior choices does not increase the across-subject correlation
- BUT running mean is highly correlated across participants (r ~= 0.5)
	- likely smooths out trial-specific choice differences 

Choice contrasts ask: in what semantic direction did the participant choose to move, relative to the alternative option?
embed(chosen) - embed(unchosen)
we normalize these to discard magnitude 

How to use these choice contrasts?
the local choice is just about that; history-conditioned contrasts can rotate because prior choices alter the context

We can classify affiliation v. power from the choice embeddings themselves  or the choice contrasts
The decoding doesn't differ as a function of DX or CTQ





