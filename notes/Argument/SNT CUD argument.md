
**TODO**

Basic approaches to take: start simple, find something robust, then enrich 
1. Discover simple features that differentiate CUD from HC, and/or high CTQ from low CTQ
	- Classifications using pattern similarity (trial-wise or averaged)
	- Temporal integration 
	- Between subject-similarity
2. Start from simple task-relevant features
	- Dimension similarity
	- Semantic representations
3. Use a high throughput RSA approach, including with semantic embedding features

## Results
### neural feature screening

This asks the question: which ROIs contain multivariate trial-level patterns that separates CD from CD, or high from low CTQ?

#### Screener methods

*fMRI features*

Each subject's decision-trial neural activity was modeled with least-squares-separate (LSS) with 24 motion regressors
For each ROI, trialwise beta patterns were turned into subject-level representational similarity vector
- drop 0 variance voxels
- Pearson correlations between patterns across trials 
- only take the upper-triangle pairs 
This gave us a feature matrix: X_roi = subjs x trial-pair similarities

*Identify ROIs with multivariate group differences*

For each trial-pair, predict similarity from dx, sex, fd and memory...
Collect the adjusted dx coefficients into one ROI-level difference vector 
Summarize the vector with RMS
Test if RMS is larger than expected with permutation testing of the subject-level DX labels 
If we get a result, it says: that ROIs trial-pair similarity structure differs between CD and HC, after covariates 
Not classification per se b/c it never predicts held-out subjects (no CV), no accuracy or AUC etc

*For ROIs with evidence of group differences, try to interpret*

First, basic structural questions
- is the difference just less or more similarity, or something more complicated?
- what is the trial-wise contribution? are a few trials driving the ROI effect, or is the effect being broadly distributed across the task
- is the difference a temporal autocorrelation effect? eg, CDs being less similar for nearby or far part trials
- is the difference about where in the task they are?
- show a time-adjusted heatmap: residualize out both time and task position

we correlate the difference vector with task-model RDVs to see if the differences were interpretable 

**Other possibilities**
Cross-validated classifications: standardize the similarities, use L2 logistic regression (fixed alpha=1) and test two targets: dx and ctq_high
- L2 b/c we have many more features than subjects
- other options, that make more complicated but later may want to consider:
	- select the regularization strength with an inner CV loop 
	- can also use an initial PCA step?


**Maybe a better approach?**
1. ROI screen: Which ROIs show multivariate CD-HC similarity differences? (or Classifier: Can those ROI similarity patterns predict dx out of sample?)
2. Basic decomposition: Is the effect global mean similarity, temporal lag, task position, or subject heterogeneity?
3. Feature importance: Which trial pairs or feature families contribute?
4. Reduced models: Do abstract/task-geometry features add predictive information beyond temporal structure?
#### Results

**Schaefer parcellation 100**

*CD*: generically less self-similar 
Across all ROIs that show differences between groups, CDs have lower trial-to-trial similarity than HCs, and more so for trials nearby in time 
- suggests less neural stability in CDs: representations drift/fluctuate more trial to trial than HCs
- could still represent task structure similarly - but likely with more variability
Does not seem to be trial-pair specific, or task-position drift, or driven by outliers
Effect seems to be more in higher-order regions than lower-order ones

This is consistent with there being lower SNR generally: is it just noisier generally?
TODO: get a trial-level estimate of tSNR on preprocessed images 
Does not seem to be character identity, dimension coding or a dimension asymmetry: the CD-HC difference is ~uniform across same-character vs different-character pairs, same-dimension vs cross-dimension pairs, and affil vs power pairs
Looks like broad noise/difference
- this would be a hard story to sell...

*CTQ*

maybe a dlpfc affiliation-power asymmetry: affiliation trials seem to be more similar to each other than power trials 
within character, the effect survives 
does not seem to be driven by outliers 
but this region doesnt seem to represent dimension itself, and high ctq doesnt correlate with affiliation consistency or asymmetry in consistency - so it seems hard to interpret 



### generic temporal features

**temporal integration**
HC appears to have stronger temporal locality in neural pattern similarity: nearby trials resemble each other more than far-apart trials
CUD shows a weaker version of that structure
The brain pattern on trial t is less predictive of the brain pattern on nearby trials
so maybe there is a less temporally organized representational structure 
this could just be not paying attention or something - not necessarily interesting....

**between-subject similarity**

### task structure features

choice semantic average, choice semantic variability, dimension, character
+ + reaction time, time and time^2
HPC effects of choice average - something like tracking a general relationship state
PCC effects of choice variation - something like tracking relationship history
Both HPC and PCC represent character
Dimension is pretty much everywhere

Semantic content and character might be tracked less by CDs

#### dimension similarity

PCC, left angular gyrus, left striatum effects
- regression approach gives effectively the same exact map as the similarity contrast approach

![[0ac94cc3-2254-4010-998b-e00d5b3f1919.png]]

neurosynth decoding: what constructs this spatial pattern most resembles, based on large-scale meta-analytic co-activation patterns
here are some basic reults: 
- parietal network, default mode, theory of mind maps correlate most strongly - not surprising, given PCC effect 
- also spatial correlates relatively weakly: r = 0.149
- memory even weaker: r = 0.11

**dx effects**
no difference in dimension representation on dx that survives correction 

**ctq effects**
social trauma could plausibly affect the representation of dimension
but no difference... or interaction with ctq




### Semantic representations


**First-level RSA model**
On the first level, we control for time, time squared, RT, dimension, character and character decision number distances
- Time / time²: remove simple effects of temporal autocorrelation, drift,  narrative progression, etc 
- RT: partial out decision difficulty / engagement
- Dimension (affil/power): remove low-dimensional task structure
- Character: remove stable identity-related differences
- Character decision number: control relationship stage and available history

We model LLM-embedding-based semantic content predictors:
- Current choice: are neural patterns similar when the selected choices have similar semantic content?
- Current choice's difference from past: are neural patterns similar when the selected choices differ similarly from the previous choices within character?
	- this would show something beyond the current choice is represented 

**Second-level linear model**
On the second level, we control for sex, FD mean, post-task memory accuracy, CTQ score 
In general, we expect positive representational alignment effects


![[d7c25aed-b814-4ad9-8982-7c5af7e3ed30.png]]

dACC has interesting sequence-related effect
- CD may represent recent context more: more about what the most recent interaction was about?
- HC may represent longer run context more
seems like it might reflect differential integration of the past into the representation of the current choice/interaction 
- test if this is specific to choice trials 

PCC may represent the past that is being integrated (need to test this more)
If so, maybe we'd see different connectivity between dACC and PCC between DX, that could explain the lack of integration?


![[ad77dc11-642f-46a0-b4e0-c25f8fc4c2e8.png]]


![[0d2b0436-2cdf-4263-b5ea-8ad81e138ae9.png]]


**Meta-analysis**
Broadly, these regions may represent something like context or memory 
![[Screenshot 2026-08-05 at 12.55.27 AM.png]]

Liang et al. examined resting-state connectivity in 47 cocaine-dependent participants and 47 matched controls: cocaine dependence was associated with reduced connectivity between the salience network and the default-mode network

**MAYBE BUILD ON**


An interesting direction is to compare different models of relationship history
and then argue that CDs are less responsive to the history in each current choice (consistent with them being more cue-bound in general)
PCC: past context (slow)
- distinguishes past same-character from past other-character information but does not clearly distinguish past same-character from future same-character information
- That can occur if their neural patterns principally represent a smoothly evolving character-specific state. Nearby past and future trials sample similar locations on the same relationship trajectory, so both predict the neural geometry. The representation tells you “where this relationship is around this point” rather than “which direction caused the transition into this point.”
dACC: current situation in relationship to that past context (fast)

ideally I'd be able to incorporate HPC as a kind of content-addressable episodic retrieval 


**TODO**
Fit models with different amounts of prior history
- e.g., n-back: use at most n previous interactions
	- but at 1st trial, there isn't a history representation; so for now we should probably drop this... 
Every participant will have beta for each model
Then test whether CD and HC differ on this 


