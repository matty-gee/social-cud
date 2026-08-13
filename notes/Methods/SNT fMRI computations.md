## High-level framing

Social relationships evolve over time, social situations are inherently high-dimensional, dynamic, open-ended
There are different possible stories that roughly fit with this
- We must balance abstraction (map-like summary) and episodic specificity (memory for specific events)
- To make most flexible use of memories in , should store episodes and abstract over them as needed

The goal is to the show some meaningful change in the neural patterns from the choices participants make, ideally in "social space"

HPC and PCC are promising neural targets
- Hippocampus: fast, relational, episodic, state
	- angle, running mean of locations, running mean of the semantic embedding of choice 
- PCC / precuneus: slow, integrative, history
	- distance, running variability of locations, running cosine distance of the semantic embedding of choice 

I also want to explore using semantic embeddings of the task narrative and of the choices as rich features that can relax some of the social space assumptions 

Confounds are an issue, so we must always stress test results w.r.t.: 
- temporal drift
- temporal proximity
- task position
- character identity
- familiarity


## fMRI exploration

Use pattern similarity mainly...

**RDV definitions**
*Task structure*
dimension: binary distance between decision dimensions (0=same, 1=different)
scene_num: absolute difference in scene number 
char_1, ..., char_5: same-character identity distances (0=same, 1=different)
character_decision_num: absolute distance in within-character decision number

*Choice-related*
choice: cosine distance between choice vectors ([affiliation_decision, power_decision])
semantic_choice: cosine distance between choice embeddings 

*Social and semantic space*
location: euclidean distance between cumulative character-specific affiliation/power locations, after the current choice  ([affil_coord, power_coord])

location_runmean: euclidean distance between running mean of character-specific affiliation/power states, after the current choice ([affil_runmean, power_runmean])
location_dispersion: absolute difference in the cosine distance of character-specific running cosine distance in choice vectors, after the current choice

semantic_runmean: cosine distance between character-specific running mean of choice embeddings, after the current choice
semantic_dispersion: absolute difference between character-specific running cosine distance in choice embeddings, after the current choice

*Nuisance covariates*
time = absolute temporal distance between onsets
time_sq = squared temporal distance between onsets
reaction_time = absolute difference between reaction times

**Temporal autocorrelation**

LSA seems to capture it, LSS seems to suppress it 
## Computational modeling

*During social interactions, do something like attention over memory*
- maintain a global episodic memory of relationship, maybe especially some subjective representation of the memory 
	- the more history the richer the set of memories/representations to use 
- the current options on how to interact define a subspace in semantic space, where the difference (n-1 dimensional) subspace tells me the possible directions of relationship change from the choice
	- in SNT this is like the affiliation or power axis changes implied 
	- Maybe this a key into memory store?
- retrieve a prototype from memory as a similarity-weighted convex combination of semantically relevant past events
	- in some sense this is as much reconstruction as recall 
	- this approximates the optimal predictor when the transition function is unknown but smooth in semantic space (there is a connection to kernel regression)
- update that recalled/reconstructed memory using the choice 

### Sufficient statistics for social space geometry

Basic questions: What features would help (e.g., state, history)? What numerical quantities can estimate them (e.g., mean, variance)? How can we compute those (e.g., oja's rule)? Can this explain both social navigation end e.g. Baldassano's stuff?

First: build very simple estimatores that capture the two main statistics for social space
- Do not put them into any realistic algorithm - just understand how the statistics relate to each other, what they give us, etc 
- what can we practically differentiate in our setting?

For each proposed set of statistics, check:
- social avoidance correlation
- mapping score, correlation with memory
- memory confusability (with behavior only)
- HPC and PCC pattern correlations
- maybe these could be signals to optimize....

This can later be extended with LLMs: e.g., the event vectors projected into social (sub)space to get the social meaning of the choice, etc... 

Each trial yields a unit choice vector $\mathbf{c}_t \in \mathbb{R}^2$ (affiliation × power)
So these are samples from a distribution on the circle
This gives us a von Mises-Fisher distribution on directions and a circular state-space model
#### Mean relationship estimate (vector)

The running mean $\mathbf{m}_t$ summarizes past choices in this space.
$\mathbf{m}_t$ is a first-moment (center-of-mass) statistic of observed behavior.
We can interpret this as the current relationship location in social space.

The learning rate $\alpha_t$ sets the temporal scale:
- $\alpha_t = 1/t$: exact average over all past trials
- constant $\alpha$: recency-weighted average over recent trials
	- exponential forgetting: better for tracking nonstationary partners 
- either way: early trials contribute more! so we can interpret this as a kind of learning that stabilizes

$$\mathbf{m}_t = \mathbf{m}_{t-1} + \alpha_t(\mathbf{c}_t - \mathbf{m}_{t-1})$$
or equivalently:
$$\mathbf{m}_t = (1-\alpha_t)\mathbf{m}_{t-1} + \alpha_t\mathbf{c}_t $$

this defines a point inside the unit ball: it at best can have magnitude = 1
#### Update / mismatch (vector)

Each new choice is compared to the current mean and the difference is the update (mismatch) vector: $$\boldsymbol{\delta}_t = \mathbf{c}_t - \mathbf{m}_{t-1}$$
$\boldsymbol{\delta}_t$ points in the direction the relationship estimate needs to move
This is a tangent vector pointing towards the new observation

We can decompose this into a direction and magnitude:
- Direction of $\boldsymbol{\delta}_t$: how the new choice differs from the current estimate.
- Magnitude $||\boldsymbol{\delta}_t||$: how unexpected the choice is.

The mean update is a scaled version of this vector: $$\Delta \mathbf{m}_t = \alpha_t\boldsymbol{\delta}_t$$Magnitude of the update $||\Delta \mathbf{m}_t||$ scales with both "surprise" and learning rate.

This is not really interpretable as an error, given the subject chooses it; it is more of a steering direction - an action that drives state dynamics 
Maybe can still be an error-related signal in that the goal is to minimize the error signal, i.e. to stabilize the relationship, and maybe in some desired end location
If so, then we might think of choice as projecting the options into some space, where they get compared against a goal location, and the one that minimizes distance is the one that gets selected 
#### Concentration / consistency (scalar)

The magnitude of the mean vector, $$R_t = ||\mathbf{m}_t|| \in [0,1],$$ summarizes choice consistency.

High $R_t$: choices align in one direction (stable relationship).
Low $R_t$: choices cancel out (ambiguous or variable relationship).

*Dispersion can be expressed as:*
$1 - R_t$ (simple, interpretable), or
$1 - R_t^2$ (matches squared-deviation intuition for unit vectors).

#### Predictions from this

**Delta update implies a lag dependence/effective window**

**big social state change should be associated with big neural state change**

**How could populations of neurons compute this?**
Need to compute the mean location via some delta-driven learning
Could use Oja's rule to compute something like this
PCC could track a slower moving summary of the events 
#### Some other things to consider

Maybe instead we track the top eigenpair of the history, which I think could give different predictions - at least when the history is not conical... if it is, probably very close 
so I should try that at some point....
### Other behavioral features

**Choice biases**
People have biases
- non-uniform distribution of {A+, A-, P+, P-}
- MRL increases over time (basically the consistency effect)
The variability in choices in either dimension doesn't seem to change early versus late, within character
Suggests people are already consistent early in the task: maybe they adopt stable sign preferences for each character and express them consistently throughout the task

**participants are more or less similar in how they interact with characters**

shared ordering in character space at the population level
but still a lot of between subject variability too I think
some subjects act differently with the characters; others seem to have a near-global interaction policy

![[Screenshot 2026-02-04 at 1.01.53 AM.png]]


### Memory, mapping and reaction time

#### Mapping happens
Mapping score: people have access to the maps and they roughly match what we model in behavior
Post-task subjective placements are close to average behavioral locations
- Think of this as a a cued readout of the latent map at the end of the task, expressed in an explicit coordinate system
	- Query = “Where is this person?”
	- Retrieval
	- Output = placement

#### The map grows over events

Reaction time slowing over the task predicts later event memory, including for individual characters 
- Not consistent with standard evidence-accumulation models where beliefs stabilize etc 
- Maybe: RT increases because retrieval gets harder as memory grows / becomes competitive
	- to the extent that this is real, this is a big constraint - I think most models would predict that RT gets faster with more evidence....
Probably need a competitive retrieval model of some sort to explain this 

 Increasing RT is associated with better memory: maybe, have memories to search over to produce choices later in task
 - target: number of iterations to convergence (or some similar measure) should increase, e.g. b/c of retrieval competition
	
#### Event memory and abstract mapping are related
Correlation between strength of mapping score and episodic retrieval

Memory confusability: distance-based interference into memory retrieval, suggests episodic memory and map construction act on the same objects
	- target: retrieval interference should scale with map distance
Closeness in (both behavioral and subjective) maps predicts confusability in later event memory
- Confusability is stronger for subjective maps - which makes sense if they are both being retrieved from behavior: memory for events is what is available, not which actions are taken 
- Think of it as: confusability depends on distance because nearby characters have overlapping episode neighborhoods
Suggests these both operate on the same representations

### Hippocampus and PCC


**HPC**
- CA3: content-addressable pattern-completion: retrieving stored states from partial cues via competitive dynamics
	- can explain some kind of path dependence: earlier episodes can bias the interpretation of later ambiguous interactions 
- during naturalistic viewing, changes activity at event boundaries (Baldassano)
- right HPC = spatial, left HPC = semantic?
	- maybe wed see the characters being more comparable in the right and the individual histories being more represent in the left? 

**PCC**
- during naturalistic viewing, it maintains a stable activity pattern that persists through contexts (across individual events; Baldassano)

Biologically plausible models such as complementary learning systems propose exactly this: the hippocampus stores detailed episodes and performs fast pattern separation and completion, while the cortex (including PCC as part of a wider network) gradually absorbs the consistent features across those episodes into a stable schema

DOUBLE CHECK THESE....
**HPC: something like normalized state**
- less self-similar over time (faster changing)
- correlated with character-specific running mean of affiliation and power
	- TODO: test that this is actually character-specific 

**PCC: something like relationship history**
- more self-similar over time (slower changing)
- correlated with semantic dimension 
- more self-similar character representations over the task 

Can we find something in the post task that the PCC-like variable explains?

### Semantic features

Somehow incorporate LLMs further specify claims

**Start with semantic embeddings of the narrative**
- Amount of prior context to include
- Relationship history as accumulating
- Scene structure as providing boundaries
- Difference of current trial from previous
	- maybe gives boundaries etc naturally
- Choice trial representation: 2 options as difference vectors?
Start with simplest possible and then build it systematically

### Psychiatric-related dysfunction

Social avoidance is associated with making less affiliative choices, more submissive choices
- also smaller real-world social networks
- A biased control policy in social space...?

## Results

**choice embedding running mean and cosine distance are task structure**
highly correlated across Ps
seem to partially reflect the character ID and familiarity (or early v late), respectively... or maybe some mix of both in the two regressors 

![[881ef69a-6871-4fd9-be29-9674f12d6e5e.png]]

![[7fe7c913-212e-42d4-87f9-3054febe6b86.png]]


show effects in HPC and PCC respectively, which makes sense
want to build on this somehow: show character-related effects, then show that choices affect those representations?


**first principles decomposition**
across participant (shared) v. within participant (specific) structure
- whats specific should relate to choices
local v. long range information
general v. character-specific 

![[Screenshot 2026-05-16 at 4.15.06 PM.png]]

*think about what kind of generative model this implies*
trial  = shared narrative + character + local info + ...

*how do region patterns vary over time?*


*which regions' patterns are shared v vary across participants?*
LOO, with shared neural = average neural rdv from other subjects
- cortical ROIs (e.g. PCC & dACC) seem to be more similar in geometry across subjects 
- shared variance is relatively small: there is a lot to explain beyond this 

![[3b333a23-fbe1-410f-8eeb-ec0aea7215c7.png]]

then add in the basic task & nuisance variables (time, dimension, scene, character, etc)

**Representational choices**

how should choices be represented? the choice itself, or the difference from previous? 
how should they be accumulated? sum or mean?

**Building a regression RSA**

*Before fitting:* 
- Can the predictors be interpreted separately?
- Are they separable from time, RT, and character identity?
- How much variance do they have within and across subjects?

*fitting:* neural_rdv ~ 1 + nuisance_rdvs + design_rdvs 
- fit a model ladder: do the regressors improve fit? what do individual regressors add?
- ensure there is a plausible computational story 

*after fitting: stress test*
- lag sensitivity analysis 
- leave one character out sensitivity analysis

**Establish some basic empirical constraints with a simple computational story**

PCC = accumulated history = what has the history with this person been like over time?
- allows for a more stable person representation
- should represent the history at least, and maybe the current state too, especially post-decision after the state is updated 

HPC = current normalized relational state = where is this person now in a comparable map? 
- allows people to be compared in a common social-coordinate system
- updates representations at meaningful transitions (like choices; e.g., Baldassano's stuff)
- if dysfunctional: maybe harder to compare people in social space, which would reduce post-task map-like effects

HPC might use the history of PCC to do this 
- BUT: crude RSA functional connectivity and beta correlation analyses don't show the correlation I'd expect 

## Current problems 

**Semantic features are mostly task** Need to find a way to get regressors that are more than just task structure...

The naive semantic regressors are highly similar across participants right now
Raw semantic RSA shows shared task/narrative coding
- but, even after nuisance regressors, seems to represent trial identity

*to test for individualized representations*
for each subject, compare fit of own choice RDVs vs other participants social RDVs
identify high-disagreement trials to run RSA on
choice-residuals: subtract out consensus choice structure 
between-subject RSA in some way 
Shared response modeling and then analyze residuals 