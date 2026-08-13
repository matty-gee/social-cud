TODO
- still follow up on the choice-contrast models

## High-level goals for better representation

Should have a clear conceptual meaning and mathematical definition
- What latent quantity it is supposed to measure (some precise computational interpretation)
	- task structure
	- choice or options: e.g., shuffle choices
	- relationship state: e.g., predict future
- On what level it varies: trial, participant, character, or participant-by-character

Shouldn't just be task structure: control time, character and scene-related structure
- Should survive lag-1 and lag-2 dropped trial pair analyses
- Should generalize across characters

It should make predictions about some task-external constraint
- better correlations to subjective placements
- memory and subjective placement confusions
- RT-memory relationships
- post-task internet sample's free responses and ratings
- real-world social behavior: LSAS, SNI, etc 
- in neural data
	- HPC: current event or update
	- PCC: accumulated state 

If semantic embeddings, it should have interpretable semantic content
- affiliation-power + other content
- compare the components to ratings, free responses and more
- ensure its not just character names, etc 
- in neural data: positive/negative valence may elicit effects in emotion-related regions, etc 

## Problems with current analyses

*Distance from POV*
distance is mainly about affiliation: the coordinate system starts with a large affiliation value

*Angle and distance from POV*
a lot of both explained by time and character: roughly half of the trialwise variance in distance is predictable from character identity and temporal structure (onset time and character familiarity)
this is somewhat expected by design b/c it is an accumulate relationship state - but it does make its interpretation trickier 

![[2e35467f-1656-4cdf-b6af-9a1780d29dbb.png]]


## Analyzing the semantic features of the task

We want specific embedding-derived variables to predict independent behavioral and neural outcomes

We want something like current context v. relationship context/history-dependence
Eventually, I want something like:
- “what happened” → high‑dimensional event trace: a naturalistic narrative produces high-dimensional event representations
	- something like an episodic history, the list of specific events that happened
- “what it meant” → low‑dimensional relational update: the brain extracts a task-relevant social transformation (affiliation/power direction),  integrates it over character-specific history,  and (maybe) uses that evolving state to guide decisions 
	- maybe: “what it meant” is not a vector; it is something like a transformation of the current interaction representation relative to relationship history
the goal is to separate out what is a generic narrative change v. a socially meaningful update

### Task structure

Task features should be:
1. readable from the embedding
2. generalizable across time and character
3. stable to small changes (e.g. to wording or LLM)
4. (Ideally) predictive of something else

#### Affiliation-power
We can classify affiliation v. power 
- This does not show a social map, or signed updates etc... it is a stimulus audit of task structure
This generalizes across time and characters

Find the separating axis & project individual choices onto it

*Classify positive v. negative change on both dimensions*

Residualize: what other features are there?
#### Character identity 

### Choice structure 

Averaging the choice embeddings within character-relationship leads to representations that are HIGHLY similar across participants, reflecting shared/task structure 

We want choice representations to vary more across participants, so that they reflect their specific choice histories with each character and not shared/task structure

*Choice difference embeddings*
normalize(embed(chosen text) - embed(unchosen text))
I then just average these within character, concatenate and then compare across participants 
without any context, this alone reduces the average across participant similarity from ~= 0.95 to ~= 0.20

*Choice difference embeddings with various amounts of context* 
I add various amounts of previous context to the embedding step, from choice contrast only, to the local scene context, to accumulating relationship history:
- current choice text only
- previous slide text + current choice text
- previous slide text + current choice text + last 1 previous same-character choice text
- etc...
Adding more context decreases similarity across participants a little bit more
- probably b/c if context is mostly additive, it just cancels out in the difference calculation

![[7621eb55-64b0-4aa7-8e53-5e79beb45ad1.png]]


*Running mean and variance of choices* 
- running mean = current prototype / relationship estimate
- running variance = how consistent or heterogeneous the relationship history is
- current choice minus running mean = update or surprise or steering 


## Possible new analyses

**TODOs**
- Frame in a way similar to the temporal receptive window literature (Hasson, Chen, Baldassano)
	- if the deficit is specific to intermediate/long-timescale regions and spares early sensory ones, that's a much sharper claim than dACC alone

**We want to model character-specific relationship histories**
We expect something like:
- A character’s return cues a character-specific relational state
	- same-character > other-character
- The current choice is represented relative to that state
- The choice then updates the state

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

**We see an interesting disassociation in dACC**
- dACC may use past information to regulate present action
- In Tavares, we interpreted an effect in this region as potentially contributing to choice selection or action
- This dACC effect may overlap with craving-related cluster from neurosynth
- A simple global SNR explanation is already implausible: worse CD data quality should weaken both representations, not strengthen current-choice coding while weakening history-relative coding within the same region
- 
![[35f69b3a-eb56-47ea-8abf-4c39cc356e16.png]]


**There are at least 3 follow-up questions**
- Is the effect character-specific?
- Does the actual ordered past matter?
- How long is the historical reference?


**We ran same-character and past/future controls**

For each trial, we represent the chosen option’s meaning with an embedding, then calculate three matched “updates”:
- Past-same: current choice minus the previous choice for the same character
- Future-same: current choice minus the next choice for the same character
- Past-other: current choice minus a matched previous choice for another character

Does past same fit the neural data better than these other two predictors? 
- dACC shows the specificity most clearly
- HPC and PCC distinguish same-character from other-character but not past from future
	- maybe a smoothly evolving character-specific state
	- but an RSA based on relative vector geometry can be partly insensitive to whether a transition points forward or backward in time: Past and future predictors can resemble one another even when the underlying directed transitions differ..... TODO think this through 

**Then we ran a history timescale analysis**

We varied how much the past history contributes to the current trial's representations 
CD dACC representations are preferentially aligned with recent relationship context and cease to align with broadly accumulated history
This explains:
- No group difference at one-back
- Strong difference in accumulated history
CD does not fail to use history, but fails to maintain or use broader relationship context

*We also tried a history amount v. coherence version*
We tested whether the effect is due to history vector norm (coherence) and the effect remains 
The result was not caused by the fact that broad histories can have smaller norms when prior choices are semantically inconsistent.... Instead, it appears to concern the direction or content of the relationship history




**Then we correlated this with other task-external variables**



**Emerging plausible story**
When a character appears, the system reinstates a latent context that is a function of previous interactions with that character 
- Hippocampus and PCC are biologically plausible components of such a system
- We can think of this as the "social location" of the character 
The current decision is interpreted through that state: the current choice is not represented in isolation; its meaning depends on the relationship state
The choice may update the relationship state

So something like: social decisions are represented according to their immediate semantic content, conditioned by a character-specific relational context. Hippocampus and PCC may contribute a persistent relationship or situation state, while dACC may use causally available relationship history to construct or select the meaning of the current action


**Other possible things**

| Analysis                                | Central question                                                                                     | Strong result would mean                                                |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Same vs matched-other vs global history | Is the reference point genuinely bound to the relevant character?                                    | Same-character accumulated history explains unique dACC geometry        |
| Disjoint lag decomposition              | Does the effect reflect remote history, rather than a smoother average containing the recent choice? | HC uniquely represents middle/remote history; CUD is dominated by lag 1 |
| Smoothing-matched null                  | Could long-window effects arise from overlapping windows and temporal smoothness?                    | Real histories outperform shuffled histories with identical smoothing   |
| Semantic vs behavioral decomposition    | Do embeddings add information beyond affiliation/power choices and locations?                        | Semantic update geometry survives exact behavioral analogues            |
| Context/lexical validation              | Is the effect semantic, or driven by names and repeated wording?                                     | It replicates across encoders and survives lexical/name controls        |
| Direction vs magnitude                  | Is dACC encoding update content or merely novelty/conflict?                                          | Update direction survives update magnitude and history coherence        |
|                                         |                                                                                                      |                                                                         |
|                                         |                                                                                                      |                                                                         |

## Ideas other new analyses

Try to explain interesting features of the post-task data 
- subjective-behavior alignment
- subjective location & memory confusion as a function of behavioral location
- 