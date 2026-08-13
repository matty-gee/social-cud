## basic empirical patterns

**Memory is positively correlated with average reaction time**
- Post-task memory is positively related to mean reaction time
- Replicates in online sample 

**Reaction time tends to increase over the course of the task in in-person samples**
- Subject-level regressions: we find a trialwise onset-RT relationship
	- log(RT) ~ decision_num, with OLS or robust regression
	- removed trials where rt = 0 b/c of non-responses
	- Control variables: character identity (dummy), dimension (dummy), word count sum (sum across options), word count diff. (absolute difference between options), and semantic similarity between choice options
		- rules out: “later trials are longer accidentally”, “power trials are harder” “one character drives it,” etc.
		- also: semantic similarity between choices doesn't relate to RT, which was surprising to me
		- does not seem to be the local stimulis 
- Replicates in Tavares sample, but not in the online sample 
	- online sample has no time pressure; removing outliers, trials > 12s, subjects with long avg. RTs does not change the effect 

**The increase in reaction time over the task is associated with better memory**
- Group-level regression: find a positive group-level memory-slope relationship 
	- Control variables:  sex, DX, ctq_score, age_years
- Basically holds for individual characters too
	- even when controlling for overall reaction time mean: what survives is structured, not nonspecific -> so this seems unlikely to be about global fatigue
		- the effect gets weaker but the direction is stable across characters and samples 
- Replicates in online sample

**For it to be some kind of fatigue effect, it has to**
- Vary within subject
- Be related to attention to the task, b/c it predicts later memory
- Be character-specific: affect memory for that same character, independently of overall speed


**So...why?**
This is consistent with increasing model-based integration: as characters accrue history, each new decision requires retrieving and integrating more
Long RTs might be querying memory, simulating future eg via HPC-PFC mechanisms
Maybe: we get a greater slowing when things matter more?

![[100653e0-c42d-4372-a885-6a382ccd4410.png]]

## literature support?

The "intermediate effect" is like this
The Knowledge Encapsulation Theory
Could be like U-shaped relationship ultimately... probably if they did another day of the task, they would make faster decisions again
The "fan" effect 