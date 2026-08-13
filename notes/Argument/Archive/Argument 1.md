
*maybe another useful framing: coordinate chart over memory...*
look into the differentiable lookup table idea: map the query onto a probability simplex, and the readout is the corresponding mixture of value vectors... could this help?

**Possible computational decomposition of the character-relationships**
- character identity (who)
- state (abstract where)
	- shared subspace or unique subspaces
- decision
	- e.g. difference between choices, prediction about outcomes etc 
- update dynamics (how state changes)


**Overall flow**
- experience unfolds as a sequence of rich, high-dimensional events. To support prediction, inference, and generalization, the brain must extract low-dimensional regularities from these events
- we map high-dimensional episodes into low-dimensional state variables that can be updated incrementally, understanding our lives as narratives
	- maybe narratives are sequences of events that afford latent-state inference and the abstraction of schemas from those sequences... something like this 
- social relationships can be understood in this way: interactions are high-dimensional, dependent and temporally extended - yet social behavior depends on having stable representations of other people, and how we relate to them; we should think of relationships not as collections of events, or as static snapshots, but as dynamic, latent states inferred from event histories 
	- relationships are inferred latent states, dynamically updated by interactions
	- the social relevance of interactions may be estimated by projecting the HD events onto a LD social subspace of affiliation and power
		- affiliation and power are two central dimensions of social life: evidence from human face perception, stereotypes, primates; interpretable as axes, rather than just categories 
		- we operationalize these two axes as the sources of the task-relevant variation 
	- a relationship is the accumulation of such events across time within a given person into a narrative
- this implies two separable operations: 
	- mapping each interaction from a high-dimensional event space into a low-dimensional social coordinate system (idk if this is the right frame or not... is it the mapping or the map per se )
	- accumulating these projected signals across time to infer a latent relationship state
- the HPC & PCC might do this (Tavares)
	- PCC
		- long temporal autocorrelation: slowly varying, history-dependent signals
		- distance-like?
		- maybe: integration of events across the relationship, so something like cumulative evidence
	- HPC
		- short temporal autocorrelation
		- direction-like?
		- maybe: normalized version of event integration within a relationship, so that relationships can be compared 
			- compressed sufficient statistic of the interaction history for the purpose of social comparison, inference, generalization (places people into a shared coordinate system)
	- can we reconcile this with e.g. Baldassano's stuff, where HPC seems to encode event boundaries?
- this framework offers a principled way to understand social dysfunction and psychopathology
- the quasi-naturalistic social navigation task, combined with LLM-based semantic representations of social events, provide a way to operationalize these computations

*People behave as if they are mapped others in a low-dimensional social space*
- Mapping score: people can retrieve or construct a map-like representation
- Memory distance-based confusability: abstract social geometry can interfere with episodic retrieval
- Reaction time & memory relationship - not sure if this helps yet....

*This mapping can be explained by simple computational principles applied to social relationships*
- Trials from different dimensions are represented differently, esp in PCC-R: trial pairs within same dimension are more similar than ones across dimensions
	- not character identity, familiarity, time, number of words, sentiment 
	- could be related to semantics - which it should be!
	- to improve
		- can't necessarily interpret as dimensions: they could just be clusters, right?
		- have to show its not RT
- Character identity seems robustly represented across regions: adjacent trial pairs that are the same character are more similar than trial pairs that are different ones 
	- should be orthogonal to dimension
	-  to improve
		- check relationship to scene
- Character-specific representations get more self-similar over time in PCC-R
	- I guess this is consistent with a kind of integrator? not really consistent with a cumulative sum though is it?
	-  to improve
		- make sure there's some cross-character null (I assume thats what it is rn?)
- HPC-R RSA: running mean > cumulative sum, and euclidean distance > cosine distance
	- The running mean has the benefit of being less correlated with time and with familiarity, by design 

Maybe we can think of memories as the weighted retrieval of episodes...? 

*Deviations arise from systematic alterations to those principles*
- We replicate the behavioral affiliation and power social avoidance effects 
- Age of onset in CUD is robustly associated with more negative affiliation and less separability of their affiliation and power trials 