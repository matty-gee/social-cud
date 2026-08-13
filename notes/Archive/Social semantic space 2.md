TODO
- clarify some concepts
	- choice v. direction: choice is the actual text and direction is a hypothesis about how that should update a representation
- clean template files per task-version and code to convert into subject-specific narratives
- get basic features per subject-specific narrative: embeddings, ratings, summaries, etc

**conceptual framework**

our core claim is that relationship is represented in an abstract space and that it is the accumulation of individual interactions that can be measured via choices in those interactions
relationship_state = cumsum(relationship_updates)
so the cumsum(relationship_updates) is something like the interaction history
so there are 2 basic things we want to represent
- who this character is (their role, history, stable properties)
- what this interaction changes (the update/change reflected/produced by the current choice)
then we can treat fMRI and the post-task measures as different readouts of that latent relationship representation

More generally, I think how a person will represent the task is a mix of local context (e.g., the current interaction; "What is happening right now?"), medium context (e.g., the scene; "What is going on in this episode?") and long-run context (e.g., the relationships they've developed; "Who is this person to me?"), i.e. representations that operate on different time-scales. 
current representation = f(immediate interaction(fast), scene/context(slower), character/relationship(slowest))
We might expect these to be represented at different levels of a cortical hierarchy
These should interact in some way too: the immediate interaction representation is a function of the current context and the relationship with the characters 
The core Q then is: "How is the current relational state constructed from interactions at multiple scales?"
We can try to then think of the cumulative affiliation and power choices as a interpretable low-dimensional approximation to this process

> **People maintain evolving relationship-specific states in an abstract representational space. At each social event, immediate information is interpreted in light of the ongoing scene and those longer-term states. This contextualized interpretation guides behavior and may update both what the person is believed to be like and what the relationship with that person has become.**

The cumulative affiliation–power model is then the simplest formal implementation of that general idea: a two-dimensional state with fixed, equally weighted transitions
We can think of this as a state-space with a transition rule

**Stages**
I want to first build a semantic atlas or dictionary of the task: use embeddings to build candidate representations, derive other interpretable features (e.g., using generative LLMs)
- establish what kinds of semantic representations can decode different features of the task
	- features
		- decision dimension
		- choice direction
		- character identity
		- character familiarity
		- social location 
	- test different embedding representations
		- choices
		- choice differences
		- previous trial context
		- previous relationship decisions context
	- Use semantic probes to 
		- show task structure
		- locate trials w.r.t. features: once a probe has been validated, every task item can be projected onto its learned feature direction
		- character previously unrecognized structure 
			- A linear probe is fundamentally supervised: it needs a feature or label defined in advance. It therefore does not, by itself, discover a completely unknown psychological construct. But we can use it to discover things by defining many probes and sweeping them across the task 
	- Eventually get: 
		- A trial-by-feature matrix revealing overlap and cross-loading among dimensions.
		- A similarity map showing which narrative and decision events are semantic neighbors.
		- A residual map identifying organization not explained by the original task model.
		- Candidate new dimensions derived from those residual associations and then independently validated.
Then ask what information participants seem to use to make choices: what features predict choice?
Then what information is sufficient to explain the current neural representation?
- Does history change current trial representation? e.g., does the brain represent current interaction or current interaction conditioned on relationship history 
	- Can do a subject RSA 
And what makes it into explicit post-task representations of the relationship: subjective placements and (in online participants) ratings and free-responses 

**core questions**

*how should previous experiences be integrated into my current representation?*
uniformly (importance=constant) => running-mean: what has happened?

more recent more (importance= f(time)) => recency-weighted: what happened recently?

similar episodes more (importance=similarity)=> similarity weighted: what happened that matters most now?
- this requires keeping the previous episodes in memory; the others can compress history into a recurrently updating state 
- have to decide what determines similarity

## embedding analyses
### test different representations

really think through the best way to represent this... I want a small set of primitives that I can test 
https://chatgpt.com/g/g-p-6993ecff06d48191bcd5f7dc20f7f674-social-computation/c/6a78cad8-5140-83ea-ae39-cc113123e86e

**Current choice** 
- Both options combined: situation and available action space
- Choice: should contain both situation and action content, response meaning
- Choice difference: isolate the semantic update produced by choosing one response rather than the other, directional semantic update

**Choice in context of history** 
- Accumulated context: character or relationship state 
	- compare different kinds: 
		- embedding full history vs. running mean of individual embeddings; choice-only context vs. full-narrative context 

**Warnings:**
- Be careful with identity and pronoun markers that might make certain features trivially correlated 
- Do not sum a sequence of embeddings that each already contains accumulated history naively


### task structure RSA decomposition results

![[Screenshot 2026-08-07 at 1.02.47 AM.png]]
### task structure decoding results

here I do not use individualized trajectories through the narrative
the goal is to get a sense of what task-structure information the embeddings encode (e.g., social dimension, choice direction, character identity) so that we can then build off of this to more individualized representations and analyses 
embedding everything v. using the running-mean doesn't make a big difference just yet....  but test this again when we go to subject-specific representations; also test whether incorporating some of the narrative context helps improve decoding 


*Dimension is decodable from semantic embeddings*

The current options/choice does best 
Choice difference does a poorer job: it subtracts out the information that's shared across options, which is the dimension! 
Previous context doesn't help either - which also makes sense: the dimension information is encoded in the local choice, not in the history


![[69ead8fa-bdcf-449c-9a97-5fbff15b8154.png]]

*Choice direction is also decodable*

can reliably decode the +1/-1 direction on each dimension

maybe some slight dimension differences in how previous history helps
- affiliation direction seems to benefit from context: more context-sensitive?
- power direction less so: more explicit locally?

![[531c14d0-3150-4712-9491-8c5598bb7cb4.png]]

![[c80b37ef-af53-4895-8258-efb8c5103d7c.png]]
to make any kind of inference from the history-related ones, I think I need to better match across the analyses?

4-way decoding benefits from choice-difference, at least when comparing across additional past context 
![[d148cc4c-3b83-4278-8763-c96192b51260.png]]

*We can decode character identity with additional context, even when masking names, etc*

One problem to keep in mind with character identity with the previous context decoding correlates all of the within-character trials by design - so decoding from them becomes trivial since we have made them strongly temporally autocorrelated 

So instead, I just focus on the representations that don't accumulate context

![[7743bc9f-ece9-48a5-be31-b105203fa176.png]]


#### Others 


*Can we map the embeddings directly onto affiliation and power?*

Basic analysis idea: 
1. Learn semantic affiliation and power axes using the task’s existing dimension and direction labels
2. Validate that these axes recover held-out trials and generalize to held-out characters
3. Test that these help decode something else 

### participant choice decoding results

*Can we build a semantic relationship state for each character?*

Accumulate the choice or the choice contrast vectors to predict dots 
- Does it do better then behavior?
- Are choices themselves or choice difference better?
- Does context help?


use an RSA-based approach: can we recover the geometry of subjective impressions better when using embeddings?


Possible robustness analyses: shuffle choice histories across participants, hold out entire characters, repeat with another embedding model, and test whether results depend on context length or embedding magnitude



## extracting interpretable features

Create subject-specific narratives and then use LLMs to encode features for each slide
Start with affiliation and power: allow basic prior locations, non-decision trial changes, and changes for multiple characters at once

Then: 
1. does this correlate with our original modeling?
2. does this improve correlations with dots? free-response embedding similarity? ratings?
	1. 
Start with the schema sample, then if it works to improve something expand it out to other online and in-person samples 

## applying semantic representations to fMRI data

