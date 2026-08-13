Our core theoretical claims are something like:
1. People maintain evolving relationship-specific states in an abstract representational space
2. At each social event, immediate information is interpreted in light of the ongoing situation and those longer-term states

The core conceptual objects: 
- a space 
- a current representation = f(current situation, previous situations)

Given this, different decisions about what "current situation" and "previous situations" means represent substantive hypotheses that make predictions
- we should test as many of those predictions as is possible 

**Current situation**

- Both options combined: situation and available action space
- Choice: should contain both situation and action content, response meaning
- Choice difference: isolate the semantic update produced by choosing one response rather than the other, directional semantic update
- Can have immediate local context (i.e., some number of previous slides)


**Some options for how previous situations might be represented**

Defining the previous situations involve 3 choices at least 
1. What is represented from each past interaction?
2. Which past interactions matter now?
3. How are the selected past interactions combined into a state?

*no history*: memoryless 
The current interaction is represented independently of prior interactions with the character
The current options contain the relevant information; relationship history is unnecessary or not represented at this moment.

*recent history*
use K past interactions with that character 

*all-history prototype*
The participant maintains a general estimate of how they tend to interact with this character
Compute this with a running-mean over all previous interactions with that character

*all-history cumulative state*
Each choice moves the relationship through social space, and the current representation is the location reached by accumulating those movements
Compute this with a cumulative sum over all previous interactions with that character
Path-integration like

*Context-gated history*
The current interaction acts as query; previous interactions similar in some way to the current one are used 
This may require semantic representations to be meaningful 


**questions to answer**

1. Does history matter?
2. How much history matters?
3. Is history evidence or movement?
4. Is the history used context-dependent?
5. Does semantic embeddings help?



## the semantics of the task structure 

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


## the semantics of participant choices

the goal here is to see that the semantic information helps us understand something about their choices

*Can we build a semantic relationship state for each character?*

Accumulate the choice or the choice contrast vectors to predict dots 
- Does it do better then behavior?
- Are choices themselves or choice difference better?
- Does context help?


use an RSA-based approach: can we recover the geometry of subjective impressions better when using embeddings?


Possible robustness analyses: shuffle choice histories across participants, hold out entire characters, repeat with another embedding model, and test whether results depend on context length or embedding magnitude


