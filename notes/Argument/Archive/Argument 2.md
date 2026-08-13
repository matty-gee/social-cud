TODO
- add more detailed code examples to chatgpt and claude


What are questions to ask of task representations, both on trial pattern and pattern similarity level:
- is it correlated with time? scene? dimension? etc 
	- remove lag-1 and lag-2 trial pairs
- does the relationship generalize across characters?
- does it vary across participants?
- does it relate to external social function and representations?
	- https://pubmed.ncbi.nlm.nih.gov/28521007/
	- ctq, social functioning or social networks
- does it help us make sense of memory or dots related effects?


**Semantic embedding analysis**

We represented the participant-specific choices as semantic vectors
We also computed the choice contrast vectors
	c_t = embed(chosen) - embed(unchosen)
We also computed a relationship-update representation
	s_t = c_t - mean(c_0:c_t-1)
	essentially direction of the choice-based update to a running mean of the previous choice contrasts
	"what I did now" - "what I normally do with this character"
	decision relative to the history with that character


A relationship is an evolving context: some integrated representation that changes as a function of choices and interactions 
History-conditioned contrast vectors can rotate because prior choices alter the context

HC and CD seem to represent these differently
- HCs appear to integrate current social decisions with an internally maintained model of the evolving relationship
- CD representations may rely more on currently available cues and less on temporally accumulated relationship context

if this is correct:
- should be subject specific
- should actually be about history: about the specific character, not the future, etc 
- differences should get bigger as more history is available
- CD differences should be stronger with more of cocaine use variables 


memory errors preserve the structure of the participant's internal relationship model
the number of these map-preserving errors correlates with the strength of the effect in right hippocampus 


model it this way...
neural ~ current + prior + update
can also add magnitude 
neural ~ current + prior + update_direction + update_magnitude 