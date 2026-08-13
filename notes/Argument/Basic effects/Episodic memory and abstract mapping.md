## Subjective locations are similar to behavioral locations

Both groups show rough alignment between their average behavioral location and post-task subjective placements of the characters
![[91a9a416-54d7-4ed6-a4cd-0ac689a8ed44.png]]
![[c37bf0a8-b439-4e7f-bd75-332c679c03f6.png]]
![[8bff822c-fe4b-4470-bb32-dc17d3247fc3.png]]

Mapping score measured in two different ways, both show in in-lab and online samples: 
1. better than chance alignment
2. correlation with memory 

![[976473f8-ce82-413a-a3d7-3863e4538743.png]]

### High memory → more faithful mapping from behavior to post-task subjective placements

Both groups (HC and CUD) seem to represent the locations from the task, more or less, as does the online sample 

We compute a mapping z-score is computed against a subject-specific permutation null
- gives ~same answers as using RSA

The mapping score positively correlates with memory in-lab and online
- Consistent with integrating a more stable internal representation of characters that integrates the interactions across contexts 
- Maybe not present in the online data...? test this more closely 


## Memory confusability as a window into the mapping
### basic empirical patterns

Characters closer in social space (smaller euclidean distance in behavior or dots) are more likely to be confused in post-task memory (confusion probabilities from memory questions)
Given that we see it in behavior and memory, it argues against dots or memory task introducing this... its prior to any such prompting 

Leaving the confusions directed or symmetrizing them doesn't change the result at all (they're effectively the same): suggests its about pairwise proximity, rather than any kind of directional effect 
- the asymmetry magnitude of a confusion matrix doesn't predict confusability on the subject-level either 

**It's not**
- an idiosyncrasy of the sample: replicates online
- temporal overlap in the narrative: survives a control for absolute difference in mean onset
- poor overall memory for task (would predict uniform confusions prob anyway)
- character-specific memory: no pattern w.r.t. character-specific memory either
- poor memory overall (MOCA)
- mapping (retrieval?) accuracy: no correlation with the mapping score
- area, minimum or average of subjective map distance 
- average RT (in either sample)
- IQ (in either sample), LSAS, SNI, ISEL, age at first cocaine use  
- memory test is before the subjective placements, so no way that placements -> memory test
- not driven by singe character-pair in leave on pair out analysis; the pattern of importances if pretty smooth and diffuse; the  pattern of character-pairs importances is similar across in-person and online 

Suggests: the pattern of memory confusions reflect the geometry of a subject’s social map, not just the timing of interactions or memory for them

**Stress test more**
Does it relate to post-task assessments from online sample?
Is it just gender or race? prob not  
Is it distance between characters, or distance to self?

### computational modeling

**Basically**
- The map formed by behavior defines stored memory locations
- Retrieval is noisy and uses a similarity-based completion rule
- Interference is inevitable and is structured by map geometry: distance predicts confusion 
- The key driver is local packing (nearest-neighbor structure), not global hull area 