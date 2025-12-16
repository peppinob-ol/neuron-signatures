Method

The experimental method consists of three main stages. First, the attribution graph is generated via Anthropic’s open-source Circuit Tracing library, mediated by Neuronpedia’s API, selecting a subset of nodes according to a defined Cumulative Influence Threshold. Second, a set of representative probe prompts is generated to isolate and characterize semantic concepts, and their feature activations are measured using Neuronpedia’s feature activation for text​ method. Third, a feature-engineering stage classifies each feature according to its role within the graph through heuristic thresholds on cross-prompt behavioral signatures, grouping features with matching semantic labels into supernodes. Fourth, the resulting supernode subgraph is then visualized within Neuronpedia to support interpretable causal analysis of model behavior. We dive into relevant sub-steps:

 

1 Attribution Graph
1.1 Attribution Graph Generation
The attribution graph is generated via Anthropic’s open-source Circuit Tracing library, mediated by Neuronpedia’s API. Attribution graph JSON represents causal computation flow from input tokens → features → target logit, with influence scores measuring each feature’s contribution to the prediction.

Component	Content	Description
Metadata	prompt_tokens, model, transcoder_set	Prompt, model ID, SAE configuration
Nodes	[{node_id, layer, feature, ctx_idx, activation, influence}]	Features (4,000-6,000 after pruning) with layer position, activation strength over each token, and influence score
Links	[{source, target, weight}]	Directed edges showing attribution flow between features
Thresholds	node_threshold: 0.8, edge_threshold: 0.85	Pruning parameters for graph sparsification
 

1.2 Filtering node subset 

The classic Cumulative Influence Threshold used in Circuit Tracing paper is used here to choose the subset of nodes to interpretate. The streamlit interface offers convenient visual feedback of influence of single nodes and numeric indicators to choose a convenient threshold.

 

2 Probe prompts
2.1 Concept Hypothesis Generation
From the seed prompt, an instructed language model can generate candidate concepts with short labels and contextual descriptions. For example, given:

Seed prompt: “The capital of the state containing Dallas is Austin”

the language model extract concepts such as:

Dallas (entity): A major city in Texas, USA
Texas (entity): A state in the southwestern United States
capital (relationship): The primary administrative city of a political region
containing (relationship): Geographic containment relationship
Austin (entity): The capital city of Texas
These concepts become testable hypotheses about what different features in the circuit might represent.

 

2.2 Probe Prompt Activations
For each accepted concept, we create probe prompts that maintain syntactic structure while varying semantic content:

Example probe prompt:

entity: A city in Texas, USA is Dallas.

This prompt:

follows the same syntactic template as the seed prompt
Places the concept token in a semantically appropriate context
includes the same functional tokens (e.g., “is”) to maintain positional structure
Feature activations over probe prompts are measured using Neuronpedia API’s feature activation for text​ method. I recreated it with a batch logic in a colab notebook environment to avoid API rate limits[2]. 

 

2.3. Cross-Prompt Activation Signatures

Feature 20-clt-hp:74108 (Say "Capital"): Activates on functional "is" across diverse semantic contexts. Consistent peak before output targets indicates procedural output promotion role, not semantic detection. Max peak token (63.31) before "capital".
For each feature, we measure how its activation pattern changes across probe prompts. This creates a behavioral signature that distinguishes functionally different features even when they have similar decoder vectors. For each feature × concept pair, we compute:

Per-probe metrics (computed on each individual probe prompt):

Cosine similarity: how similar is the feature’s activation pattern on the probe prompt to its pattern on the original prompt.
Robust z-score (IQR-based): how much does activation deviate from baseline, using outlier-resistant statistics.
Peak token: which token has maximum activation (excluding BOS).
Activation density: what fraction of tokens exceed the activation threshold (75th percentile).
Sparsity ratio: (peak_activation - mean_activation) / peak_activation—measures concentration
High sparsity (~1.0): Feature activates sharply on one token
Low sparsity (~0): Feature activates diffusely across many tokens
Aggregated cross-probe metrics (computed across all probes for a feature):

Peak consistency: across all probe prompts containing a concept, how often does the peak fall on the concept token.
Number of distinct peaks: how many different token types does the feature peak on across probes.
Functional vs. Semantic percentage: what percentage of probe prompts does the feature peak on functional tokens (e.g., “is”, “the”) vs. semantic tokens (e.g., “Dallas”, “capital”).
Semantic confidence: peak_consistency when the feature peaks on semantic tokens
Functional confidence: peak_consistency when the feature peaks on functional tokens
These metrics compare the same feature across different prompts. A feature that consistently peaks on “Dallas” tokens across varied contexts (high peak_consistency, low n_distinct_peaks) behaves differently from one that peaks on different tokens depending on context—even if both have high activation on some “Dallas” token in the original prompt.

Example: Consider two features that both activate strongly on “Dallas” in the original prompt:

Feature A: In probe prompts, peaks on “Dallas” (90% of time), “Houston” (5%), “Texas” (5%) → peak_consistency = 0.90, n_distinct_peaks = 3, semantic_confidence = 0.90
Feature B: In probe prompts, peaks on “Dallas” (40%), “is” (30%), “city” (20%), “state” (10%) → peak_consistency = 0.40, n_distinct_peaks = 4, mixed behavior
Feature A is a stable “Dallas detector,” while Feature B is context-dependent and polysemantic. The aggregated cross-probe metrics reveal this distinction.

 

 

3. Concept-Aligned Grouping
Features are grouped into supernodes by a 3 stage process: target token preprocessing, node type classification, and node naming. 

3.1  Target token preprocessing
Before classification, all tokens in probe prompts are preprocessed into two categories:

Semantic tokens: Content-bearing words with specific meaning (e.g., “Texas”, “capital”, “Austin”)
Functional tokens: Low-specificity words serving as “semantic bridges” (e.g., “is”, “the”, punctuation). These have usually low embedding magnitude[3] but accumulate contextual meaning from adjacent semantic tokens.
For each feature that peaks on a functional token, we identify the target token—the nearest semantic token within a ±5 token window (configurable). Direction is determined by a predefined functional token dictionary:

"is" → forward (e.g., “is Austin”)
"," → bidirectional (e.g., “Texas,” or “, USA”)
"the" → forward (e.g., “the capital”)
Caveat: Some early-layer features (layer ≤3) detect functional tokens themselves (e.g., context-independent “is” detectors). These are later classified as semantic despite the token type.

Purpose: Target tokens enable interpretable naming of Say X nodes. A feature peaking on “is” before “Austin” across contexts becomes “Say (Austin)”, linking functional behavior to semantic content it promotes.

 

3.2 Node type classification

Node type classification is obtained applying transparent, testable thresholds to the aggregated cross-probe metrics defined in section 3. These thresholds were tuned on held-out examples and are designed to capture distinct functional roles:

Dictionary/Semantic nodes (entity detectors, e.g., “Dallas”):

peak_consistency ≥ 0.80: When the concept token appears in a probe, it must be the activation peak ≥80% of the time
n_distinct_peaks ≤ 1: Feature should peak consistently on the same token type (not scatter across multiple tokens)
layer ≤ 3 OR semantic_confidence ≥ 0.50: Early layers where token detection occurs, OR high confidence in semantic behavior
Relationship nodes (spatial/abstract relations, e.g., “containing”):

sparsity < 0.45: Relationship features activate diffusely across relation phrases (median sparsity ratio across probes < 0.45)
Typically found in layers 0-3 where early binding operations occur
Often polysemantic until disambiguated by attention—hence lower peak consistency
Say X nodes (output promotion, e.g., “Say Austin”):

func_vs_sem ≥ 50%: Feature peaks on functional tokens (e.g., “is”, “the”) in ≥50% of probe prompts, indicating functional rather than semantic role
confidence_functional ≥ 0.90: High peak consistency when peaking on functional tokens
layer ≥ 7: Output promotion occurs in mid-to-late layers
Cross-prompt stability (applied to all groups):

Supernode features must activate consistently. Entity swaps show partial transfer (64%), though paraphrases and structural variations remain untested.
Peak tokens shift appropriately (e.g., Dallas→Houston when the city changes)
Duplicate prevention: each feature belongs to at most one supernode; conflicts are resolved by highest alignment score (computed as weighted combination of peak_consistency, semantic_confidence, and layer appropriateness).

 

3.3 Concept Aligned Naming
After classification, features receive interpretable names based on their functional role. The naming system applies distinct strategies per node type:

Node Type	Source	Selection Criterion	Format	Example
Semantic	Peak token (max activation)	Highest activation_max on semantic tokens	"token"	"Texas"
Say X	Target token (preprocessing)	Nearest semantic target from functional peak 	"Say (token)"	"Say (Austin)"
Relationship	Aggregated semantic tokens (all probes)	Highest aggregated activation on extended vocabulary	"(token) related"	"(containing) related"
Semantic nodes are named by selecting the semantic token where they activate most strongly. The system filters records where features peak on semantic tokens, sorts by activation strength, and selects the highest-activation token not in a user-configurable blacklist. For example, a feature that consistently peaks on “Texas” receives the name "Texas". If no semantic peaks are found, the system falls back to the token position from the original prompt.

Say X nodes are named by identifying the target semantic token they predict or promote. Since these features peak on functional tokens (like “is” or “the”), the naming uses the target tokens discovered during preprocessing.  The final format is "Say (token)", such as "Say (Austin)" for a feature peaking on “is” before “Austin”.

Relationship nodes are named through a two-staged process. The system constructs an extended semantic vocabulary combining tokens from the original prompt with concept names discovered during Semantic node classification. It then aggregates activation values for each semantic token across all probes and selects the highest-activation token (excluding blacklisted entries). The format is "(token) related", producing names like "(containing) related" for features that activate diffusely on spatial relationship phrases.

Blacklist system: A user-configurable set of generic tokens (e.g., {“entity”, “attribute”}) enables filtering uninformative words. 

 

4. Subgraph Construction
Once concept-aligned supernodes are defined, we construct interpretable subgraphs by (1) pinning features grouped by supernode name, (2) pinning corresponding token embedding nodes when semantic supernodes exist, and (3) pinning the output logit to complete the circuit from input to prediction.

These subgraphs are uploaded to Neuronpedia for metric computation (Replacement/ Completeness scores) and interactive analysis. Researchers can navigate the subgraph to examine individual feature attributions and compare against three baselines: the raw pruned graph, feature-specific activation patterns, and autointerp labels.

 

Empirical finding: the heuristic naming system often converges with Neuronpedia autointerp labels for monosemantic features, but often reveals complementary insights for polysemantic "Relationship" and "Say X" features. For example, in Dallas-Austin graph feature 1_12928 receives the generic autointerp label "AssemblyCulture" but our system labels it "(texas) related".

Examining its full top activations in Neuronpedia reveals an official Texas government document containing both "Texas" and "Austin" (the state capital, output logit of the seed prompt), grounding the interpretation in specific corpus content. This suggests probe-based concept alignment can serve as both hypothesis generation and validation for mechanistic interpretability.


Feature 1_12928 top activation
Texas government document containing "Texas" and "Austin" (dark green = high activation). Validates probe-based label "(texas) related" vs. autointerp's "AssemblyCulture."
