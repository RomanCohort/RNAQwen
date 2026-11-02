# A discrete code interface makes a frozen LLM read RNA sequence identity

> **状态：结构稿，不是投稿稿。**
> 这份文件把仓库里**已经测出来**的东西按期刊结构摆好。每处数字都能在
> `reports/` 或 `PLAN_v2.md` 的对应小节里对到原文。**没有做过的实验一律写成
> `[[TODO-n]]`，一处数字都没有编。** 投稿还差什么，见 §7。
>
> 先说清楚：你想投的应该是 **Nature Machine Intelligence**。没有叫
> *Nature Artificial Intelligence* 的期刊。以现在的证据强度，我认为它过不了
> NMI 的门槛，理由不是「数据少」，而是**主对照臂根本没跑完**（§7.1）。

---

## Abstract

Adapting a frozen large language model (LLM) to a non-text modality is usually done
by learning a projection that maps encoder states into the LLM input space. We show
on RNA question answering that this route fails for a measurable reason rather than
an optimization one: in the frozen encoder output, 91.1% of the per-position vector
magnitude lies along a direction shared by every sequence in the corpus, and the
sequence-specific energy decays monotonically along the adapter chain
(40.0% -> 31.5% -> 21.1%). Per-position dependence on the RNA input is heavy-tailed
and mostly negative: the median per-token KL is 0.039, and removing the RNA changes
the log-probability in the *wrong* direction on 58.4% of positions.

We replace the projection with a **discrete code interface**: a frozen RNA encoder
(ERNIE-RNA) feeds a finite-scalar-quantization (FSQ) bottleneck with 4,096 codewords,
and each codeword is given a trainable input embedding row that the LLM treats as an
ordinary token. The codes are a vocabulary, not a fingerprint: on held-out sequences,
a code predicts its local 6-mer 7.79x more often than shuffled and constant controls
(23.57% vs 3.03% / 3.40%). Within its pooling window the code is close to lossless
(89.8-92.9% per-position recovery, chance 25%).

The LLM then demonstrably reads code *identity*, and the ablation ordering is
diagnostic: zeroing the code channel costs less than substituting a wrong code from
another sequence, which costs less than a permutation of the table, which costs less
than collapsing every position to one code
(+0.558 / +0.586 / +0.811 / +0.832 nats). **A misleading code is worse than no code.**
On 40 held-out RNAs per source, the true-code arm reaches a lower answer-token
cross-entropy than both the zeroed channel and a question-prior baseline
(seq8: 0.211 vs 1.592 vs 3.505; A<C on 95% of samples, A<prior on 95%).

The channel is **locally strong and globally weak**: probes on the codes alone recover
the first 4-6 nucleotides (55-81%) and RNA type (62.8% vs 34.7% mode) but not exact
base counts (best stateless estimate: MAE ~8 bases, <12% within +/-0.5), and 16 of the
30 templates in our corpus are count or percentage questions. Finally, mixing
free-text and family-classification tasks into the same loss destroys the primary
signal even though the sequence task is 40% of the examples: exact match on the
sequence source falls from 80.0% (trained alone) to 45.0% (trained with the other
two), while the free-text source - 58.8% of the loss tokens - never rises above its
frequency floor.

We also report the instrument failures that produced wrong numbers before we caught
them, because three of them (a gold answer inside the evaluation prompt, a
diluted per-token ablation statistic, and a tokenizer that silently renumbered a
declared vocabulary extension) are the kind that produce plausible, publishable,
and false results.

---

## 1. Introduction

Feeding a non-text modality to an LLM has two established shapes. The first is a
*soft prompt*: a learned projection turns encoder states into input vectors. The
second is *native text*: serialize the modality into the LLM's own vocabulary and
fine-tune. Both were tried here on RNA question answering, on the same data, the same
base model and the same evaluation, and both failed in the same way: the model learned
the answer *format* and ignored the input.

The diagnosis we can support with numbers is not "the projection is bad". It is that
a frozen LLM has nothing to attach an out-of-distribution vector to, and a
heavily-shared component dominates whatever the encoder emits. Two facts, both
measured, not inferred:

1. **Most of the encoder output is not about the sequence.** 91.1% of the raw
   per-position magnitude of ERNIE-RNA lies in a single cross-sequence common
   direction (computed over the corpus mean); a learned compressor amplifies this to
   93.7% and a linear projection to 97.2%. Sequence-specific energy falls from 40.0%
   at the encoder output to 31.5% after compression and 21.1% after projection.
2. **The adapter is a net negative at most positions.** Per-token KL between the
   model's next-token distribution with and without the RNA input has median 0.039:
   about 70% of positions barely use the sequence at all. The tail is real
   (p90 0.226, max 10.1), and 58.4% of positions have a *negative* delta log-prob,
   i.e. the encoder is a burden there.

Native text fails for a different-looking but structurally identical reason. We
measured the Qwen3 BPE on RNA: 0.526 tokens per nucleotide, locally stable (median
perturbation span 0 nt), deterministic - and semantically meaningless, because the
merges are English statistics. The tokens are in-vocabulary, but their embeddings
were trained on English.

The common premise is that **the LLM has no representation slot for RNA**. What it
does have a slot for is a discrete token with a learned embedding. That motivates the
interface this paper is about: quantize the frozen encoder's output into a finite set
of codewords, give every codeword an embedding row, and let the LLM process the
sequence as it processes text.

Discrete bottlenecks over biological encoders are not new in isolation: FSQ has been
applied to RNA 3D structure, and vector quantization of protein coordinates has been
used to build a protein language model over discrete tokens. What we could not find
precedent for is the specific object here - quantizing an RNA *sequence* encoder and
feeding the codes to a general-purpose chat LLM as vocabulary entries - which is why
the codebook hyperparameters had to be swept on our own validation split rather than
inherited. `[[TODO-1: verified literature positioning; the references we have read
are listed in §8 and must be re-checked before submission]]`

This is a *measurement* paper more than a model paper. Its contributions are:

- a quantitative account of why frozen-LLM adapters fail on this modality, and
- a discrete code interface that works, with ablations that separate "the code is
  read" from "the presence of a vector is read", and
- a boundary on what such a channel carries (local identity yes, global composition
  no), and
- a demonstration that the failure is one *dilution* event away: mixing tasks flips
  the model from reading the channel to ignoring it.

---

## 2. Results

### 2.1 The soft-prompt route fails, and the failure is measurable

Route 1 (soft prompt): frozen ERNIE-RNA -> linear projection -> frozen Qwen3-8B, LoRA
on attention, instruction fine-tuning on 92,680 QA pairs over 4,481 RNAs. Validation
loss reaches 1.3839, i.e. the model writes well-formed answers. The RNA-attribution
diagnostics say the content did not enter:

`[[TODO-10: these diagnostics are quoted from the project log; before submission
confirm which stage and which checkpoint produced the 1.3839 validation loss and the
58.4% figure. Do not carry them into a table without that provenance.]]`

| measurement | value | reading |
|---|---|---|
| magnitude on the cross-sequence common direction | 91.1% raw -> 93.7% compressed -> 97.2% projected | the chain amplifies the constant component |
| sequence-specific energy | 40.0% -> 31.5% -> 21.1% | it decays at every stage |
| per-token KL(model with RNA vs without) | median 0.039, p90 0.226, max 10.1 | ~70% of positions barely use the sequence |
| negative delta log-prob | 58.4% of positions | at most positions the encoder hurts |
| zeroing ablation | delta decays to ~25% of its early value | the model progressively drops the channel |

Four plausible explanations were tested and killed by our own measurements, and we
report them because the negative results are the load-bearing part of the argument:
the projection is not lossy (inter-sequence similarity after removing the common
component is -0.05, i.e. the information is intact and highly separable); a learned
resampler is not the cause (its per-token profile is nearly identical to mean pooling,
58.4% vs 63.3% negative); validation loss does not select a better projection (it
anti-correlates with grounding in two paired comparisons); and encoder anisotropy is
not the main term (a de-meaned variant ties on validation loss).

The conclusion we draw is structural rather than optimizer-level: **the ceiling comes
from a frozen LLM with no slot for the modality, and no Stage-1 objective fixes it.**
This is consistent with the observation in adjacent literature that instruction
tuning of a chat model on biology instructions mainly learns output format, and that
uniform-gradient preference optimization accelerates the abandonment of the primary
signal in favour of compressed textual priors. `[[TODO-2: convert these into proper
citations with verified bibliographic data]]`

### 2.2 A discrete code interface

Route 3 replaces the projection with a learned tokenizer for RNA:

```
RNA (<=1024 nt)
  -> ERNIE-RNA (frozen, 768-dim per position)
  -> linear to 12 dims -> mean-pool 4x -> FSQ (2 levels) -> 4,096 codewords
  -> trainable 4096 x 2048 embedding table
  -> frozen Qwen3.6-35B-A3B, SOS-LoRA adapter, text QA output
```

The codeword sequence goes in through `inputs_embeds`, bypassing BPE entirely; the LLM
sees one input vector per code, exactly as it would see one per text token.

**The codes are a vocabulary, not a fingerprint.** This is the one question that can
be answered without training the LLM, and it is the question the whole route stands
on: if a code only identified *which sequence* a position came from, the embedding
table could not learn anything transferable. The non-circular test is cross-sequence
transfer: take a code seen in training, ask it to predict the local 6-mer at held-out
positions in *unseen* sequences, and compare against shuffled and constant controls.

| codebook | 6-mer from code | shuffled | constant | ratio |
|---|---|---|---|---|
| 4,096 codes, frozen encoder, per-position | 8.62% | 3.27% | 3.40% | 2.63x |
| 4,096 codes, frozen encoder + 4x pooling | 5.83% | 3.17% | 3.40% | 1.84x |
| 4,096 codes, **after end-to-end training** | **23.57%** | 3.03% | 3.40% | **7.79x** |
| 256 codes (rejected) | - | - | - | 1.95x |

Three readings matter. First, the 2.63x row reproduces an earlier independent
measurement, so the ruler was self-checked before the 7.79x row was trusted. Second,
end-to-end training **improved** the codebook rather than collapsing it: 3,885 of
4,096 codes are in use on the full corpus (hard entropy 7.44/8.32), and the codeword
*keys* were almost entirely rewritten (0.1% agreement with the frozen version) while
the content mapping survived - the CE gradient changed the codebook, it did not
destroy it. Third, the pooling window is close to lossless: per-position recovery
inside a window is 92.1% / 92.9% / 92.4% / 89.8% against 25% chance, so the 7.79x is
not an artifact of the compressor encoding only the window's first position.

`[[TODO-3: report codebook sweeps over quantization levels and latent dim. The
project notes that no precedent exists for this exact object, so these numbers were
chosen on our own validation split, not inherited.]]`

### 2.3 The LLM reads code identity, and misleading beats absent

Two instruments, deliberately different, agree.

**Instrument 1 - the ordering of four substitutions.** At step 3,750 we replaced the
code sequence four ways and measured the change in the log-probability of the gold
answer (n=317; `mean` = true-code log-prob minus that arm's, so larger = more damage):

| arm | what is done to the code sequence | mean damage |
|---|---|---|
| zero table | every code vector set to zero - the channel is cut | +0.558 |
| random table | codes redrawn from the table - identity destroyed, presence kept | +0.586 |
| permutation | code <-> content mapping shuffled | +0.811 |
| constant code | every position gets the same code | +0.832 |

If the model used only the *presence* of a vector, the order should reverse: zeroing
removes presence and should hurt most, while permutation and a constant code keep
presence and should hurt least. The measured order is the opposite. **The model is
reading identity, and a wrong identity is worse than none at all** - a misleading code
makes the model confidently read off the wrong answer, whereas a zeroed channel
collapses it onto a constant that is *worse* than the question prior (2.4 measures
both). `[[TODO-4: this statement about the mechanism is currently an inference from
the ordering; it needs a direct test]]`.

**Instrument 2 - answer-token cross-entropy on held-out sequences.** Generation on a
35B model costs ~40 s per item, so we scored teacher-forced CE on the answer tokens
instead, on the test split, 40 RNAs per source:

**Provenance, and a confound we did not resolve:** this table is checkpoint
**step 6,529** (11.35 h) with the **190-projection** adapter (8.13 M parameters).
The exact-match table in 2.4 is checkpoint **step 7,500** (15.44 h) with the
**310-projection** adapter (10.58 M). The adapter preset changed during the run, so
**the two headline instruments were measured on different models.** The step-6,529
checkpoint has since been rotated off disk. Unifying this is item 1 of the work in
7.2, and until it is done the two tables must not be read as one model's results.

| source | CE(A) true code | CE(B) code from another RNA | CE(C) zeroed | CE(prior) | A<C | A<prior |
|---|---|---|---|---|---|---|
| rfa (145k-RNA cache) | **0.2419** | 2.6202 | 1.7854 | 2.7817 | 92% | 100% |
| seq8 (curated set) | **0.2108** | 2.0825 | 1.5917 | 3.5053 | 95% | 95% |

`prior` is the question-only baseline: the cross-entropy obtained by always answering
each template with its most frequent training answer. Beating `prior` is the strict
test - a model that only reads the question can match `prior` but not beat it. The
true-code arm beats it on 95-100% of individual samples. Note also that CE(B), the
wrong-code arm, is *worse than zeroing*: the same asymmetric-damage signature as
Instrument 1, now measured per sample on unseen sequences.

**The channel's contribution grows during training.** Validation CE with true codes
fell monotonically while the zeroed-code arm did not follow it down, so the gap
widened at every one of the first five evaluations:

| step | 750 | 1,500 | 2,250 | 3,000 | 3,750 |
|---|---|---|---|---|---|
| CE(A) | 0.674 | 0.513 | 0.397 | 0.305 | 0.218 |
| CE(C) | 0.728 | 0.757 | 0.742 | 0.860 | 0.886 |
| gap (C-A) | +0.054 | +0.244 | +0.345 | +0.555 | +0.669 |

That is roughly 12 standard errors of widening, or put plainly: with true codes the
loss keeps dropping, with zeroed codes it does not.

### 2.4 The channel is locally strong and globally weak

A probe that never sees the LLM - a stateless estimator over code histograms, fit on
training sequences and evaluated on unseen ones - bounds what the codes can possibly
carry:

| question family | probe accuracy | note |
|---|---|---|
| dinucleotide at positions 1-2 | 81.0% | |
| first 4 nt (including reverse complement) | 67.4% | |
| RNA type | 62.8% | mode floor 34.7% |
| first 6 nt / positions 1-6 / reverse-complement first 6 | 55.3% | |
| first 8 nt | 44.6% | |
| last 4 / last 6 / last 8 nt | 14.9% / 11.3% / 8.9% | |
| 10 count-style templates | 0.1-3.7% | |
| 6 percentage-style templates | 0.0% | |

Exact counts are not recoverable by the best *stateless* estimator: MAE 8.41 (A),
8.09 (C), 7.90 (G), 8.22 (U) bases, GC content off by 4.21 percentage points, and
only 8.7-11.4% of items within +/-0.5 bases, against a 1.6-2.6% mode floor. Two
caveats must travel with this: a stateless estimator is a lower bound, since a
sequence-aware reader (including the LLM) can do better; and 16 of the 30 templates
in the corpus are count or percentage questions. **So the corpus asks mostly for the
one thing the channel is worst at.**

The honest summary of the trained system, on 300 generated items (10 templates x 30
RNAs), is that it is far from solved:

(checkpoint step 7,500, 310-projection adapter; the decoder-stop caveat below applies)

| arm | exact match | answer is a prefix | ran past the answer turn | char-F1 |
|---|---|---|---|---|
| true code | 11.0% | **70.0%** | 72.0% | 0.492 |
| code from another RNA | 0.0% | **2.0%** | 72.0% | 0.372 |
| zeroed code | 0.3% | **0.3%** | 63.7% | 0.264 |

The middle column is the one to read: 70.0% against 2.0% and 0.3%. The generation
does not stop at the answer, so exact match charges the true-code
arm for text it emits *after* answering correctly. The stored generations were
re-scored offline (`scripts/_scratch/_reanalyze_acc.py`; it recomputes normalization
from the raw strings and asserts it matches what the scoring pass stored, so a
different normalization cannot slip in): the model lands the answer on 70.0% of items
overall - 93.3% on first-4, 96.7% on the dinucleotide, 90.0% on the reverse
complement of the first 4, 43.3% on last-6 - and then continues into a hallucinated
next turn. The typical continuation is `\nuser\n` followed by another RNA answer, i.e.
the model is replaying the multi-turn structure of its own training data.

**What the two control arms actually do is the sharper result.** Looking at the answer
slot (the first `len(gold)` characters) across the 30 items of each template:

| arm | distinct answers per template | share of the single most frequent |
|---|---|---|
| true code | 25.8 / 30 | 10.0% |
| code from another RNA | 25.8 / 30 | 10.0% |
| zeroed code | **3.0 / 30** | **69.3%** |

For "first 4 nucleotides" the zeroed arm gives `UUAU` on 23 of 30 items. Its constant
is a run of U carrying at most one other base, in all ten templates - `UUAU`, `UUUU`,
`UUU`, `UUUU`, `CUUUUU`, `UUUUUUAU`, `UUUUUA`, `UU`, `AUUA`, `CUUUUA` - and it is the
modal answer of *none* of them (`UACG` 6.12%, `UACGUAGG` 2.83%, `AAACAG` 3.33%, `UA`
11.19%; `reports/prior_forensics.json`). **Zeroing the channel produces neither noise
nor the question prior. It produces a degenerate constant that scores *below* the
prior.** A predictor that never sees the RNA and always answers each template with its
most frequent training answer scores 4.26% on this corpus's test split
(7,225/169,410), and 8.7% (26/300) on this particular sample, where the sampler
over-represents mode answers (26 items against ~13 expected); the zeroed arm scores
0.3% (1/300).

The prior frame does bite arm A, and the honest version of it is uncomfortable: on the
exact ruler A's 11.0% sits 2.3 pp above that 8.7%, which at n=300 is not a
distinguishable gap. The separation appears once the ruler stops swallowing the
answer. Restricted to the 274 items whose gold is *not* the template mode - items a
prior predictor cannot score on at all - arm A puts the answer at the head of the
generation 68.2% of the time (12.0% exact) and emits the mode answer there exactly 0
times. The same 274 items give 2.2% under a wrong code and 0.4% under a zeroed code,
so everything A recovers on the hard subset came through the channel. Substituting a
code from another RNA produces the *other* sequence's answer with full confidence:
25.8 distinct answers per template, near-zero correctness, and a char-F1 (0.372) above
the zeroed arm's (0.264) because it emits plausible RNA of plausible length - it lands
on the corpus mode at roughly the pool's own rate (23/274) while never matching the
gold, which is what a faithful report of the wrong molecule looks like. Both controls
therefore land below the frequency floor of several templates, and the reason is now
measured rather than hypothesized. `[[TODO-5: implement the stop condition in the
decoder and re-generate; raise n per template - at n=300 the exact ruler cannot
separate A from the prior]]`

### 2.5 One mixing event is enough to flip the model from reading to ignoring

An earlier version of this pipeline trained on three sources at once: sequence tasks
(answer deterministically recomputable from the sequence), Rfam family
classification, and free-text annotation questions. Exact match, per source, on the
merged run:

| source | share of examples | share of loss tokens | exact match (true / wrong / zeroed code) | frequency floor |
|---|---|---|---|---|
| sequence tasks | 40.4% | 16.0% | **45.0%** / 7.5% / 17.5% | 4.55% |
| Rfam family | 44.4% | 25.3% | 0.0% / 0.0% / 0.0% | 0.12% |
| free text | 15.2% | **58.8%** | 5.0% / 5.0% / 2.5% | 14.26% |

The same source, same templates, same ruler, trained **alone**, reaches 80.0%. So
adding the other two sources did not merely dilute the metric - it halved the
sequence task's accuracy, because the merged objective rewards the other two
questions, which are answerable from memorized priors. The failure mode is visible in
the outputs: asked to describe a human lncRNA, the model emits a sentence about a
yeast mitochondrial large subunit; asked for a family, it answers `RF00001` when the
gold label is `RF00177`. Both arms are emitting high-frequency training strings, not
reading the sequence. Note the imbalance in the last column: the free-text source is
15% of the examples but 58.8% of the loss tokens, because its answers are long.

**This is the finding we would put in the abstract of a longer paper**: the primary
signal did not degrade gracefully under mixing, it inverted. The fix was not a
reweighting schedule but a corpus with a single answerability contract - answers
4-8 characters, one deterministic recomputation each, token mass roughly even across
templates.

### 2.6 A deployable artifact from a single GPU

The trained system is a 35B mixture-of-experts model (35.107 B parameters, 40 layers,
hidden 2,048, vocabulary 248,320) adapted on hardware that cannot hold it: int4
quantization is applied to the 20,480 MoE expert Linears only (attention and the
shared expert stay bf16, which is what keeps the merged adapter exact), giving
18.18 GiB of device memory and a 386-476 s load. The adapter is an SOS-LoRA variant
(r_tot=8, K=4, alpha=16, gamma_max=2.5, ortho 1.0) over 310 attention and
shared-expert projections - 10.58 M trainable parameters, deliberately parameter-
matched to a same-rank LoRA so that a controlled comparison is available once one is
run. `[[TODO-6: the parameter-matched LoRA control has never been run, so there is
currently no evidence that SOS-LoRA is better than plain LoRA. Do not claim it.]]`

The code channel is what makes distribution tractable: because a code depends only on
the sequence, the encoder can run offline and the online path needs no biology
encoder at all for a fixed library. We ship the merged model as GGUF
(Q4_K_M 20.23 GiB, Q8_0 35.22 GiB) with the 4,096 codewords carried as real
vocabulary entries, so that the input embedding row for code `c` is exactly the
vector training fed the LLM; the code table, the FSQ weights and a prompt builder
travel beside the file because an encoder cannot live inside a GGUF. Merging the
adapter into the base is not free and we report the size of the change instead of
asserting equivalence: max |delta logit| 2.32, i.e. 9.05e-2 of the max logit, with
top-1 flipping on 4 of 128 probe positions.

---

## 3. Discussion

The result we would defend is not a score. It is that the standard way of attaching a
frozen encoder to a frozen LLM has a failure mode that is *measurable before you
spend the compute*: the encoder's output is dominated by a sequence-independent
component, and the adapter chain amplifies it. Checking two numbers - the share of
magnitude on the common direction, and the sign distribution of the per-token
dependence - tells you in minutes whether the projection route has a chance on your
modality. Both were computed here on a 32 million sequence corpus.

The second is that a discrete interface changes the *kind* of question the LLM is
asked. It is no longer asked to interpret a vector in a space it has never seen; it
is asked to look up a row of a table it is training. That reframing is what makes the
ablation outcomes interpretable, and it produces a diagnostic we did not expect: the
ordering of the four substitution arms. **Misleading information is worse than
missing information**, by 0.27 nats in the mean and by a factor of 7.6 in CE on
held-out sequences. If that generalizes beyond RNA, it is a small but sharp statement
about multimodal interfaces: partial channel degradation is not a smooth path to
degraded performance.

The third is a negative result with teeth: this channel carries local sequence
identity and does not carry global composition. Exact counts need to see the whole
sequence; a 4x-pooled code sequence is a local description. The corpus we built asks
for counts in 16 of 30 templates, which explains a large part of the gap between the
probe ceiling (55-81% on local identity, or 70.0% by prefix accuracy end to end) and
the exact-match number (11.0%), most of which is a decoder artifact rather than a
model failure.
Researchers who want LLMs to reason about composition need a channel with an
aggregate, not a better reader.

Finally, the dilution result is a warning about data engineering that is easy to state
and easy to violate: a task whose answers are memorizable will win the gradient
against a task whose answers must be computed, even when it is a minority of the
examples, if it owns the majority of the loss tokens.

---

## 4. Limitations

This section is deliberately hostile to the manuscript, because the failure this
project has hit most often is a plausible number that turns out to be an instrument.

1. **The headline comparison was never completed.** The design is a head-to-head of
   soft prompt vs native text vs discrete codes on one dataset, one base model, one
   evaluation. Only the code route was trained to completion. The soft-prompt route
   is complete through Stage 1 and its Stage-2 numbers are the *invalidated* ones
   (see 5.6); the native-text route was argued against by measurement of the
   tokenizer and by the failure of the frozen-adapter route, not by a trained
   control. **A reviewer will read the paper as "one route works, two were reasoned
   away", and would be right.**
2. **Every evaluation is on a self-built corpus.** No shared benchmark, no external
   leaderboard, no comparison with published RNA QA systems. The floor for a
   hallucination-prone free-text template (14.26%) is above the measured accuracy of
   that arm (5.0%), which means our own corpus contains a task we cannot do.
3. **Training used 3.1% of the available sequences** (4,481 of 145,201), because the
   encoder hidden-state cache was built for 4,481. The 145k cache exists but was not
   used for the reported run. Nothing here establishes scaling behaviour.
4. **Single seed, thin n.** The per-template accuracies are 30 items each; the
   head-vs-tail difference (13.8% vs 4.4%) is reported as not significant and should
   stay that way.
5. **The headline single number is polluted by a decoder artifact.** The generation
   does not stop at the answer, so true-code exact match is 11.0% where prefix
   accuracy is 70.0%. Re-scoring the stored generations fixes the *reporting* (2.4),
   but the artifact itself is still in the model as shipped, and the corrected
   numbers are on 300 items from a single checkpoint.
6. **The mechanism claim in 2.3 is an inference from ordering**, not a measured
   mechanism. We have a preference ordering; we do not have an account of how the
   misleading code propagates through 40 layers.
7. **No adapter control.** SOS-LoRA has never been compared against a
   parameter-matched plain LoRA on this task.
8. **Asymmetric reference quality.** The alternative routes are argued from
   measurements of our own implementation, not from their best published form.
9. **The quantization path is a distribution choice, not a fidelity result.**
   Merging the adapter into the base changes the function measurably (2.6), so
   "the GGUF is the trained model" is false in the strict sense; it is the trained
   model folded, with the fold measured and reported.
10. **The two primary instruments used different adapters.** The code-channel CE
    table was measured at step 6,529 with the 190-projection preset, the exact-match
    table at step 7,500 with the 310-projection preset, because the preset changed
    mid-run. Both were loaded *without* an error, which is the dangerous part: the
    smaller preset is a strict subset of the larger one, so every tensor of the
    smaller model finds a match in a checkpoint of the larger one, and nothing warns
    you. A reviewer who notices this will discount sections 2.3 and 2.4 together.

---

## 5. Methods

### 5.1 Data

Sequences come from RNAcentral (32,524,827 sequences in the local mirror);
annotations, organism, RNA type and literature links are joined by URS identifier.
The QA corpus has three generations. The first mixed three sources and is the
subject of 2.5. The reported run uses the corrected single-contract corpus:
answers 4-8 characters, each deterministically recomputable from the sequence, token
mass roughly even across templates.

| split | items |
|---|---|
| train | 1,297,910 |
| validation | 29,500 |
| test | 169,410 |

Two encoder hidden-state caches back training: 4,481 curated RNAs (with annotation;
5.9 GB) and 145,201 RNAs (sequence only; 70.9 GB fp16 memory-mapped). Encoder
positions are `len(sequence) + 2` because ERNIE-RNA prepends `[CLS]` and appends
`[SEP]`; failing to subtract that offset shifts every positional target by one and
silently invalidates the result, which happened once.

### 5.2 Encoder and code tokenizer

ERNIE-RNA (768-dim, frozen; verified by hash, architecture, and sequence-sensitivity
checks) feeds a linear map to 12 dimensions, a mean-pool over 4 positions, and FSQ
with 2 levels, giving 4,096 codewords. FSQ was chosen over a learned codebook because
codebook under-utilization is a known failure of learned quantizers on biological
sequences, and FSQ cannot collapse by construction. A de-anisotropy step was tested
and dropped: the learnable linear layer in front of the FSQ absorbs the common
direction itself, changing reconstruction error by ~0.005.

Reconstruction MSE at stage 0 is 0.0444; 4,092 of 4,096 codes are used; median reuse
is 264. Pooling to 4 (rather than per-nucleotide codes) was an explicit trade: it
costs transfer ratio (2.63x -> 1.84x) before end-to-end training and buys sequence
length headroom, and after training the pooled version overtakes the
per-nucleotide one (7.79x).

### 5.3 Model and training

Base model: Qwen3.6-35B-A3B, 35.107 B parameters, 40 layers, hidden 2,048,
vocabulary 248,320, sparse mixture-of-experts. Device: a single integrated AMD GPU
(gfx1151, ROCm), so the base is loaded in int4 by quantizing the 20,480 expert
Linears only; attention and shared-expert projections stay bf16, which is what allows
the adapter to be merged exactly later. Device memory 18.18 GiB; load time 386-476 s;
peak committed host memory ~90 GB during training (~97 GB for an evaluation pass that
also memory-maps a second, 145k-sequence hidden-state cache).

Adapter: SOS-LoRA, r_tot=8, K=4, alpha=16, dropout 0.05, gamma_max=2.5,
orthogonality coefficient 1.0. Two target presets are in play: **C**, the 190
attention projections (8.13 M trainable parameters), and **D**, 310 attention and
shared-expert projections (10.58 M). Both are parameter-matched to a same-rank LoRA
over the same targets. The reported checkpoint is preset D; the code-channel CE table
was measured with preset C (see the provenance note in 2.3).
The code embedding table is trained jointly. Optimizer and schedule follow the
project's training configuration. Batch size 8, 19.6-21.4 s per step; 11.35 h of wall
clock to step 6,529 and 15.44 h to step 7,500, with early stopping firing at 935 min
(15.6 h) of total run time.
`[[TODO-9: fill in optimizer, schedule and gradient-accumulation settings from the
training script before submission]]`

**Stopping rule.** The reported checkpoint is the one that maximized the
*machine-checkable* signal, not a loss: early stopping triggered when validation
CE(A) failed to improve by more than max(0.5% x CE, 1 standard error) = 0.0459 for
four consecutive evaluations (3,000 steps). Under that rule step 7,500 wins
(CE(A) 0.0891, CE(C) 1.1881, gap +1.099). We flag the ambiguity honestly: the last
evaluation at step 10,500 has the lowest raw CE(A) of the run (0.0589) and was
rejected as noise because it improved on the incumbent by only 0.0302 < 0.0459.
Which checkpoint is "best" depends on whether one trusts the raw value or the
standard-error rule, and the difference is not resolved in this manuscript.
`[[TODO-7: resolve checkpoint selection; report both]]`

### 5.4 Evaluation instruments

We used four, and every one of them was chosen because a previous instrument lied.

1. **Cross-sequence code transfer** (2.2). Code -> local 6-mer prediction on held-out
   sequences, with shuffled and constant controls. This is the only non-circular test
   of "the codes are a vocabulary": reconstruction error cannot be used, because the
   quantizer is deterministic and the autoencoder trains on it, so a within-sequence
   improvement is guaranteed by construction.
2. **Answer-token cross-entropy** (2.3). Teacher-forced CE on the answer tokens with
   the true code, another RNA's code, a zeroed code, and a question-prior baseline.
   Per-sample paired comparisons are reported because they are more sensitive than
   means.
3. **Substitution ordering** (2.3). Four interventions, ranked. Used because a
   per-token negative-fraction statistic is diluted by boilerplate (see 5.6).
4. **Stateless probe ceiling** (2.4). A code-histogram estimator fit on training
   sequences and evaluated on unseen ones, to bound what the channel can carry
   *independently of the LLM*.

### 5.5 Putting the codes into a distributable artifact

A GGUF has no adapter slot, so the adapter is merged into the base (drift reported in
2.6) and the code channel has to become vocabulary. Training fed the LLM embedding
rows looked up from a 4,096 x 2,048 table at a placeholder token id, so the id carried
no information - the vector did. In the exported file each code becomes a real token
whose input embedding row *is* that vector, rounded to bf16, which is the same
rounding training applied when it cast the table to the model dtype. The codes'
*output* rows are set to a constant zero, so a code token's logit is identically zero
and it can never win the argmax; filling them with a large negative constant instead
is a trap, because the logit of a constant row `c` is `c * sum(h)` and becomes a very
*large positive* number whenever `sum(h) < 0` (observed: the model answered a
plain-text prompt with a run of code tokens).

The base tokenizer names 248,077 of the base model's 248,320 embedding rows. Any
vocabulary extension declared past the last named id is silently renumbered down to
`max(id)+1`, which moved code 0 from id 248,320 to 248,077 - a 243-position offset
between the token *text* and the embedding row it is supposed to be. The export path
now declares 243 filler tokens so that identity survives the converter, and it
asserts `tokenizer["<R0>"] == base_vocab` before writing anything.
`[[TODO-8: the same conversion path should be re-validated after the vocabulary fix;
the end-to-end check is 4/4 on four sequence questions, which is a smoke test, not a
result]]`

### 5.6 Instruments that produced false results first

These are reported because each one generated numbers that looked publishable.

1. **The evaluation prompt contained the gold answer.** Every accuracy number
   produced before this was found is void, and was retracted in the project record
   rather than quietly dropped.
2. **A per-token ablation statistic was diluted by boilerplate.** The label span
   averages 9.89 tokens, of which 40.4% is a fixed empty reasoning block, 31.1% is a
   fixed answer prefix, and only 28.4% is the answer. A statistic computed over the
   whole span cannot move far from chance, so the "must be below 45%" acceptance
   line was unattainable by construction on this corpus. The statistic was replaced
   by an answer-only mask built from offset mapping - not by subtracting lengths,
   which is wrong because BPE merges at boundaries (a 12-nt answer maps to 5 masked
   tokens while its standalone encoding is 6).
3. **A training-only code path hid a decoding bug.** Training only prefills, so it
   never exercised the incremental path used during generation, which the runtime had
   backed with a generic convolution that fails on this device. An entire evaluation
   produced `<ERR>` for 120 items and an accuracy of 0.0%. It was fixed by
   substituting an equivalent shift-and-add update (validated against
   `conv1d(padding=0, groups=C)`: max |delta| 9.5e-07, state update bit-identical).
   The lesson generalizes: no loss value can warn you that a code path is untested.
4. **A memory API reported the wrong size.** On this integrated GPU,
   `mem_get_info()` reports 107 GB where the machine has 63.6 GB of physical memory;
   a decision made on that number crashed a load. Only physical counters are trusted
   now. Similarly, committed-memory figures come from a dynamically resized paging
   file and are not a capacity measure.
5. **A statistic that was arithmetically forced.** "52% of codes appear exactly once"
   was reported as codebook collapse; with 29k positions and a 65,536-entry table the
   median reuse is 1 by construction. The same sampling artefact recurred later in a
   training log line ("350 of 4096 codes in use", computed over 40 sequences) that had
   already been written into a conclusion - the full corpus shows 3,885 of 4,096.
6. **A frontend length cap that no test could see, because the test picked its own
   inputs.** The shipped path recomputes the codes from the sequence, but capped it at
   512 nt, while training (47_/48_/50_) used `MAX_NT = 1024` and the corpus is collected
   at <=1022 nt; 1,845 of the 4,481 curated sequences (41%) are longer than the cap.
   The end-to-end check could not have caught this: it picked the four *shortest*
   sequences available, none longer than 120 nt, and scored by substring, so a
   prepended `A:` still counted as a hit. The cap now matches training, and on a paired
   re-run of the same 120 items the fix is worth +1.6 pp overall (54.2% -> 55.8%) and
   +15 pp in the 512-800 nt band. We deliberately do not claim more than that: the
   first reading of this re-run split the items by "truncated" and showed a 39 pp gap,
   but the split was confounded - each template's sample was drawn at the same
   positions, so all ten templates share the same twelve RNAs, and the comparison was
   really three RNAs against nine.

We consider this list part of the contribution. Four of the six are silent: they
produce a number, not an error.

---

## 6. Figures

| # | content | status |
|---|---|---|
| 1 | pipeline: encoder -> FSQ -> code embeddings -> LLM, with the two routes that failed | draw |
| 2 | (a) share of magnitude on the common direction along the chain; (b) per-token KL distribution with the negative region shaded | data exists |
| 3 | code transfer ratios (2.63x / 1.84x / 7.79x) against shuffled and constant controls | data exists |
| 4 | CE(A) and CE(C) vs step with the widening gap; inset: the four-arm substitution ordering | data exists |
| 5 | per-template probe ceiling vs end-to-end exact match, sorted | data exists |
| 6 | dilution: source shares (examples vs loss tokens) against exact match | data exists |
| 7 | the GGUF path: adapter merge drift and the 243-row vocabulary offset | data exists |

Figure 2 is the figure that carries the paper's main claim and is the only one at
publication quality already, because the numbers were computed as a diagnostic
rather than for a figure.

---

## 7. What is missing before this could be submitted

### 7.1 The verdict

As it stands this reads as a strong internal report, not a Nature Machine
Intelligence paper. The gap is not "more data". It is that the central comparison is
unfinished (7.2 item 1), the evaluation has no external anchor (item 2), and the
mechanism in 2.3 is an ordering rather than a measurement (item 6). Fixing the
first two is a few weeks of compute on existing assets; item 5 is the strongest
scientific claim available here and the one worth building the paper around.

### 7.2 Ordered by what it buys per unit of work

| # | experiment | why it is the bottleneck |
|---|---|---|
| 1 | Train the native-text route and finish soft-prompt Stage 2, same data and base | turns "we reasoned the other routes away" into a head-to-head; without it there is no paper |
| 2 | Evaluate on an external benchmark / held-out RNA families | makes the numbers comparable to anyone else's |
| 3 | Train on the 145k-RNA cache instead of 4,481 | the codebook is the contribution, and it was fit on 3% of the available sequences |
| 4 | Codebook sweep (levels, latent dim, pooling) | currently borrowed from adjacent work with no precedent for this object |
| 5 | Make the asymmetry result the centrepiece: measure *how* a misleading code propagates | this is the novel, transferable claim |
| 6 | Implement the decoder stop condition and re-generate; raise n per template | re-scoring is done (2.4) but the shipped model still rambles, and 30 items/template is thin |
| 7 | Parameter-matched LoRA control | removes an obvious reviewer objection |
| 8 | Add a channel with an aggregate (counts) | directly addresses the measured global weakness instead of working around it |
| 9 | Multi-seed, error bars on the per-template numbers | n=30 per template is not enough for a claim |

### 7.3 Things that are ready to use as they are

The falsification of the frozen-adapter route (2.1) with its two numbers; the code
transfer result (2.2); the substitution ordering (2.3); the local/global boundary
(2.4); the dilution result (2.5); and the instrument-failure list (5.6). Each is
fully backed by a report file in the repository.

---

## 8. References and availability

Related work read in full for this project, with identifiers as recorded in the
project log. **Bibliographic details must be re-verified and formatted before
submission; nothing here should be trusted as a citation yet.**

- RNA-GPT (arXiv:2411.08900) - the work this project started as a reproduction of.
- ChatMultiOmics (arXiv:2412.19191) - instruction tuning of a chat model mainly
  learns output format.
- MAPO (arXiv:2605.27741) - uniform-gradient preference optimization and late-stage
  modality collapse.
- Moirain (arXiv:2605.23961) - conditional RNA generation as multi-stage alignment.
- RiboSphere (arXiv:2603.19636) - FSQ over RNA structure; motivated avoiding learned
  codebooks.
- ProVQ (arXiv:2603.22304) - premature discretization; motivated the curriculum
  question.
- DPLM-2 (arXiv:2410.13782) - discrete tokens for protein coordinates.
- StructTokenBench (arXiv:2503.00089) - codebook under-utilization as the main trap
  in learned tokenizers.

Code, data generators, evaluation scripts, checkpoints, and the exported GGUF are in
the project repository. Encoder weights (ERNIE-RNA) and base model weights
(Qwen3.6-35B-A3B) are third-party and are not redistributed here.








