# Experiment Summary

## Overview

This document summarizes experiments analyzing LLM behavior in a hierarchical action sequence learning task (star-making paradigm). The task involves learning 4-key action sequences during a learning phase and applying them during a transfer phase.

---

## Latest gpt-oss-120b Batch (Nov 29-30, 2025)

- Ran `analysis/analyze_result.py` on all transfer_high/transfer_low JSONs (main, no_item, notify_transfer, generalize); outputs live under `results/gpt-oss-120b/analysis/transfer_{low,high}/{setting}` with CSVs and plots.
- Use **main** runs for primary analyses and **no_item** for item-ablation comparisons; **notify_transfer** and **generalize** are baselines only.

| Transfer | Setting | Overall success | Early->late gain (first 5 -> last 5) | Comment |
|----------|---------|-----------------|------------------------------------|---------|
| Low | main | 12.0% | +9.4 pp (2.8% -> 12.2%) | Better than high; modest chunking improvement |
| Low | no_item | 8.3% | +6.0 pp (2.4% -> 8.5%) | Item removal hurts accuracy |
| High | main | 10.4% | +10.2 pp (0.3% -> 10.5%) | Slow start but catches up late |
| High | no_item | 6.1% | +5.1 pp (1.1% -> 6.2%) | Weakest of primary conditions |

Baseline references (not used for hypothesis tests):

- Low notify_transfer: 14.6% overall, +12.7 pp gain (stars 0-3)
- High notify_transfer: 12.9% overall, +11.0 pp gain (stars 0-3)
- Low generalize: 12.6% overall, +10.5 pp gain (stars 0-5)
- High generalize: 13.0% overall, +10.8 pp gain (stars 0-5)

## Experiment Scripts (gpt-oss-120b)

- Ran all `analysis/original_experiments` and `analysis/new_experiments` on main and no_item results; outputs live under `results/gpt-oss-120b/experiments/{learning_phase,transfer_phase,item_stimulus,sequence_transfer_usage,success_failure}/{main,no_item}`.
- Original learning_phase: main shows strong validity boost and positive validity x trial (valid chunks grow over trials); no_item flips validity negative or flat (low: validity=-0.07, interaction ns; high: validity=-0.06 with a small positive interaction).
- Original transfer_phase: with items, high-transfer outperforms low and widens over trials (transfer_type=+0.027, interaction=+0.00145); without items, low beats high (transfer_type=-0.042) with only a tiny positive interaction.
- Item-stimulus (four-condition): validity x item_feedback is large in learning (+0.319) and transfer (+0.702); items lift learning accuracy (+2.8 pp main, positive trial interaction) and specifically boost high-transfer accuracy (item x transfer=+0.069) despite a negative main item effect.
- Sequence transfer usage: with items, OLD reuse starts near 20-28% and valid pairs dominate (seq_type_indicator ~0.91) with mild decay; without items, OLD reuse is ~0 and decays, and valid preference is much weaker (seq_type_indicator 0.19 high / 0.13 low) with steeper negative slopes and sharp drops in valid-pair reuse.
- Success vs failure splits: top learners improve valid use faster (transfer_low main group=+1.22, group×trial=+0.0077); learning-transfer correlations stay weak (high main r≈-0.14 ns; no_item high Spearman 0.186, p≈0.042 only marginal).

---

## 1. Learning Phase Analysis (`learning_phase.py`)

### Hypothesis
During learning, participants will progressively discover and preferentially use valid 2-key action sequences (chunking) over invalid sequences, and this effect will increase over trials.

### Mixed Model
**Outcome**: Count of 2-key sequences (valid or invalid) in incorrect trials
**Predictors**:
- Validity (0=invalid, 1=valid)
- Trial (centered around mean)
- Validity × Trial interaction

### Key Results

| Condition | Validity Effect (p) | Trial Effect (p) | Validity x Trial Interaction (p) |
|-----------|---------------------|------------------|----------------------------------|
| Transfer Low / No Item | p < 0.001 (negative) | p = 0.35 (ns) | p < 0.001 (positive) |
| Transfer Low / Item | p < 0.001 (positive) | p < 0.001 (negative) | p < 0.001 (positive) |
| Transfer High / No Item | p < 0.001 (negative) | p = 0.07 (ns) | p = 0.04 (positive) |
| Transfer High / Item | p < 0.001 (positive) | p < 0.001 (negative) | p < 0.001 (positive) |

### Interpretation
- **Supported**: The significant validity x trial interaction across all conditions shows that valid 2-key sequences increase relative to invalid sequences over trials
- The **item condition** shows stronger chunking effects (positive validity main effect, meaning more valid than invalid sequences)
- The **no-item condition** shows initial preference for invalid sequences but learns to correct this over trials

---

## 2. Transfer Phase Analysis (`transfer_phase.py`)

### Experiment 2.1: 2-Key Sequence Transfer Times

**Hypothesis**: Valid 2-key sequences discovered in one slot will be transferred to the other slot faster than invalid sequences.

**Mixed Model**:
- **Outcome**: Difference in transfer time (valid - invalid)
- **Predictors**: Block (centered around mean)

**Results (Item condition)**:
- Block-centered effect: B = 0.44, z = 3.25, p = 0.001
- Transfer time difference (valid - invalid) increases over blocks

**Results (No-Item condition)**:
- Block-centered effect: B = -0.04, z = -0.56, p = 0.57 (ns)

**Interpretation**: With item feedback, participants show faster transfer of valid 2-key chunks. Without items, no differential transfer is observed.

### Experiment 2.2: Transfer Accuracy Comparison (High vs Low)

**Hypothesis**: Transfer accuracy will differ between high and low transfer conditions.

**Mixed Model**:
- **Outcome**: Trial accuracy (0=incorrect, 1=correct)
- **Predictors**: Transfer type (0=low, 1=high), Trial (centered), Transfer type × Trial interaction

**Results (Item condition)**:
- Transfer type effect: B = 0.028, z = 10.2, p < 0.001
- Interaction (type x trial): B = 0.0014, z = 6.14, p < 0.001
- High transfer shows better accuracy that improves over trials

**Results (No-Item condition)**:
- Transfer type effect: B = -0.05, z = -16.8, p < 0.001
- Low transfer actually shows better accuracy than high transfer

**Interpretation**: Item feedback enables better transfer to high-similarity contexts; without items, low transfer (more different) is paradoxically easier.

---

## 3. Item Stimulus Effect (`item_stimulus.py`)

### Experiment 3.1: Effect on 2-Key Sequence Chunking

**Hypothesis**: Item feedback during learning improves chunking of 2-key sequences and facilitates their reuse during transfer.

**Mixed Models**:
1. **Learning Phase Model**:
   - **Outcome**: Count of 2-key sequences (valid or invalid) in incorrect trials
   - **Predictors**: Validity (0=invalid, 1=valid), Trial (centered), Item feedback (0=no_item, 1=with items), and all interactions up to 3-way

2. **Transfer Phase Model**:
   - **Outcome**: Count of 2-key sequences (valid or invalid)
   - **Predictors**: Validity, Trial (centered), Item feedback, Transfer type (0=low, 1=high), and interactions up to 4-way

**Learning Phase Model (2-Key Sequences)**:
| Term | Coefficient | z | p |
|------|-------------|---|---|
| Validity | -0.081 | -5.72 | < 0.001 |
| Trial | -0.009 | -6.87 | < 0.001 |
| Item Feedback | 0.042 | 2.92 | 0.003 |
| Validity x Item | 0.245 | 12.14 | < 0.001 |
| Validity x Trial x Item | 0.027 | 10.86 | < 0.001 |

**Transfer Phase Model (2-Key Sequences)**:
| Term | Coefficient | z | p |
|------|-------------|---|---|
| Validity | 0.246 | 18.08 | < 0.001 |
| Item Feedback | -0.200 | -14.68 | < 0.001 |
| Validity x Item | 0.660 | 34.31 | < 0.001 |
| Validity x Trial x Item x Transfer | 0.018 | 5.48 | < 0.001 |

**Interpretation**:
- **Strongly Supported**: Item feedback dramatically increases valid sequence usage
- The 3-way interaction (validity x trial x item) shows item feedback accelerates learning of valid chunks
- During transfer, item-trained models show much stronger preference for valid sequences (validity x item B = 0.66)
- The 4-way interaction shows this effect is modulated by transfer type

### Experiment 3.2: Effect on Learning Accuracy (First 40 Trials)

**Hypothesis**: Item feedback improves model performance (accuracy) during learning.

**Mixed Model**:
- **Outcome**: Trial accuracy (0=incorrect, 1=correct)
- **Predictors**: Item feedback (0=no_item, 1=with items), Trial (centered), Item feedback × Trial interaction

**Learning Accuracy Model**:
| Term | Coefficient | z | p |
|------|-------------|---|---|
| Item Feedback | 0.056 | 14.82 | < 0.001 |
| Trial | -0.002 | -10.66 | < 0.001 |
| Item x Trial | 0.002 | 6.87 | < 0.001 |

**Interpretation**:
- **Supported**: Item feedback significantly improves learning accuracy (B = 0.056, ~5.6% higher)
- Accuracy decreases over trials as task difficulty increases
- The interaction shows item feedback helps maintain accuracy over trials (B = 0.002)

### Experiment 3.3: Effect on Transfer Accuracy (First 40 Trials)

**Hypothesis**: Item feedback improves model performance (accuracy) during transfer.

**Mixed Model**:
- **Outcome**: Trial accuracy (0=incorrect, 1=correct)
- **Predictors**: Item feedback (0=no_item, 1=with items), Trial (centered), Transfer type (0=low, 1=high), and all 2-way and 3-way interactions

**Transfer Accuracy Model**:
| Term | Coefficient | z | p |
|------|-------------|---|---|
| Item Feedback | -0.043 | -14.51 | < 0.001 |
| Trial | -0.0002 | -1.35 | 0.178 (ns) |
| Transfer Type | -0.050 | -16.74 | < 0.001 |
| Item x Trial | 0.0001 | 0.54 | 0.586 (ns) |
| Item x Transfer Type | 0.078 | 18.34 | < 0.001 |
| Item x Trial x Transfer Type | 0.001 | 5.54 | < 0.001 |

**Interpretation**:
- **Partially Supported**: Item feedback shows a complex effect on transfer accuracy
- Main effect of item feedback is negative (B = -0.043), but this is qualified by interactions
- Critical interaction: Item x Transfer Type (B = 0.078) shows item feedback improves high transfer accuracy relative to low transfer
- The 3-way interaction confirms item feedback benefits are stronger in high transfer and improve over trials

---

## 4. Sequence Transfer and Reuse (`sequence_item_transfer_usage.py`)

### Experiment 4.1: 4-Key Sequence Reuse

**Hypothesis**: Previously rewarded 4-key sequences from learning will be reused during transfer.

**Mixed Model** (per condition):
- **Outcome**: OLD sequence flag (0=NEW sequence, 1=OLD sequence from learning)
- **Predictors**: Trial (centered)

**Results (Item condition)**:
| Condition | Intercept | Trial Effect | p |
|-----------|-----------|--------------|---|
| Transfer High | 0.234 | -0.0009 | 0.016 |
| Transfer Low | 0.215 | 0.00006 | 0.87 (ns) |

**Results (No-Item condition)**:
| Condition | Intercept | Trial Effect | p |
|-----------|-----------|--------------|---|
| Transfer High | 0.00 | -0.003 | < 0.001 |
| Transfer Low | 0.126 | -0.005 | < 0.001 |

**Interpretation**:
- With items: ~22% of transfer sequences are OLD (from learning), stable over trials
- Without items: OLD sequence reuse is near zero or rapidly declines
- Item feedback creates persistent memory representations

### Experiment 4.2: 2-Key Valid vs Invalid Usage

**Mixed Model** (per condition):
- **Outcome**: Count of 2-key pairs (valid or invalid)
- **Predictors**: Sequence type indicator (0=invalid, 1=valid), Trial (centered), Sequence type × Trial interaction

**Results (Item condition)**:
- Seq type indicator (valid vs invalid): B = 0.91, z = 79.1, p < 0.001
- Strong preference for valid sequences (~91% higher)

**Results (No-Item condition)**:
- Seq type indicator: B = 0.17-0.19, z = 13-14, p < 0.001
- Weaker but significant preference for valid sequences

**Interpretation**: Item feedback creates much stronger 2-key chunking (5x larger effect size).

### Experiment 4.3: Proportion of Valid 2-Key Pairs Within OLD Sequences

**Hypothesis**: Among trials with valid 2-key pairs, what proportion of those pairs appear within OLD 4-key sequences (learned from training) vs NEW sequences? This experiment measures whether valid pairs are reused compositionally in novel contexts.

**Mixed Model** (per condition):
- **Outcome**: Percentage of valid pairs appearing within OLD sequences (0-100%)
- **Predictors**: Trial (centered)
- **Note**: Only trials with at least one valid pair are included

**Results (Item condition)**:
| Condition | Trial Effect (centered) | z | p |
|-----------|-------------------------|---|---|
| Transfer High | -0.101 | -1.92 | 0.054 (marginal) |
| Transfer Low | 0.048 | 0.95 | 0.343 (ns) |

**Results (No-Item condition)**:
| Condition | Trial Effect (centered) | z | p |
|-----------|-------------------------|---|---|
| Transfer High | -0.493 | -7.94 | < 0.001 |
| Transfer Low | -0.748 | -10.68 | < 0.001 |

**Interpretation**:
- **Strongly Divergent by Condition**:
  - **With items**: 75-80% of valid pairs appear within OLD sequences (verbatim repetition), while 20-25% appear in NEW sequences. This shows the **primary mechanism is memory-based repetition** with modest compositional reuse. The proportion remains stable over trials.
  - **Without items**: Valid pairs appear almost exclusively within OLD 4-key sequences at the start, and this proportion *decreases* significantly over trials (models abandon valid pairs even within OLD sequences).
- The negative trial effects in no-item conditions (B = -0.49 to -0.75, p < 0.001) show models progressively move away from using valid pairs even in previously learned sequences
- Item feedback enables both (1) strong memory of OLD sequences (75-80% of valid pairs) AND (2) modest compositional exploration of valid pairs in NEW sequences (20-25% of valid pairs)
- Without items, valid pair usage collapses entirely - models can't sustain even memory-based reuse

---

## 5. Success vs Failure Case Analysis (`success_failure_cases.py`)

### Experiment 5.1: Learning Phase Success Predicts Exploration

**Hypothesis**: High-success participants show different exploration patterns (more valid 2-key usage) than low-success participants.

**Mixed Model** (per condition):
- **Outcome**: Count of valid 2-key pairs in incorrect trials
- **Predictors**: Group (0=low-success, 1=high-success based on learning phase performance), Trial (centered), Group × Trial interaction

**Results (High transfer condition)**:
- Group x Trial interaction: B = 0.014, z = 6.08, p < 0.001
- High-success participants increase valid sequence usage faster over trials

**Results (Low transfer condition)**:
- Group x Trial interaction: B = 0.005, z = 2.40, p = 0.016
- Similar pattern but weaker effect

**Interpretation**: Successful learners show accelerated discovery of valid action chunks.

### Experiment 5.2: Transfer Success Predicts Learning Exploration

**Hypothesis**: Participants who succeed in transfer show different learning phase exploration.

**Mixed Model** (per condition):
- **Outcome**: Count of valid 2-key pairs in incorrect trials
- **Predictors**: Group (0=low-success, 1=high-success based on transfer phase performance), Trial (centered), Group × Trial interaction

**Results (High transfer)**:
- Group x Trial interaction: B = 0.006, z = 3.29, p < 0.001
- Transfer success associated with faster learning of valid sequences

**Results (Low transfer)**:
- Group x Trial interaction: B = 0.003, z = 1.56, p = 0.12 (ns)
- No significant relationship in low transfer condition

**Interpretation**: Transfer success is predicted by learning phase exploration in high-similarity transfer contexts.

### Experiment 5.3: Learning-Transfer Correlation

**Hypothesis**: Learning success correlates with transfer success.

**Statistical Analysis**: Pearson and Spearman correlation tests (not mixed model)
- Compares learning phase success rates with transfer phase success rates

**Results**:
| Condition | Pearson r | p |
|-----------|-----------|---|
| High | -0.015 | 0.87 (ns) |
| Low | -0.020 | 0.83 (ns) |

**Interpretation**: **Not Supported** - No significant correlation between learning and transfer success. This suggests transfer requires additional capabilities beyond learning performance.

### Experiment 5.4: Valid Pair Proportion in OLD Sequences by Learning Success

**Hypothesis**: High-success participants (based on learning phase performance) show different proportions of valid 2-key pairs within OLD 4-key sequences during transfer compared to low-success participants.

**Mixed Model** (per condition):
- **Outcome**: Percentage of valid 2-key pairs appearing within OLD sequences (0-100%)
- **Predictors**: Group (0=low-success, 1=high-success based on learning), Trial (centered), Group × Trial interaction
- **Note**: Only trials with at least one valid pair are included; top 10 and bottom 10 participants by learning success

**Results (Item condition)**:
| Transfer Condition | N Trials | Group Effect (high vs low) | z | p | Group × Trial | z | p |
|-------------------|----------|---------------------------|---|---|---------------|---|---|
| Transfer High | 577 | ~0.00 | ~0 | 1.0 (ns) | -0.029 | -0.13 | 0.90 (ns) |
| Transfer Low | 616 | ~0.00 | ~0 | 1.0 (ns) | 0.071 | 0.29 | 0.78 (ns) |

**Results (No-Item condition)**:
| Transfer Condition | N Trials | Group Effect (high vs low) | z | p | Group × Trial | z | p |
|-------------------|----------|---------------------------|---|---|---------------|---|---|
| Transfer High | 308 | 49.24 | 4.61 | < 0.001 | -0.88 | -1.72 | 0.085 (marginal) |
| Transfer Low | 329 | 71.86 | 16.85 | < 0.001 | -1.52 | -5.85 | < 0.001 |

**Interpretation**:
- **Strongly Condition-Dependent**:
  - **With items**: No difference between high-success and low-success learners in valid pair reuse from OLD sequences. Both groups show similar compositional exploration patterns.
  - **Without items**: High-success learners show dramatically higher proportions of valid pairs within OLD sequences (49-72 percentage points higher, p < 0.001). This suggests successful learners without item feedback rely heavily on memory-based repetition of learned sequences.
- The negative Group × Trial interaction in no-item LOW transfer (B = -1.52, p < 0.001) indicates that the advantage for high-success learners diminishes over trials - even successful learners eventually abandon valid pair usage.
- **Key insight**: Item feedback enables compositional exploration that equalizes performance between high and low learners. Without items, only the most successful learners can maintain valid pair usage, and they do so primarily through rote repetition rather than compositional generalization.

---

## Summary Table

| Experiment | Hypothesis | Support |
|------------|------------|---------|
| Learning Phase Chunking | Valid sequences increase over trials | Supported |
| Transfer Time Difference | Valid sequences transfer faster | Partially Supported (Item only) |
| High vs Low Transfer | High transfer shows better accuracy | Mixed (depends on item condition) |
| Item Feedback on Chunking | Items improve 2-key chunking | Strongly Supported |
| Item Feedback on Learning Accuracy | Items improve learning performance | Supported |
| Item Feedback on Transfer Accuracy | Items improve transfer performance | Partially Supported (High transfer) |
| 4-Key Sequence Reuse | OLD sequences reused in transfer | Supported (Item only) |
| 2-Key Preference | Valid > Invalid during transfer | Strongly Supported |
| Valid Pairs via Exploration | Valid pairs from NEW sequence exploration | Weakly Supported (Item: 20-25% in NEW, No-Item: 0%) |
| Success Predicts Exploration | High-success shows better chunking | Supported |
| Learning-Transfer Correlation | Learning predicts transfer | Not Supported |
| Valid Pair Reuse by Success Group | High-success shows more valid pair reuse in OLD sequences | Strongly Supported (No-Item only) |

---

## Key Conclusions

1. **Item feedback is critical**: It enables formation of robust action chunks that persist into transfer
2. **Chunking develops over learning**: Valid 2-key sequences increase relative to invalid ones across all conditions
3. **Valid pair usage primarily relies on memory-based repetition (item condition only)**:
   - With item feedback, 75-80% of valid pairs come from OLD sequences (verbatim repetition) while 20-25% appear in NEW sequences (compositional reuse)
   - **Primary mechanism**: Memory-based repetition of learned 4-key sequences (~22% of all transfer sequences are OLD)
   - **Secondary mechanism**: Modest compositional exploration where valid pairs are recombined in novel sequences (NEW sequences average only ~0.15-0.19 valid pairs out of 2 possible)
   - Without item feedback, valid pairs appear almost exclusively in OLD sequences and this usage declines over trials - models cannot sustain even memory-based reuse
4. **Transfer is context-dependent**: High-similarity transfer benefits from item-based learning; low-similarity transfer may not
5. **Learning and transfer are dissociated**: Success in learning does not predict transfer success, suggesting transfer requires additional mechanisms
6. **Item feedback equalizes performance primarily through memory support**:
   - With items: High-success and low-success learners show no difference in valid pair reuse patterns - both groups rely heavily on memory-based repetition (75-80% from OLD sequences) with modest compositional exploration
   - Without items: High-success learners show 49-72 percentage points higher valid pair usage in OLD sequences (p < 0.001), achieved through rote memorization, but even this advantage diminishes over trials
   - This explains why item feedback is critical: it provides scaffolding that allows all learners to build and maintain memory-based representations that persist during transfer, with some compositional generalization as an additional benefit
