# split-fragility

Four modelling approaches on the ten ChEMBL pIC50 data sets and the six
splitting schemes of Lee, Moldagulov and Grzybowski (*Chem. Eur. J.* 2026,
[doi:10.1002/chem.71208][paper]), on that paper's own folds, plus the paper's
naive baseline and its own six arms as published.

The paper's claim is that bioactivity prediction is fragile: that k-nearest
neighbours performs comparably to modern machine learning under every splitting
scheme, and that as the schemes get more stringent every method's accuracy
collapses towards a naive baseline. It tests that on six arms, all of which take
a fingerprint or a molecular graph and learn from the target's own labels alone.
This study asks whether the claim survives two things the paper did not have:
a pre-trained graph representation that is fine-tuned, and a pre-trained
representation that is frozen and read in context, with nothing trained
downstream at all.

11 arms · 10 targets · 6 schemes · 5 folds · 1,500 fold models fitted here

[paper]: https://doi.org/10.1002/chem.71208

## What was compared

| Method | Description | Fitted here |
| --- | --- | :-: |
| `knn` | k-NN on ECFP4 Jaccard distance, distance-weighted, k from {1, 3, 5} on the validation split | yes |
| `lgbm` | LightGBM on Morgan count fingerprints, radius 2, 2048 bits, library defaults | yes |
| `chemeleon` | ChemProp D-MPNN, message passing initialised from CheMeleon, then fine-tuned | yes |
| `monroe35` | Monroe's frozen encoder plus TabPFN 3.5 in context, nothing trained downstream | yes |
| `median` | the paper's naive baseline: the training median for every test molecule | yes |
| `pub_knn` … `pub_gnn` | the paper's own k-NN, SVM, RF, XGBoost, MLP and GNN | no — as published |

`chemeleon` and `monroe35` are the same arms `studies/expansion-ml-comparison`
fits under those names, at the same settings, so a fold here is the same model
as a fold there on different molecules. `lgbm` likewise.

Every target in this collection is a single pIC50 column, so every arm is
single-task by construction. There is no multitask variant to compare against;
that question belongs to the other study.

The six `pub_` arms enter as per-fold *metrics*, not predictions, because
predictions were not released. They are numbers taken on trust, they cannot be
re-scored, and every figure and table marks them. They were each given 50 Optuna
trials per fold, while the four arms fitted here run at their libraries'
defaults — an asymmetry that favours the published arms and is left uncorrected.

## The protocol

Nothing is split here. The upstream release ships, for each of the ten targets
and each of five folds, six splitting schemes as explicit `train` / `val` /
`test` index lists, and those lists are the protocol. Every arm fits on a fold's
`train` indices and is scored on its `test` indices, so every arm sees identical
training molecules in every fold and a difference in the metrics is a difference
in the method.

The `val` indices are used only where an arm needs them: k for k-NN, early
stopping for ChemProp. LightGBM and Monroe have nothing to tune per fold and no
training loop to stop, so they leave the validation molecules unused rather than
training on molecules the other arms never see. That is the convention
`studies/expansion-ml-comparison` follows.

The six schemes, in the paper's order of stringency:

| Scheme | How the paper builds it | Paper says | Released files are |
| --- | --- | --- | --- |
| Random | no structural constraint | 60 / 20 / 20 | 72 / 8 / 20 |
| Bemis-Murcko scaffold | whole scaffold groups held out | 72 / 8 / 20 | 72 / 8 / 20 |
| Butina cluster | ECFP4 Tanimoto clustering at 0.35, whole clusters held out | 60 / 20 / 20 | 72 / 8 / 20 |
| Diverse (MaxMin 25%) | a MaxMin-selected quarter of the data, then five folds of it | 64 / 16 / 20 of the subset | 72 / 8 / 20 of the subset |
| UMAP cluster | PCA to 50-d, UMAP to 2-d, 15 agglomerative clusters into five groups | 72 / 8 / 20 | 72 / 8 / 20 |
| Time | a per-fold publication-year cutoff | 51 / 6 / 43 on average | 50 / 6 / 42 on average |

The last two columns do not always agree. The paper describes the random and
Butina schemes as 60 / 20 / 20, and every cluster-based file in the release is
72 / 8 / 20. Nothing in this study turns on it — every arm reads the same files,
so the comparison is unaffected — but the training sets are larger than the paper
describes, and a reader going from the text to the numbers should know which one
the numbers came from.

### Which copy of the splits, and why

The upstream release ships the splits **twice**, and the two copies disagree.
`data/splits/indices/*.pkl` carries `{train,val,test}_ids` as Python sets;
`data/splits/*.json` carries them as lists. They agree on the random, scaffold
and time schemes and differ on butina, umap and diverse_25 — for umap the two
test sets do not share a single molecule.

The `.pkl` files are the ones the paper's numbers were computed from. Re-running
the released k-NN on them reproduces **250 of 250** published per-fold MAE and R²
values to within 4 × 10⁻⁹ on the five schemes whose splits are recorded in full;
on the `.json` indices, butina and umap do not reproduce at all. The `.pkl`
indices are therefore the protocol here, and that reproduction is kept as a
standing check in `07_collect_metrics.py` rather than as a note — a regression in
the folds or the metric shows up there first.

Spearman ρ reproduces only to 3.5 × 10⁻³, and that is expected: a
distance-weighted k-NN with k as low as 1 predicts the same value for many
molecules, so the folds are full of rank ties and how a tie is broken moves ρ in
the third decimal. The tolerance is set per metric, not globally.

One scheme needs a repair. `diverse_25` has an **empty** `val_ids` in every one
of the fifty `.pkl` files: upstream carved that scheme's validation set out of
the training pool inside the run and did not record it. The `.json` copy does
record one, and on all fifty combinations the `.json` validation set is a subset
of the `.pkl` training pool, the two test sets are identical, and `.pkl` train
equals `.json` train plus val. Taking the validation molecules from the `.json`
copy therefore contradicts nothing in the `.pkl` one, and it is what gives
ChemProp something to stop early on and k-NN something to pick k on. It is done
once, for every arm, and `00_prepare_data.py` asserts all three relations first.
The fifty k-NN folds that do not reproduce are all on this scheme, and the
`pub_` arms on it trained on a different 237 molecules of the same 263-molecule
pool against an identical test set — comparable, but not identical, and the
report says so where it matters.

`upstream/` carries both copies, the ten raw CSVs and the two released metric
files verbatim, with the source revision in `upstream/UPSTREAM_REV`.

## The statistics

Each scheme gives fifty folds per method — ten targets by five folds — and the
between-target spread is far larger than anything separating the methods. An
MAE difference worth reporting here is a few hundredths of a pIC₅₀ unit; the
target effect is around 0.2. Pooling the fifty raw numbers and running Tukey HSD
on them, which is how the paper aggregates, therefore answers a different
question: whether a method's performance *across targets* is separable, and it
almost never is.

So `split_stats.py` adds two things to the `model_comparison.py` this repository
shares, which is byte-identical to the copy in the other studies:

- **The primary test is paired on the fold.** A fold is one (target, scheme,
  fold) triple, and every arm fitted here saw the identical training molecules
  in it, so the fifty differences between any two arms are genuinely paired and
  the target effect cancels in each one rather than being estimated away. Holm's
  correction is applied within each (scheme, metric) across every pair on the
  page. Three tests go into `head_to_head.csv` and they are not redundant:
  `t` is the headline paired t-test over the fifty folds, `wilcoxon` drops the
  normality assumption on the differences, and `t_target` drops the
  *independence* assumption by averaging each method's five folds within a
  target and pairing on the ten targets instead. That last one matters because
  the time split gives each fold its own year cutoff, so a target's five test
  sets overlap heavily — 28,057 test assignments over 9,486 distinct molecules
  — and fifty differences computed on overlapping molecules are correlated,
  which makes the fold-level test anti-conservative there. The target-level test
  costs power (ten paired observations instead of fifty) so it is the
  sensitivity check rather than the headline, and `08_report.py` prints how often
  the three agree on every run. Where they disagree, the fold-level result is the
  one to distrust.
- **The Tukey figure is drawn on target-centred values**, the standard
  randomized-block adjustment, so its intervals mean what they look like. With a
  balanced design this leaves every method's mean exactly where it was and only
  narrows the intervals. It is very slightly anti-conservative — statsmodels
  does not know nine degrees of freedom went into the block means, about 4% of
  the residual df here — which is why the exact paired test is the primary and
  this is the picture. The raw pooled Tukey is drawn too, beside it, because the
  difference between the two is itself worth seeing.

Where two methods cannot be separated, both are reported as tied. There are no
bold maxima.

### Measuring the schemes rather than naming them

`10_distance_analysis.py` replaces the six-valued scheme label with the quantity
underneath it. For every fold, each test molecule's nearest training neighbour is
found by ECFP4 Tanimoto, and the fold is summarised by the mean of those
similarities — the paper's own MAX-ECFP-SIM. Mean similarity falls from 0.751 on
the random split to 0.393 on the UMAP split, where 40% of test molecules have no
training neighbour above the 0.35 Butina threshold the paper clusters at, so the
paper's stringency ordering is broadly confirmed (time and UMAP swap, and the
diverse split sits closer to the hard end than its position suggests). Fold means
span 0.251 to 0.788 across the 300 folds, so the schemes overlap: the easiest
Butina fold is closer in than the hardest scaffold fold.

That makes the independent variable continuous, and each method's gap to the
naive baseline can then be regressed on it directly. The slope is how fast a
method loses its advantage as the test molecules move away, which is precisely
what the paper claims is the same for every method.

### Do the six schemes vary one thing, or two?

They vary two, and `tables/fold_sizes.csv` says by how much. Four of the schemes
train on **72.0% of a target and test on 20.0%**, and they do it to within a
rounding error — across one target's five folds the largest and smallest
training sets differ by about one molecule on random, scaffold and Butina. UMAP
averages the same 72% but wanders (± 4.7, and a 156-molecule spread within a
target) because whole clusters are assigned greedily. The other two are
different experiments: **diverse trains on 18% of a target (242 molecules
against 968) and time on 50% (672)**.

That matters because training-set size and distance travel together: Spearman
ρ = +0.38 between them over all 300 folds. Within the four size-matched schemes
it vanishes (ρ = +0.09, p = 0.18). So random → scaffold → Butina → UMAP is a
clean four-point experiment — similarity falls 0.751 → 0.393 with training data
held fixed — and diverse and time confound moving the test set away with taking
the training data away. A model that does worse on the diverse split than on the
random one has been handed a quarter of the data *as well as* a harder test set.

Nothing is dropped on that account; the paper's six schemes are what this study
set out to run. But `distance_decay.csv` carries the decay fitted both ways —
pooled over all six, and refitted on the 200 folds of the size-matched four —
and where the two disagree the four-scheme version is the one to believe.

One further quantity carries the paper's actual argument: the **gap to the naive
baseline**, per fold. A raw MAE cannot be read across schemes, because the
schemes select structurally different training sets and the naive baseline drifts
with them — the paper notes this and the numbers here bear it out. The gap can be
read across schemes, and it is paired on the fold, so nothing about a target's
difficulty is in it.

## Running it

```bash
python 00_prepare_data.py          # master table, folds, fingerprints
python 01_run_knn.py               # 300 folds, seconds
python 02_run_lightgbm.py          # 300 folds, about a minute
./run_chemprop.sh                  # 300 CheMeleon fine-tunings, four GPU workers
python 04_run_monroe.py --embed    # 13,257 molecules, once
python 04_run_monroe.py            # 300 in-context fits
python 05_run_median.py            # the naive baseline
python 06_import_published.py      # the paper's six arms
python 07_collect_metrics.py       # -> results/fold_metrics.csv
python 08_report.py                # figures and tables
python 10_distance_analysis.py     # what "stringent" measures out at
python 09_build_page.py            # -> results/report.html
```

Steps 3 and 4 each want their own environment; see `requirements.txt`. Step 3 is
the long pole — 300 CheMeleon fine-tunings, about three hours across four workers
on one RTX 5070 Ti, against eight and a half in a single process. Every runner
skips folds that already have a prediction file, so any of them can be
interrupted and relaunched.

## Data and licensing

The ten data sets are the ChEMBL 30 bioactivity collections assembled by Janela
and Bajorath and redistributed in the upstream release, which states that all
code and data are open-sourced but carries no licence file. They travel with this
study in `upstream/` so the folds can be reproduced; the underlying measurements
are ChEMBL's, released under CC BY-SA 3.0. Please cite the paper and ChEMBL.

No third-party pre-trained weights are vendored. CheMeleon is fetched by chemprop
from its authors' release; the Monroe checkpoint comes from the authors' git-LFS
checkout; TabPFN 3.5's weights are licence-gated and need a `TABPFN_TOKEN`
accepted for that version.
