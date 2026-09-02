# ecfp-pretrain-comparison

A graph network pre-trained to predict Morgan fingerprints, put against the
fingerprints themselves.

PT-GIN (Money-Kyrle et al., [arXiv 2605.10722](https://arxiv.org/abs/2605.10722))
is a Graph Isomorphism Network pre-trained on 462,189 QMugs molecules to predict
each molecule's own 2048-bit hashed ECFP4. Downstream the encoder is frozen, its
per-layer graph-pooled outputs are concatenated, and LightGBM predicts the
endpoint. The authors report that on five of six Biogen ADME regression tasks it
significantly beats hashed ECFP, hashed FCFP, Sort & Slice ECFP and a GIN trained
from scratch.

Here it is run against four methods from
[expansion-ml-comparison](../expansion-ml-comparison), on fifteen endpoints across
two data sets, under that project's protocol: 25 replicate models per endpoint,
every one scored on a held-out test set it never saw, and Tukey HSD rather than a
bolded maximum.

The report is at
[docs/reports/ecfp-pretrain.html](../../docs/reports/ecfp-pretrain.html), or on
the [web](https://patwalters.github.io/model-validation-central/reports/ecfp-pretrain.html).

## What was compared

| Method | Description | Tasks |
| --- | --- | --- |
| `lgbm` | LightGBM on Morgan count fingerprints, radius 2, 2048 bits | one model per endpoint |
| `chemprop_st` | ChemProp D-MPNN trained from scratch | one model per endpoint |
| `chemprop` | the same D-MPNN, trained from scratch | one model per assay family |
| `chemeleon` | the same D-MPNN, message passing initialised from CheMeleon | one model per assay family |
| `ptgin` | a GIN pre-trained on ECFP4, frozen, its embeddings handed to LightGBM | one model per endpoint |

The four baselines are not re-run. Their per-fold predictions are copied over by
`00_import_baselines.py`, so the numbers here are the same numbers that appear in
the other repository and a difference between arms is a difference between
methods rather than between runs.

`ptgin` and `lgbm` are the same pipeline with different inputs — one single-task
LightGBM regressor per endpoint per fold, at library defaults — so the comparison
between those two is a comparison of representations with the predictor held
fixed. That is the comparison the paper is built around, and it is the reason
this arm is a good fit for a protocol that changes nothing else.

## Headline result

It does not reproduce. Not as a failure of the method, but as a much narrower
claim than the paper's.

Tukey HSD over 25 folds, counted across every endpoint and metric combination.

| Method | ExpansionRx, of 27 | Biogen, of 18 |
| --- | --- | --- |
| ChemProp + CheMeleon | 14 alone, 10 tied | 12 alone, 6 tied |
| ChemProp single-task | 3 alone, 0 tied | 0 alone, 2 tied |
| ChemProp multi-task | 0 alone, 4 tied | 0 alone, 4 tied |
| PT-GIN + LightGBM | 0 alone, 6 tied | 0 alone, 1 tied |
| LightGBM + Morgan | 0 alone, 0 tied | 0 alone, 3 tied |

PT-GIN is best alone on nothing, on either data set.

**Against the fingerprints it was trained to predict**, paired over folds, it
wins six endpoints of fifteen and loses six. On Biogen — the paper's own primary
benchmark, where it reports beating hashed ECFP on five of six tasks — it wins
one and loses three.

**Against a graph network with no pre-training** it does much better: six
endpoints to two on ExpansionRx, including +0.405 R² on Caco-2 A→B, where a
from-scratch D-MPNN scores −0.307 and is worse than predicting the training mean.

### What decides which endpoints it wins

Not the amount of data. Over all fifteen endpoints the PT-GIN advantage has
Spearman ρ = −0.01 with the number of measurements (p = 0.98). LOG_MLM has 5,692
measurements and PT-GIN wins 23 folds of 25; LOG_MBPB has 1,426 and it loses.

What predicts it is how well the fingerprints were already doing. Sorted by the
LightGBM baseline, the table sorts itself:

| Endpoint | Data set | Measurements | LightGBM R² | PT-GIN − LightGBM | Folds won |
| --- | --- | ---: | ---: | ---: | ---: |
| LOG_Caco_Efflux | ExpansionRx | 3,777 | −0.135 | **+0.136** | 25/25 |
| LOG_Caco_AB | ExpansionRx | 3,773 | −0.014 | **+0.112** | 22/25 |
| LOG_MLM | ExpansionRx | 5,692 | 0.084 | **+0.090** | 23/25 |
| LOG_RPPB | Biogen | 168 | 0.118 | −0.051 | 7/25 |
| LOG_HPPB | Biogen | 194 | 0.129 | +0.031 | 16/25 |
| LOG_HLM | ExpansionRx | 4,541 | 0.139 | **+0.039** | 19/25 |
| LOG_MPPB | ExpansionRx | 1,756 | 0.229 | −0.026 | 9/25 |
| LOG_MGMB | ExpansionRx | 431 | 0.260 | **+0.162** | 21/25 |
| LOG_SOL | Biogen | 2,173 | 0.275 | −0.016 | 7/25 |
| LOG_HLM | Biogen | 3,087 | 0.326 | **+0.032** | 24/25 |
| LogS | ExpansionRx | 7,298 | 0.335 | **−0.078** | 0/25 |
| LOG_RLM | Biogen | 3,054 | 0.389 | −0.001 | 13/25 |
| LOG_MDR1_ER | Biogen | 2,642 | 0.449 | **−0.027** | 2/25 |
| LOG_MBPB | ExpansionRx | 1,426 | 0.477 | **−0.043** | 8/25 |
| LogD | ExpansionRx | 7,309 | 0.510 | **−0.091** | 0/25 |

Spearman ρ = −0.65 between the baseline R² and the PT-GIN advantage (p = 0.009);
on ExpansionRx alone ρ = −0.73 (p = 0.025). The crossover sits around a baseline
R² of 0.3. Below it PT-GIN helps, above it PT-GIN hurts, and on the two endpoints
where the fingerprints do best it loses **every one of 25 folds**.

The same rule fits the other comparisons. PT-GIN's two losses to from-scratch
ChemProp are LogD and LogS, which are that method's two strongest endpoints. Its
only two wins over from-scratch ChemProp on Biogen are the two plasma protein
binding endpoints, 128 and 109 training molecules, where that method scores
R² of −0.064 and −0.005 and is worse than useless. LOG_MGMB is the one clear
exception, and it is the 431-measurement endpoint with a fold standard deviation
of 0.29.

That is consistent with the embedding being a smoothed, denser ECFP4 and not much
more. It was trained to reconstruct ECFP4 and cannot carry substructure
information ECFP4 does not have; what it has instead is 2,048 to 3,072 dense
correlated coordinates in place of 2,048 sparse counts. Smoothing helps when the
signal is weak enough that variance dominates, and costs resolution when it is
not.

### The checkpoint selection buys nothing

Reproducing the paper's per-task checkpoint choice turned out to be the cheapest
finding here. Across all fifteen endpoints:

- the ten checkpoints span a median of **0.027 R²** on validation;
- the winner beats the runner-up by a median of **0.002 R²** on ExpansionRx and
  0.010 on Biogen;
- **not one of the fifteen endpoints** has a margin as large as one fold's worth
  of the winner's own standard deviation.

Five different checkpoints get chosen on ExpansionRx and four on Biogen, which
looks like per-task variation and is not. A radius-0 network — one token per atom
type, no circular environment at all — wins LogD and three of the six Biogen
endpoints, which is hard to square with the vocabulary carrying signal.

### No pre-training overlap explains it

Neither test set has a single exact or connectivity-block InChIKey match in the
462,189-molecule pre-training corpus. Median nearest-neighbour Tanimoto is 0.372
on ExpansionRx and 0.431 on Biogen. The data set that was never filtered against
is the *less* overlapping of the two, so the asymmetry runs the wrong way to
explain anything.

Label leakage is not possible in either direction: the pre-training target is a
fingerprint computed from the molecule's own structure.

### A note on validation splits, for free

PT-GIN's validation R² on ExpansionRx runs 0.11 to 0.44 above the test R² it then
achieves, on eight of the nine endpoints — 0.720 against 0.420 on LogD. On
Biogen's four well-populated endpoints the same gap is −0.01 to +0.03. Nothing
here selects on validation in a way that could inflate it; this is one fixed
model's honest score on a held-out fifth. The gap is a property of the
ExpansionRx split, which shipped with the challenge and is not cluster-pure,
against the Biogen split, which holds out whole BitBIRCH clusters exactly as the
folds do.

## The checkpoint selection

The paper does not have one PT-GIN. It pre-trains a grid of maximum substructure
radius by vocabulary size and picks, per task, whichever pre-trained model does
best in downstream hyperparameter tuning. Ten of those checkpoints are released,
and on the authors' own Biogen results no one of them dominates: four different
checkpoints win somewhere, and the spread between best and worst on a task is
0.03 to 0.08 R².

Reproducing the method therefore means reproducing the choice. It is made here on
the validation fifth, which is the only place this protocol allows a choice to be
made. For each endpoint, every one of the ten checkpoints is fit on the four
fifths of all 25 folds and scored on the held-out fifth; the checkpoint with the
best mean validation R² becomes that endpoint's PT-GIN, and only then is the test
set touched.

`results/<ds>/ptgin_selection.csv` keeps the whole 10 × endpoint × fold table, not
just the winner, because the margin between the best and the second best is what
says whether the selection step is doing any work.

## Evaluation protocol

Unchanged from the source repository. The `ds` column of the raw file fixes a
train and test split. The 25 replicates come from five repeats of a five-fold
`GroupKFold` over the training molecules, grouped by BitBIRCH `cluster`. Each fold
fits on four fifths of the training set, and every method is scored on the same
untouched test set. Every method sees identical training molecules in every fold.

Statistics follow the approach in [Even More Thoughts on ML Method
Comparisons](https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html).
Fold-level distributions, Tukey HSD corrected for multiple comparisons, and
paired tests using the folds as the pairing. There are no bold maxima.

## Data

Two collections, both from the source repository, selected by `ADME_DATASET`
which defaults to `expansion`.

- **ExpansionRx** — 7,608 molecules, 5,326 train and 2,282 test, nine ADME and
  physicochemical endpoints from the OpenADMET-ExpansionRx blind challenge, put
  on a `log10(x + 1)` scale.
- **Biogen ADME** — 3,521 molecules, 2,463 train and 1,058 test, six endpoints
  from [molecularinformatics/Computational-ADME](https://github.com/molecularinformatics/Computational-ADME),
  already log transformed at source.

The Biogen set is the paper's own primary benchmark, though not with this split:
the authors use 200 repeats of 5-fold Butina-clustered cross validation with no
fixed holdout, where this project holds out whole BitBIRCH clusters once and
cross-validates within what remains. Absolute numbers do not transfer between the
two designs. Rankings within each do.

That difference cuts one other way worth knowing. The authors filtered QMugs at
Tanimoto 0.5 against every benchmark they used, Biogen included, so the released
checkpoints have never seen a molecule closely resembling a Biogen compound.
ExpansionRx was not one of their benchmarks and was not filtered against.
`05_check_pretraining_overlap.py` measures how much that matters.

Label leakage is not possible in either direction: the pre-training target is a
fingerprint computed from the molecule's own structure, so no experimental
measurement of any kind entered the encoder.

## Reproducing the analysis

### From the stored predictions

The figures, tables and report rebuild from `predictions/` without refitting
anything.

```bash
pip install -r requirements.txt
for ds in expansion biogen; do
  ADME_DATASET=$ds python 02_collect_metrics.py
  ADME_DATASET=$ds python 03_report.py
  ADME_DATASET=$ds python 04_ptgin_selection.py
done
python 06_build_page.py          # results/report.html, both sets on one page
```

### From scratch

```bash
for ds in expansion biogen; do
  ADME_DATASET=$ds python 00_import_baselines.py   # folds + the four baselines
done
```

Then the PT-GIN arm, which needs the authors' checkout. It is BSD-3 licensed and
ships the ten pre-trained checkpoints in the repository itself, so nothing has to
be pre-trained:

```bash
git clone --filter=blob:none --no-checkout \
    https://github.com/oxpig/topological-pretraining.git ~/software/topological-pretraining
cd ~/software/topological-pretraining
git sparse-checkout set --no-cone "/*" "!/data/"   # the data/ directory is 1 GB
git reset --hard HEAD
export PTGIN_HOME=~/software/topological-pretraining
```

Their `environment.yaml` pins torch 2.8 and rdkit 2024.09.3, and the package
declares `scikit-learn==1.6.0`, `lightgbm==4.6.0` and `optuna==4.2.1`. Only the
featurisation half of it is used here, but the pins are worth honouring since the
checkpoints were written under them:

```bash
conda create -n ptgin python=3.12 rdkit=2024.09.3 -c conda-forge
conda activate ptgin
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-geometric lightgbm==4.6.0 scikit-learn==1.6.0 optuna==4.2.1 \
            torcheval==0.0.7 pandas scipy pyyaml python-dotenv joblib pyarrow
pip install -e ~/software/topological-pretraining --no-deps
```

Then, per data set:

```bash
ADME_DATASET=$ds python 01_run_ptgin.py --embed          # GPU, about 2 minutes
ADME_DATASET=$ds python 01_run_ptgin.py --select --jobs 24
ADME_DATASET=$ds python 01_run_ptgin.py --jobs 24
```

`run_ptgin.sh` does all three for both data sets in order, and every phase skips
work that is already done, so it can be stopped and restarted.

### What each phase costs

`--embed` standardises every molecule the way the authors standardised their
pre-training corpus — sanitize, fragment parent, uncharge, canonical tautomer with
chirality kept — then runs all ten checkpoints over them and caches one embedding
matrix per checkpoint, 2,048 to 3,072 dimensions wide. On an RTX 5070 Ti the
7,608 ExpansionRx molecules standardise in 5 seconds and embed in about 11 seconds
per checkpoint.

`--select` is the expensive phase and it is CPU-bound: 10 checkpoints × 225 folds
on ExpansionRx and × 150 on Biogen, 3,750 LightGBM fits in total, on dense
matrices an order of magnitude wider than the sparse Morgan counts the baseline
uses. It parallelises across folds.

`--select` refits the winner rather than caching its test predictions from the
sweep, which costs 375 more fits and saves about 10 GB of prediction files. The
seeds are fixed per fold, so the refit model is the same model.

## What is the authors' and what is not

The encoder, its tokeniser, its checkpoints and the standardisation are all the
authors' own code from their own repository. Two choices are this project's:

1. **LightGBM runs at library defaults.** The paper Optuna-tunes it, 50 trials per
   task. It tunes it identically for every representation it compares, so the
   comparison it draws is unaffected by tuning; leaving it untuned here preserves
   that while keeping the only difference between `lgbm` and `ptgin` the thing
   being studied. Both arms are untuned, and both are LightGBM.
2. **The checkpoint is selected on the validation fifth**, as described above,
   rather than on a dedicated tuning repeat the way the paper does it. This
   protocol has no spare repeat to give it.
