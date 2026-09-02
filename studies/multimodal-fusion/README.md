# multimodal-fusion

Thirty-three ways of combining four molecular representations, run over fifteen
ADME and physicochemical endpoints across two unrelated data sets, 25 replicate
models each, every one scored on a held-out test set it never saw.

This is a reproduction of the design in Wasswa, Kajjumba and Ramsundar,
*Unimodal vs Multimodal Learning* (J. Chem. Inf. Model. 2026), on ADME data
instead of environmental-chemistry data, and on the folds of
[expansion-ml-comparison](../expansion-ml-comparison) so that a fingerprint
baseline and three D-MPNNs, already scored on exactly these molecules, can be
read on the same axis.

The sweep is finished. The report is at
[docs/reports/multimodal-fusion.html](../../docs/reports/multimodal-fusion.html),
or on the
[web](https://patwalters.github.io/model-validation-central/reports/multimodal-fusion.html).

## What came out

Their central finding replicates. Fusing modalities moves accuracy very little,
and none of it survives a correction for multiple comparisons. Stepping up the
modality ladder separates in 0 of 24 comparisons on either data set after Holm
correction, and so does the choice of final learner. On Biogen the single best
configuration in the whole grid is a unimodal one.

What fusion does buy is calibration. Against the best unimodal model, the best
multimodal model improves all three uncertainty measures on both data sets, and
on ExpansionRx all three survive the correction. The best fusion configuration
reaches an error-uncertainty correlation of 0.229 on ExpansionRx and 0.303 on
Biogen, where the best of the four reference methods manages 0.182 and 0.189.

Two things the paper could not see from inside its own design also show up. A
single well-initialised graph network beats all thirty-three configurations on
both data sets: ChemProp initialised from CheMeleon averages 0.407 R² on
ExpansionRx and 0.376 on Biogen, against 0.324 and 0.348 for the best of the
grid. And the released code's GNN modality is not a learned representation at
all; making it one is worth +0.073 R² on ExpansionRx, and it inverts the paper's
ranking of which modality matters. With a real graph embedding the GNN block
carries 49.9% of the fused model's SHAP attribution against RDKit's 4.6%, and
dropping it costs -0.170 R² under early fusion where dropping RDKit costs -0.009.

The late-fusion leak is real and small: refitting the identical meta-learner on
the held-out fifth instead of on training predictions is worth +0.011 R² on
ExpansionRx and +0.013 on Biogen.

## What is being compared

Their design has three axes, and the whole grid is run rather than a sample of
it, because the paper's claims are about the shape of the grid rather than about
any one cell.

| Axis | Levels |
| --- | --- |
| modality | RDKit descriptors, Mol2Vec, a supervised GNN's graph embedding, a character BiGRU's SMILES embedding |
| modality set | each one alone, then GNN+RDKit, +Mol2Vec, +SMILES, and all four |
| fusion | early (concatenate feature vectors) or late (stack per-modality predictions for a meta-learner) |
| final learner | LightGBM, random forest, or AttentiveFP |

Nine unimodal baselines and twenty-four fusion models: 33 configurations per
endpoint per fold, 7,425 fits on ExpansionRx and 4,950 on Biogen.

The paper's fifth modality is MS2 fragmentation spectra. No ADME collection
carries them, so that modality is dropped and nothing else about the design
changes.

## The reference methods

`lgbm`, `chemprop_st`, `chemprop` and `chemeleon` are not fit here. Their
predictions are copied from
[expansion-ml-comparison](../expansion-ml-comparison), where they were produced
on these same fold files, and they run through the same collection and reporting
code. They are fixed points, not entries in the grid: a fingerprint model, a
single-task D-MPNN, a multitask D-MPNN, and a D-MPNN initialised from a
foundation model.

## Evaluation protocol

Unchanged from that project. The `ds` column of each data file fixes a train and
test split. The 25 replicates come from five repeats of a five-fold `GroupKFold`
over the training molecules, grouped by `cluster`. Each fold fits on four fifths
of the training set, early-stops on the held-out fifth where there is a training
loop to stop, and is scored on the same untouched test set.

Every configuration sees identical training molecules in every fold, and so did
the four reference methods when they were run. A difference in the metrics is a
difference in the method.

### Uncertainty

The paper estimates epistemic uncertainty from three independently seeded models.
Here the five folds within a repeat play that role: each test molecule is
predicted five times, by five models fit on overlapping four fifths of the same
training set, and the spread of those predictions is the ensemble sigma. Five
repeats give five ensembles per configuration rather than one, it costs nothing
extra, and it means the four reference methods can be scored on calibration
without being rerun.

The four quantities are the paper's, unchanged: mean epistemic sigma, the Pearson
correlation of absolute error with sigma, a regression-style expected calibration
error over eight equal-frequency bins, and the L1 area between the marginal
quantile curves of error and sigma.

That ECE compares an error magnitude directly against a standard deviation, so a
perfectly calibrated Gaussian scores about 0.2 sigma rather than zero, and it
carries the units of the endpoint. It is a within-panel comparison, not a number
to carry to another paper.

### Statistics

Two levels, kept apart because they answer different questions.

Across endpoints is the paper's own unit: one number per endpoint per
configuration, and a two-sided Wilcoxon signed-rank test with a Holm correction
across each family of comparisons. Within an endpoint, where 25 paired replicates
exist and the paper's single 80/20 split has none, Tukey HSD is used instead,
following [Even More Thoughts on ML Method
Comparisons](https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html)
and the protocol paper it points at. There are no bold maxima.

## This is a reimplementation, and three things were changed

The authors' code is MIT-licensed and public, which is more than most releases
offer. It is also a set of Colab dumps: hard-coded `/content` paths, six empty
notebook stubs, top-level scripts that consume globals left behind by other
scripts, and no committed generator for the feature matrix everything loads. It
cannot be run, so this is written from that source read as a specification,
together with Sections S1 to S7 of the paper's Supporting Information.

Three deviations are deliberate and are the reason the report can say anything
the paper does not.

**The GNN embedding is a learned representation here, and is not one there.**
Their extractor hooks the first `nn.Linear` it finds by module order. In
DeepChem's AttentiveFP that is `gnn.init_context.project_node[0]`, the projection
applied to raw atom features *before any message passing*, so what it returns is
a 30-wide mean of unlearned atom features. Their own Table S3 lists the GNN
modality as 30 features while calling it a learned graph representation; the
readout is 200. This repository uses the 200-wide readout and reruns the
LightGBM configurations on the 30-wide block as a control, so the difference is
measured rather than argued about.

**Late fusion's meta-features leak, and the leak is measured.** Their base
learners fit the whole training set and then predict that same training set;
there is no out-of-fold scheme anywhere in the release. With LightGBM base
learners those training predictions nearly interpolate, so the meta-learner is
fit on optimistically biased inputs. The released procedure is reproduced as
written, and a control fits the identical meta-learner on the fold's held-out
fifth instead. Nothing else differs between them.

**Both encoders early-stop.** Theirs train for a fixed 50 and 20 epochs with no
validation monitoring, which they can afford because their protocol has no
validation split. This one has one, and every other method in the comparison uses
it.

Two smaller things. AttentiveFP comes from PyTorch Geometric rather than
DeepChem-on-DGL, because DGL pins a torch too old to share an environment with
anything else here; the architecture is the same. And embeddings are always
extracted over an unshuffled sequence: their `fusion_early.py` pulls training
SMILES embeddings from a loader built with `shuffle=True`, so that block comes
back in a different order from the three it is concatenated with.
`fusion_late.py` keeps a separate unshuffled loader and does not have the
problem.

## Hyperparameters

The paper's protocol, unchanged: 60 sampled settings from their search spaces,
scored by mean squared error over a 3-fold split, searched once and reused for
every replicate. Two adaptations. The inner folds are grouped by chemical
cluster, the rule the outer folds already use, so a chemotype cannot straddle the
inner boundary either. And the search runs on the fitting molecules of repeat 0
fold 0, because the GNN and SMILES blocks only exist relative to a fold. Nothing
in the search touches the test set, and the settings are fixed across all 25
folds.

AttentiveFP is not searched. The released code hardcodes DeepChem's defaults at a
fixed 50 epochs and never searches it, and a 60-point search over a network that
has to be retrained per fold would cost more than the rest of this repository put
together.

## Running it

`data/`, `folds/` and `predictions/` are not tracked; every one of them is
rebuilt by the scripts below. The GPU half and the CPU half are separate drivers
because they contend for nothing.

```bash
pip install -r requirements.txt

for ds in expansion biogen; do
  ADME_DATASET=$ds python 01_make_folds.py             # fold assignment
  ADME_DATASET=$ds python 01b_make_single_task_folds.py
  ADME_DATASET=$ds python 02_modality_cache.py         # RDKit + Mol2Vec, cached once
done

./run_gpu.sh    # encode every fold, then the graph meta-learners
./run_cpu.sh    # tune, then the tabular grid, then the controls and ablation
```

Then, from the analysis machine:

```bash
for ds in expansion biogen; do
  ADME_DATASET=$ds python 06_collect_metrics.py
  ADME_DATASET=$ds python 07_uncertainty.py
  ADME_DATASET=$ds python 10_report.py
  ADME_DATASET=$ds python 11_contribution_report.py
done
python 12_build_page.py
```

That is how it was actually run: the sweep on a machine with an RTX 5070 Ti and
32 cores, the report locally.

Every step is resumable. A fold whose output file exists is skipped, so any of
them can be stopped and restarted.

### Mol2Vec needs a checkpoint

`02_modality_cache.py` reads the pre-trained 300-dimensional word2vec model of
Jaeger, Fulle and Turk. Download `model_300dim.pkl` from
[samoturk/mol2vec](https://github.com/samoturk/mol2vec) and point
`MOL2VEC_MODEL` at it. The `mol2vec` package itself is not required; it is
unmaintained and pins a gensim old enough to conflict with scipy, so
`featurize.py` reimplements the twenty lines of it that matter.

## Data

Both collections come from the sibling project unchanged, including their
`cluster` and `ds` columns, so the splits are identical by construction rather
than by reimplementation.

`expansion_log_scaled.csv` holds 7,608 molecules on nine endpoints, from the
[OpenADMET-ExpansionRx Blind
Challenge](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data),
contributed by Expansion Therapeutics and released under CC BY 4.0.

`biogen_adme_3521.csv` holds 3,521 molecules on six endpoints, from
[molecularinformatics/Computational-ADME](https://github.com/molecularinformatics/Computational-ADME),
released by Biogen under the MIT License with the paper of Fang et al.

Both are described in full in the sibling project's README, including the log
transform applied to the ExpansionRx endpoints and the caveat about reading MAE
through it.

## Files

| Path | What it is |
| --- | --- |
| `config.py` | paths, endpoints, the design grid, CV settings, search spaces |
| `featurize.py` | the four modalities as features: descriptors, Mol2Vec, graphs, tokens |
| `nets.py` | AttentiveFP and the SMILES BiGRU, and the graph meta-learner |
| `fusion.py` | modality blocks, early and late fusion, the three learners |
| `folds.py` | which molecules a fold fits, validates and is scored on |
| `analysis.py` | the paper's statistics, and the slices they run over |
| `uncertainty.py` | epistemic sigma, error-uncertainty correlation, ECE, miscalibration area |
| `01_make_folds.py`, `01b_make_single_task_folds.py` | fold assignment, from the sibling project |
| `02_modality_cache.py` | RDKit descriptors and Mol2Vec, cached per molecule |
| `03_encode_folds.py` | the two supervised encoders, once per endpoint and fold |
| `04_tune.py` | the hyperparameter search, once per endpoint and configuration |
| `05_run_grid.py` | the 33 configurations, and both controls |
| `06_collect_metrics.py` | predictions to per-fold R², Spearman ρ and MAE |
| `07_uncertainty.py` | the calibration side, from the same predictions |
| `08_modality_contribution.py` | drop-one ablation and grouped SHAP |
| `09_timing.py` | what each configuration costs, measured on a quiet machine |
| `10_report.py`, `11_contribution_report.py` | the figures and tables |
| `12_build_page.py` | the standalone HTML report |
| `model_comparison.py`, `page_kit.py` | Tukey helpers and page furniture, shared with the sibling project |

## References

1. Wasswa, J.; Kajjumba, G. W.; Ramsundar, B. Unimodal vs Multimodal Learning: A
   Systematic Evaluation of Fusion Strategies and Model Design for Molecular
   Property Prediction and Uncertainty Quantification. *J. Chem. Inf. Model.*
   2026. [doi:10.1021/acs.jcim.6c01878](https://doi.org/10.1021/acs.jcim.6c01878).
   Code at
   [github.com/jwasswa2023/Multimodal_Fusion](https://github.com/jwasswa2023/Multimodal_Fusion)
   (MIT).
2. Xiong, Z.; Wang, D.; Liu, X.; et al. Pushing the Boundaries of Molecular
   Representation for Drug Discovery with the Graph Attention Mechanism.
   *J. Med. Chem.* 2020, 63 (16), 8749-8760.
   [doi:10.1021/acs.jmedchem.9b00959](https://doi.org/10.1021/acs.jmedchem.9b00959).
   AttentiveFP.
3. Jaeger, S.; Fulle, S.; Turk, S. Mol2vec: Unsupervised Machine Learning
   Approach with Chemical Intuition. *J. Chem. Inf. Model.* 2018, 58 (1), 27-35.
   [doi:10.1021/acs.jcim.7b00616](https://doi.org/10.1021/acs.jcim.7b00616)
4. Ke, G.; Meng, Q.; Finley, T.; et al. LightGBM: A Highly Efficient Gradient
   Boosting Decision Tree. *NeurIPS* 2017.
5. Lundberg, S. M.; Lee, S.-I. A Unified Approach to Interpreting Model
   Predictions. *NeurIPS* 2017. SHAP.
6. Graff, D. E.; Morgan, N. K.; Burns, J. W.; et al. Chemprop v2: An Efficient,
   Modular Machine Learning Package for Chemical Property Prediction.
   *J. Chem. Inf. Model.* 2026, 66 (1), 28-33.
   [doi:10.1021/acs.jcim.5c02332](https://doi.org/10.1021/acs.jcim.5c02332)
7. Burns, J. W.; Zalte, A. S.; Abreu, C. R. A.; et al. Deep Learning Foundation
   Models for Low-Data Regimes from Classical Molecular Descriptors.
   *J. Chem. Inf. Model.* 2026.
   [doi:10.1021/acs.jcim.6c01546](https://doi.org/10.1021/acs.jcim.6c01546).
   CheMeleon.
8. Ash, J. R.; Wognum, C.; Rodríguez-Pérez, R.; et al. Practically Significant
   Method Comparison Protocols for Machine Learning in Small Molecule Drug
   Discovery. *J. Chem. Inf. Model.* 2025, 65 (18), 9398-9411.
   [doi:10.1021/acs.jcim.5c01609](https://doi.org/10.1021/acs.jcim.5c01609)
9. Fang, C.; Wang, Y.; Grater, R.; et al. Prospective Validation of Machine
   Learning Algorithms for ADME Prediction: An Industrial Perspective.
   *J. Chem. Inf. Model.* 2023, 63 (11), 3263-3274.
   [doi:10.1021/acs.jcim.3c00160](https://doi.org/10.1021/acs.jcim.3c00160)
10. BitBIRCH-Lean, the clustering behind the `cluster` column.
   [github.com/mqcomplab/bblean](https://github.com/mqcomplab/bblean)
