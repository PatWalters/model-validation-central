# Prompts

The prompts that shaped this analysis and the report that came out of it, in order.

The work ran as one Claude Code session against the project directory, from 15 to 26 August 2026. These 22 prompts are what a human typed. Everything else, the scripts, the fold files, the 2,175 fold models and the report, came out of them.

## How this list was made

Extracted from the session transcript, then filtered to the prompts that specified, changed, or corrected the work. Left out: requests for progress, questions answered in conversation without changing anything, choices about scheduling and which machine to run on, and a side errand drafting a note to the MEGA-CL authors. The transcript holds 69 prompts in total.

The wording is copy-edited. Spelling and punctuation are corrected, and one link that had been pasted from the wrong tab now points where it was meant to. Local file paths are replaced with plain descriptions. No prompt is reordered, and nothing is added.


## Specifying the comparison

*15 August 2026*

Three arms to begin with: LightGBM on Morgan fingerprints, ChemProp from scratch, and ChemProp initialised from CheMeleon.

**1.**

> Write a python script to perform 5x5 cross validation using the data in @expansion_log_scaled.csv. The training and test sets are specified in the ds column. All the other endpoints are labeled in the csv file. Please list the endpoints that will be considered.

**2.**

> No, don't build the model yet, just tell me what the endpoints are. I'll tell you what to do after that.

**3.**

> I want to go with the data as is. As I mentioned before, I want you to build models and perform 5x5 cross validation on this dataset. At each CV fold I want you to save R^2, spearman rank correlation, and mean absolute error. I want you to build models three ways. 1. LightGBM regression with Morgan count fingerprints. 2. Chemprop. 3. Chemprop with CheMeleon. Make sure all model predictions are stored. Once all the models have been built and evaluated, I want you to report the results using comparisons like the ones in this blog post https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html.

**4.**

> Let's multitask the ChemProp models. Look at the way I set this up in my UNIQUE reproduction.


## The single-task control, and reaching for MEGA-CL

*16 August 2026*

A multi-task model beating a single-task one confounds the architecture with the transfer, so a fourth arm separates them. The first attempt at MEGA-CL stalled: the checkpoint the prompt expected to find on Zenodo had not been released.

**5.**

> Yes, please include a single-task ChemProp arm.

**6.**

> Let's make a couple of changes in the R2 and MAE plots. First, label ChemProp multi-task as ChemProp multi-task rather than ChemProp. Second, let's have the same x-axis range on all the plots. Third, keep the plots in a 3x3 grid, but only put the y-axis labels on the leftmost plot. In other words, share_x and share_y.

**7.**

> OK, time for another comparison. The paper in @2607.24314v1.pdf describes a foundation model, MEGA-CL, for ADMET prediction. The paper has a git repo and the model weights are on Zenodo. Set this model up and compare it with the models we've run so far. Use the same datasets and splits.

**8.**

> Create a version of the table from the "Ranking is healthier than R2" section of report.html. Put this in an image with a white background. Don't show the standard deviation. Just put the best value in bold.


## Running MEGA-CL

*18 to 19 August 2026*

The authors published a checkpoint, so the fifth arm went ahead. 225 fine-tuning runs, 21 hours 32 minutes on a consumer GPU.

**9.**

> The MEGA-CL git repo now includes a checkpoints directory with the pre-trained model. Let's try that again.

**10.**

> Can we add MAE to the analysis?


## Report, references, and the repository

*23 August 2026*

Turning the results into something publishable.

**11.**

> Prepare a report as a shareable artifact.

**12.**

> Update the report to include references to the recent ChemProp and CheMeleon papers as well as the MEGA-CL paper.

**13.**

> Actually, the CheMeleon paper was just published. https://pubs.acs.org/jcisd8/article-abstract/doi/10.1021/acs.jcim.6c01546/5250516/Deep-Learning-Foundation-Models-for-Low-Data Please update.

**14.**

> I'd like to create a GitHub repo so that others can reproduce this analysis. Create the repo and update the report with a link to it.

**15.**

> The dataset doesn't come from Novartis. It comes from here: https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge. I simply did a log transform on all columns except LogD, which was already on a log scale.

**16.**

> Let's add a short explanation of the plots showing Tukey HSD. There's an example at the top of the "Things I Wish People Would Do" section of this blog post: https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html. You can also reference this paper: https://pubs.acs.org/jcisd8/article-abstract/65/18/9398/3687588/Practically-Significant-Method-Comparison

**17.**

> How can I fix the one check worth mentioning?


## Reading the transform, and two more arms

*26 August 2026*

Two questions about the log transform, then two new papers, five days apart.

**18.**

> Is everything really on a log10(x+1) scale? Will that skew the MAE calculation?

**19.**

> I used x+1 because there are zero values, which then throw an exception when converting to log scale.

**20.**

> There is a new paper, monroe_paper.pdf, in this directory. Let's add that one to the analysis. Remember that we'll run the actual analysis on apollo, which has a GPU.

**21.**

> Ok let's now incorporate the method described in Mol_JEPA.pdf.


## A second data set

*28 August 2026*

The comparison had been run once, on one collection.

**22.**

> I'd like to expand this analysis to also use the Biogen dataset from https://github.com/molecularinformatics/Computational-ADME


## A second comparison

*29 August 2026*

A method that is not an architecture, and could not be run as released.

**23.**

> I added a new paper @trimole_hybrid.pdf with antoher method. Let's do a comparison of this one the same way we did with the others. However, in this case, I want to just compare with the 3 ChemProp variants, and LightGBM with Morgan fingerprints. Generate a new report for this. Use the ExpansionRx and Biogen sets again. Run on apollo to speed this up a bit.


## Notes on reading these

A few prompts did more work than their length suggests.

Prompt 4 points at an existing project rather than restating its settings. The multi-task groupings, hyperparameters and cluster-aware validation split were all read from there.

Prompts 13 and 15 are corrections. One updates a citation that had moved from preprint to journal. The other corrects the provenance of the data set, which had been attributed to the wrong source. Both changed what was published.

Prompt 17 asks for a fix to a caveat raised in an answer rather than in a prompt, which is why it reads as a fragment on its own. It produced the balance check in `model_comparison.py`.

Prompt 18 is a question, not an instruction, and it is listed because the answer changed the report. Checking it against the upstream data turned up an offset the report had described wrongly, and produced the MAE caveat panel. Prompt 19 supplied the reason for the offset, which is now recorded alongside it.

Prompt 20 carries one decision that is not in the text. Monroe adapts to a new endpoint through TabPFN, and its published numbers use TabPFN v3, whose weights sit behind a licence. Asked whether to use the gated v3 or the open v2, the answer was v3. That turned out to be the only workable choice anyway: v2 accepts at most 500 features and Monroe hands it 720.

Prompt 21 carries a decision of the same kind. Mol-JEPA's frozen embeddings need a downstream predictor, and the paper tries several. Asked which to use, the answer was to run the authors' own choice as the arm and a second one as a control. That control is what makes it possible to say that the gap to Monroe is the representation and not the head.

Prompt 22 is the one that changed the conclusion. It also forced the only structural rewrite in the project: `config.py` had been built around a single collection, so every path had to be namespaced by data set before a second one could be added. Two choices inside it were settled by asking, and both are recorded in the report: all six Biogen endpoints rather than the four well-populated ones, and a cluster-pure holdout of our own rather than the repository's per-endpoint splits.

A follow-up in the same exchange pointed at the HuggingFace dataset. That turned out to matter more than a convenience: the authors released the whole pre-training table, which is what `11_check_pretraining_overlap.py` reads.

Prompt 23 is the first one whose method could not simply be run. The other
pre-trained arms ship code and weights; this one ships an audit package, with
placeholder paths, no checkpoints, and all rights reserved. That was surfaced
before any compute was spent, because it changes what the report is allowed to
claim: the arm is a reimplementation from the paper, not a rerun of it. The one
decision settled by asking was how much of the candidate pool to build, and the
answer was all four molecular views, which meant obtaining three checkpoints
separately from their original authors.

It also produced the second structural change in the project. `config.py` had
been built around one report covering every method; a second report covering a
different subset meant naming the comparisons and namespacing the figures by
them. The seven-method report was verified byte-identical afterwards.

This file was itself asked for, and then revised, by three further prompts. They are not listed above, since they produced the file rather than the analysis.

