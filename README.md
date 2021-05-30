This repository contains analyses and data from the following work:

Samir, F., Beekhuizen, B., Stevenson, S. (to appear). A formidable ability: Detecting adjectival extremeness with distributional semantic models. Findings of the ACL 2021.

# Getting started
1. Run `make_dirs.sh` with `./make_dirs.sh`. This will create some directories where we store
   data and results.
2. Install required packages with `conda env create -f acl_submission.yml`. The code for this project was written with using `conda 4.9.2` and `python 3.7.4`.
3. There are a number of constants in `packages/adverbs/constants.py` that will be referenced in this README. 

# Experiments

## Extremeness classification (Section 3 of paper)
### Important files
The relevant files for this Section are: 
1. `data/acl_spreadsheets/acl_extremeness_predictions.csv`: contains extreme adjectives, frequency-matched non-extreme adjectives, and binary classifier predictions with Distrib. Test and DSM Centroid methods.
2. `main_final_extremeness_eval.py`: the main file for replicating the results reported in Section 3 of the paper.
We also provide a number of pre-computed artifacts for making the reproduction of the main figures easier.
3. `data/acl_artifacts/extremeness_classification_artifacts/extr_clf_all_w_to_embeds.pkl`: word2vec embeddings for the extreme and non-extreme adjectives, for assessing the DSM Centroid method. 
4. `data/acl_artifacts/extremeness_classification_artifacts/query_ea_coca_dms_2adjs_2counts.pkl`: bigram frequencies of (degree modifier, adjective) pairs. 
5. `data/acl_artifacts/extremeness_classification_artifacts/query_ea_coca_freq_dms_and_adjs.pkl`: unigram frequencies of degree modifiers and adjectives.
6. `data/acl_artifacts/extremeness_classification_artifacts/npmi_pr_and_r.pkl`: precision/recall curve coordinates
for the Distrib. method in Figure 2. 
7. `data/acl_artifacts/extremeness_classification_artifacts/npmi_pr_and_r.pkl`: precision/recall curve coordinates
for the DSM centroid method in Figure 2. 

### Steps to replicate: figure only
1. To produce Figure 2, the main figure for Section 3, simply run `python main_final_extremeness_eval.py --produce_final_graph`. This will use the coordinates from files (6) and (7).
2. The result will be stored in `results/{YYYY-MM-DD}/{EXTR_CLF_PRECISION_RECALL_FIG_FNAME}`.

### Steps to replicate: full
1. Run `python main_final_extremeness_eval.py --eval_single_centroid_model`. This will evaluate the DSM centroid method on binary classification of extreme adjectives using the embeddings in (3). 
2. Run `python main_final_extremeness_eval.py --evaluate_adjs_npmi`. This will evaluate the Distrib. test method on binary classification of extreme adjectives using the bigram and unigram frequencies in (4) and (5).
3. The precision/recall curve coordinates from both methods will be stored in `{DYNAMIC_ARTIFACTS_PATH}/{YYYY-MM-DD}`.  
4. Run `python main_final_extremeness_eval.py --produce_final_graph generated`; this will generate Figure 2 using
the newly generated precision recall curve coordinates stored in `{DYNAMIC_ARTIFACTS_PATH}/{YYYY-MM-DD}`.
2. The result will be stored in `results/{YYYY-MM-DD}/{EXTR_CLF_PRECISION_RECALL_FIG_FNAME}`.

## Intensifier emergence prediction 
### Important files
The relevant files for this Section are: 
1. `{ACL_SPREADSHEETS}/intf_prediction_experiment.csv`: shows the intensifiers, control adverbs,
 and predictions from both models. 
2. `{ACL_SPREADSHEETS}/intensifier_dates.csv`: shows intensifiers along with the year where
they were attested to gain an intensifying sense (according to the HTE).
3. `{INTF_PRED_ARTIFACTS_PATH}/intf_experiment_adjs_and_advs.pkl`: contains the frequency
  of the intensifiers and control adverbs per decade. Determined from Google N-grams.
4. `{INTF_PRED_ARTIFACTS_PATH}/match_frame_include_attd.pkl`: pandas DataFrame containing
  intensifiers mapped to their frequency-matched control adverbs. 
5. `{INTF_PRED_ARTIFACTS_PATH}/collated_adj_mod_counts.pkl`: Map of adjectives
 to decades to the number of times the adjectives were modified by an adverb for that decade. Determined 
 from Google Syntactic N-grams.
6. `{INTF_PRED_ARTIFACTS_PATH}/collated_all_adjs_modified.pkl`: Map of adverbs to decade
to the adjectives that the adverb modified in that decade. Determined from Google Syntactic N-grams.
7. `{INTF_PRED_ARTIFACTS_PATH}/adjs_modded_2total_freq.pkl`: Map of adjectives
 to decades to the frequency of the adjectives for the decade. Determined from Google N-grams.
8. `{INTF_PRED_ARTIFACTS_PATH}/feat_extremeness_single_centroid_frame`: pandas DataFrame containing
the data for the Extremeness feature (Section 4.2: Extremeness).
9. `{INTF_PRED_ARTIFACTS_PATH}/feat_simadjmod_frame`: pandas DataFrame containing
the data for the SimAdjMod feature (Section 4.2: Extremeness).
10. `main_simadjmod.py`: Module for containing dependencies for the SimAdjMod feature. Specifically,
it contains code for computing artifacts (5), (6), and (7).
11. `main_ngram_control.py`: Module corresponding to Section 4.1 and 4.2 of the paper. It serves to set up
the analysis reported in Section 4.3 of the paper. 
12. `main_intf_prediction.py`: Module corresponding to Section 4.3 of the paper. Performs intensifier
prediction analysis.

### Steps to replicate: figure only
1. Run `main_intf_prediction.py --predict_intensifier_emergence`. This will perform intensifier prediction
using the provided artifacts (10) and (11).

### Steps to replicate: full
1. Run `main_ngram_control.py --compute_extremeness`. This will compute the extremeness feature (Section 4.2: Extremeness) and store the resulting pandas DataFrame in `{DYNAMIC_ARTIFACTS_PATH}/{YYYY-MM-DD}/INTF_EXTREMENESS_FEATURE_FRAME`. 
2. Run `main_ngram_control.py --compute_simadjmod`. This will compute the SimAdjMod feature (Section 4.2: SimAdjMod) using diachronic word embeddings as well as artifacts (5), (6), (7). The results will be stored in the resulting pandas DataFrame in `{DYNAMIC_ARTIFACTS_PATH}/{YYYY-MM-DD}/INTF_SIMADJMOD_FEATURE_FRAME`. 
3. Run `main_intf_prediction.py --predict_intensifier_emergence generated`. This will perform intensifier
prediction using the dynamically generated artifacts from the previous 2 steps.