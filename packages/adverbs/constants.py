from enum import Enum
###### General constants 
DYNAMIC_ARTIFACTS_PATH = "data/artifacts"
ACL_ARTIFACTS_PATH = "data/acl_artifacts" # .pkl files
ACL_SPREADSHEETS_PATH = "data/acl_spreadsheets" 
COCA_VOCAB_SIZE = 218851
USE_PROVIDED = "provided"
USE_GENERATED= "generated"
DYNAMIC_RESULTS_FOLDER = 'results'

###### Constants for extremeness classification (Section 3)
NON_EA_MODS = ['very', 'almost', 'slightly']
TOTALITY_MODS = ['totally', 'absolutely', 'simply', 'positively', 'downright', 'outright'] 
EXTR_CLF_ARTIFACTS_PATH = f"{ACL_ARTIFACTS_PATH}/extremeness_classification_artifacts"
EXTR_CLF_WORD_TO_EMBEDS_FNAME = "extr_clf_all_w_to_embeds"
EXTR_CLF_CENTROID_AUC_ITEMS = "single_centroid_pr_and_r"
EXTR_CLF_NPMI_AUC_ITEMS = "npmi_pr_and_r"
EXTR_CLF_NPMI_DM_ADJ_COLLOCATION_FREQ = "query_ea_coca_dms_2adjs_2counts"
EXTR_CLF_NPMI_DM_ADJ_FREQ = "query_ea_coca_freq_dms_and_adjs"
EXTR_CLF_SPREADSHEET = f"{ACL_SPREADSHEETS_PATH}/acl_extremeness_predictions.csv"
EXTR_CLF_PRECISION_RECALL_FIG_FNAME = "final_extr_eval"

###### Constants for intensifier prediction (Section 4)
INTF_PRED_ARTIFACTS_PATH = f"{ACL_ARTIFACTS_PATH}/intensifier_prediction_artifacts"
ADV_FREQUENCY_PER_DEC = "intf_experiment_adjs_and_advs"
INTENSIFIER_TO_CONTROL_MATCH_FRAME = "match_frame_include_attd"
INTF_TO_ATTD_DATE_CSV = "intensifier_dates.csv"
INTF_PRED_SPREADSHEET = f"{ACL_SPREADSHEETS_PATH}/intf_prediction_experiment.csv"
INTF_EXTREMENESS_FEATURE_FRAME = "feat_extremeness_single_centroid_frame"
INTF_SIMADJMOD_FEATURE_FRAME = "feat_simadjmod_frame"
CTRL_ADVS_SPREADSHEET = f"{ACL_SPREADSHEETS_PATH}/control_adverbs.csv"
ADV_TO_DEC_TO_ADJS_MODDED = f"{INTF_PRED_ARTIFACTS_PATH}/collated_all_adjs_modified"
ADJ_TO_DEC_TO_FREQ_MODDED = f"{INTF_PRED_ARTIFACTS_PATH}/collated_adj_mod_counts"
ADJ_TO_FREQ = f"{INTF_PRED_ARTIFACTS_PATH}/adjs_modded_2total_freq"

###### Constants for leave-one-cluster out extremeness classification (Appendix A)
APPENDIX_EXTR_CLF_ARTIFACTS_PATH =f"{ACL_ARTIFACTS_PATH}/appendix_extremeness_classification_artifacts"
EXTR_ADJ_CLUSTER_SPREADSHEET = f"{ACL_SPREADSHEETS_PATH}/extr_adj_clusters.csv"


###### Constants for intensifier classification extremeness classification (Appendix B)
APPENDIX_INTF_PRED_ARTIFACTS_PATH =f"{ACL_ARTIFACTS_PATH}/appendix_intensifier_prediction_artifacts"
CTRL_ADVS_SPREADSHEET = f"{ACL_SPREADSHEETS_PATH}/control_adverbs.csv"
INTF_ADVS_FULL_SPREADSHEET = f"{ACL_SPREADSHEETS_PATH}/intensifiers_full.csv"
EXTREMENESS_AVGD_FRAME = f"{APPENDIX_INTF_PRED_ARTIFACTS_PATH}/extremeness_avgd_frame_single_centroid"
SIMADJMOD_AVGD_FRAME = f"{APPENDIX_INTF_PRED_ARTIFACTS_PATH}/simadjmod_avgd_results_frame"



class Vec(Enum):
    NAN = 1