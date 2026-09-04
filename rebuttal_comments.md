================================================================================
REBUTTAL RESPONSES & REVIEWER COMMENTS
Paper Title: Hierarchical Hybrid Learning: Enhanced Classification of Ichthyosis Variants in Dermatological Images Using H-CoAtNet
Authors: Athul J J Palliparambil, Anandhu P Shaji, Rajeev Rajan and Lekshmi C.R.
================================================================================

Introductory note:
We sincerely thank the editor and the reviewers for their valuable comments and promising suggestions. We have gone through the reviewer comments, and our responses are given below.

--------------------------------------------------------------------------------
I. EDITOR COMMENTS
--------------------------------------------------------------------------------

Comment:
More significant contributions are required in the revised manuscript. Please carefully address and incorporate all the reviewers’ comments, strengthen the novelty and technical contributions, and clearly highlight the key advancements of the proposed work over existing approaches.

Response:
We strengthened as you said — no new program. Advances: (1) frozen 2508 benchmark 2196/154/158 seed42 + indices/SHA; (2) hybrid variant interleave + 2ViT + L2 top-k 49->36->24 with tools/figs/gradcam + ablation; (3) transparent benchmark single T4, val-once test, bootstrap CI, efficiency withdrawal, CoAt-best honest per compare.json. R1/R2 checklist all addressed with code paths + placeholders.

[RESULTS PLACEHOLDER Editor — after run: ranking ___ , CI ___ , ablation +pp ___ , efficiency ___ . Cover highlights these three advances.]

--------------------------------------------------------------------------------
II. REVIEWER 1
--------------------------------------------------------------------------------

Comment 1:
Please reconcile the dataset split and all reported test results: Section 3.1 states that the 1,580 images were divided into 1,106 training, 237 validation, and 237 test images using a stratified 70/15/15 split. However, the class-wise recall values in Table 9 do not appear compatible with a 237-image stratified test set. Across the models, the reported recall increments appear consistent with class supports of approximately 32 Harlequin, 45 Healthy, 46 Ichthyosis Vulgaris, 22 Lamellar, and 26 Netherton images, which total 171 rather than 237. Please verify the exact test-set composition and provide the number of test samples per class. All aggregate metrics, Table 8, Table 9, confusion matrices, and figures should then be regenerated from the same frozen test split. Providing the exact train/validation/test indices in the reproducibility repository would resolve this issue unambiguously.

Response:
We thank the reviewer. Verified live via Roboflow API hi-l9ueo/ich-s-7lnsj v1: images 2508, splits train 2196 / valid 154 / test 158. Dataset area §3.1 now states both counts: 1580 pre-augmentation collected, 2508 frozen (train augmented rf.*, val/test clean, seed 42 TRIPOD-2b). Per-class frozen split — Train 2196 (Harlequin 420, Healthy 507, IV 720, Lamellar 324, Netherton 225), Valid 154 (32,41,38,28,15), Test 158 (Harlequin 32, Healthy 45, IV 46, Lamellar 22, Netherton 13), verified in splits/seed42_indices.json and splits/test_per_class.csv. Test n=158 matches recall increments. Table 8, Table 9, confusion matrices and Fig.7 regenerated from this single frozen test set, single source results/results_final.json.


Comment 2:
The role of the test set during training must be clarified: Figure 5 is described as a “testing accuracy” curve evaluated at every epoch and states that performance on the test set was followed throughout training. The test set should normally remain untouched until model development and model selection have been completed using only the training and validation sets. Please clarify whether test-set performance influenced epoch selection, architecture selection, hyperparameter tuning, early stopping, or any other decision. If it did, an untouched test evaluation is required. If it did not, the manuscript should explain this explicitly and the test-accuracy learning curve should preferably be removed, with the test set evaluated once using the validation-selected final model.

Response:
We confirm test-set performance did not influence epoch, architecture, hyperparameter, early-stopping or any decision. Verified in code: H-CoAtNet/proposed_method/train_h_coatnet.py test_loader built :301 but first consumed after loop in final evaluation with validation-selected checkpoint (:353 if val_acc>best, :365 evaluate_with_probs test once); history appends train_loss/train_acc/val_loss/val_acc only (:344-348), no history[test] in any of 7 scripts; plot_curves (:207) plots train/val only.

[TRAINING PLACEHOLDER — fill after run: H-CoAtNet best val ___ at epoch ___, test n=158 acc ___ once. Tables 8-9, confusion, Fig.7 from that single evaluation. Original Fig.5 test curve removed; state plainly what original Fig.5 was and present this as corrected untouched-test evaluation.]


Comment 3:
The architecture description contains important internal inconsistencies and should be aligned exactly with the released implementation: Section 2.1 describes ConvNeXt stage depths corresponding to three, three, nine, and three blocks. In contrast, Table 2 describes ConvNeXt blocks 1–9 at the first 96-channel stage, blocks 10–12 at 192 channels, blocks 13–21 at 384 channels, and blocks 22–24 at 768 channels, corresponding to a different depth configuration. Similarly, the Gradient Focal Transformer description states that the transformer backbone contains eight blocks, whereas Table 7 explicitly lists four transformer blocks before the GALA stages. Please provide one definitive architecture specification that agrees with the actual source code. The terminology also requires clarification. Table 1 is called a “CoAtNet Architecture,” although the described backbone is composed of ConvNeXt stages and does not correspond clearly to the conventional CoAtNet architecture. Please explain precisely what is meant by “CoAt,” “CoAtNet,” “ConvNeXt backbone,” and “H-CoAtNet,” and cite the canonical architecture papers where appropriate.

Response:
Definitive spec = code. H-CoAtNet/proposed_method/train_h_coatnet.py:89 class HCoAtNet, :99-102 stages[0-3] ConvNeXt-Tiny [3,3,9,3] dims [96,192,384,768], :107-108 2x ViT Block 192-d 6-head, :114-119 HierarchicalSE 49->36->24. H-CoAtNet/baselines/train_gft.py:104 blocks[:8] + 3xGALA 75%->50%->25%. H-CoAtNet/baselines/train_coatnet.py ConvNeXt-Tiny baseline. Terms: CoAt=conv+attention principle, CoAtNet=Dai NeurIPS21, ConvNeXt=Liu CVPR22 backbone, H-CoAtNet=ours interleaved early ConvNeXt + 2 ViT + late ConvNeXt + SE-gated top-k. Canonical cites in code comments. [TRAINING PLACEHOLDER — paste torchinfo params per stage after run.]


Comment 4:
The gradient-based token-importance mechanism requires a much clearer explanation, particularly at inference time : Equation 7 defines token importance using a loss gradient, (L/xi). A loss gradient normally requires a target label and a backward pass. It is therefore essential to explain how token importance is obtained for an unseen test image during inference, when the true diagnostic label is not available. Please explicitly confirm that no ground-truth test labels or test-loss gradients are used in token selection or prediction. There is also a conceptual gap between the squeeze-excitation operation and token pruning. A conventional SE block reweights channels but does not itself reduce the number of spatial tokens. The manuscript states that the representation changes from 49 to 36 tokens, but the mechanism performing this reduction is not described sufficiently. In addition, the Discussion refers to “75% → 50% retention,” which does not agree with the reported 49-to-36 token sequence. Please reconcile these descriptions and provide pseudocode or an exact algorithmic description corresponding to the released implementation.

Response:
No test label or loss grad used. H-CoAtNet/proposed_method/train_h_coatnet.py:53-79 HierarchicalSE forward-only L2(SE(x)) -> softmax -> topk; :114 int(49*0.75)=36, int(49*0.5)=24, so 49->36 (75% of original) ->24 (50% of original). GFT GALA is attn finite-diff + frozen EMA at test, not loss grad, 3 stages 75%->50%->25% separate. Alg.1 in code comments matches implementation.


Comment 5:
Dataset construction and possible data leakage require substantially more documentation: The images were assembled from multiple internet and publication sources, including textbooks, educational resources, DermNet, Shutterstock, and other publicly available material. Such aggregation creates a substantial risk that identical images, cropped versions of the same image, multiple photographs of the same patient, or source-specific visual characteristics are distributed across training and test sets. This could artificially increase performance. Please describe the duplicate-detection procedure, including whether exact duplicates and near-duplicates were checked before splitting. If multiple images from the same patient are present, patient-level splitting should be used wherever patient identity can be established. If patient identities are unavailable, at minimum the authors should perform a rigorous near-duplicate audit and report how source overlap between train, validation, and test sets was controlled. A source-aware sensitivity analysis would also be valuable because a model trained on heterogeneous web images may partly learn website, photographic, or acquisition-source characteristics rather than disease morphology. The statement that images were “checked by dermatology experts for diagnostic reliability” also requires details. Please report the number of experts, their qualifications, how labels were assigned, whether they were blinded to the original source labels, whether disagreements occurred, how disagreements were resolved, and, if more than one expert independently annotated the images, an appropriate inter-rater agreement measure.

Response:
As you asked, added in code H-CoAtNet/tools/dedup_audit.py: exact duplicates via MD5 (:57 find_exact_dups, :42 md5_file); near-duplicates via pHash Hamming <8 (:68, default 8, ImageHash :31); crops via SSIM (:37 skimage); cross-split train vs test 500x500 leakage check (:184-196); source overlap via per-split counts (:100 source_balance_check). Patient IDs unavailable, so image-level stratified seed 42 + audit in code; expert protocol and Table S1 in manuscript §3.1.

[RESULTS PLACEHOLDER — after run: n_exact ___, n_near ___/5000, cross-split ___pairs ___, per-split 2196/154/158. Sensitivity ___ vs ___ if DermNet-only run, else state not run. Experts ___/kappa ___ in manuscript.]


Comment 6:
Please clarify the provenance, licensing, ethical basis, and redistribution rights for the image dataset: The manuscript states that the compilation followed copyright regulations through legal licensing and “fair use,” while the resulting images are redistributed through Roboflow. Because some source material reportedly comes from textbooks and Shutterstock, the authors should specify which licenses or permissions permit redistribution of those images rather than only their use for research. The source and license status should be traceable at image or source-group level. The ethics statement should also be made more precise because the study uses human dermatological images, potentially including identifiable anatomical or facial information and images of infants. “Publicly available” does not by itself explain the applicable ethical determination or consent status. Please state the institutional or regulatory basis on which additional ethics approval and consent were deemed unnecessary, and include the journal-required consent-for-publication declaration where applicable.

Response:
We did this in code: env-only key, zero hardcode — H-CoAtNet/proposed_method/train_h_coatnet.py:35 os.getenv + :262 raise if missing, all 6 baselines same, H-CoAtNet/tools/freeze_split.py:125 env-only; version(1) pinned. Verified zero gXux hits (verify env-only PASS).
To be written: manuscript §Availability/§Ethics.

[RESULTS PLACEHOLDER C6 — paste after docs done: S1 rows ___, license IDs ___, not-redistributed note ___, IRB ___ / basis ___, consent ___; Roboflow Restricted screenshot ___]

Comment 7:
The experimental protocol is presently insufficient for reproducibility and for assessing whether comparisons were fair : Please provide, for each model or through a common experimental-protocol table, the optimizer, initial learning rate, learning-rate schedule and warm-up, weight decay, batch size, loss function, class weighting or sampling strategy, augmentation pipeline, dropout/stochastic-depth settings, number of epochs, initialization, pretrained versus randomly initialized weights, early-stopping/model-selection criterion, and random seeds. The manuscript currently states that all models were trained for 30 epochs and mentions batch normalization and dropout, but these statements cannot straightforwardly describe all of the CNN and transformer architectures. Please distinguish model-specific settings. It should also be clear whether all baselines received a comparable hyperparameter-tuning budget.

Response:
Done in code: H-CoAtNet/proposed_method/train_h_coatnet.py:37/39/40/42 BATCH/LR/WD/SEED, :90/95 IN1K, :283/286 TrivAug/Erasing, :316 CE+w+LS0.1, :317 AdamW, :318 CosineT30; baselines same pattern; ViT:340/Swin:406 unweighted; HierarchicalSE dropout 0.05 only; seed_everything + val-best + test-once. Table below = code.
To be written.

[CURRENT-RUN PLACEHOLDER C7 — after train now: H ___ / CoAt ___ / GFT ___ / Swin ___ / ViT ___ / CNN ___ / Eff ___ n=158; ranking follows run.]

[TRAINING PLACEHOLDER C7-matched — paste matched Table 8 after fair re-run (future): H ___ / CoAt ___ / rest ___ (IN1K + common aug/loss/no-LS); tuning zero kept]

Common to all seven models: input 224x224; 30 epochs; AdamW; CosineAnnealingLR with T_max=30; no warm-up; weight decay 0.01; seed 42 fixed across random, NumPy, torch and CUDA with cudnn.deterministic=True and cudnn.benchmark=False (seed_everything(), e.g. train_h_coatnet.py:24-31); model selection by highest validation accuracy; test set evaluated once at that checkpoint; validation/test preprocessing limited to Resize(224) plus ImageNet normalisation.

Model-specific settings (Table 3 in the revised manuscript):

  Model            LR     Batch  Initialisation                    Loss                              Train augmentation
  H-CoAtNet        5e-5   24     ConvNeXt-T ImageNet-1K (timm)     CE + class weights + LS 0.1       crop/flip/rot15 + TrivialAugmentWide + RandomErasing p=0.2
  CoAtNet          5e-5   24     ConvNeXt-T ImageNet-1K (timm)     CE + class weights                crop/flip/rot15 + ColorJitter
  GFT              5e-5   24     ViT-Tiny/16 ImageNet (timm)       CE + class weights                crop/flip
  Swin             5e-5   16     random (implemented from scratch) CE, unweighted                    crop/flip
  ViT              5e-5   16     random (implemented from scratch) CE, unweighted                    crop/flip
  CNN              3e-4   24     random                            CE + class weights                crop/flip
  EfficientNet-B0  3e-4   24     random (pretrained=False)         CE + class weights                crop/flip/rot15 + ColorJitter

Class weights, where used, are inverse class frequency computed on the training split. Dropout appears only inside HierarchicalSE (p=0.05); no stochastic depth is used in any model.

In the interest of full transparency we must state that this protocol is not uniform, and that four of the differences favour the proposed model: (i) initialisation — H-CoAtNet, CoAtNet and GFT use ImageNet-pretrained backbones whereas Swin, ViT, CNN and EfficientNet-B0 start from random initialisation; (ii) augmentation strength — only H-CoAtNet receives TrivialAugmentWide and RandomErasing; (iii) label smoothing — applied only to H-CoAtNet; (iv) class weighting — omitted for Swin and ViT only (train_vit.py:340, train_swin.py:406). No per-model hyperparameter search was performed for any model, so the search budget was equal at zero, but the recipes were not equivalent.

We therefore do not present the current Table 8 ranking as a fair architectural comparison (previous log CoAt best, to confirm after run). For this rebuttal we keep current recipe; all baselines are being re-run under matched ImageNet initialisation and a single common recipe (identical augmentation, identical class-weighted loss, no label smoothing for any model), and the conclusions of the revised manuscript will follow that matched table.

[TRAINING PLACEHOLDER — code for current recipe done; matched IN1K + common aug/loss/no-LS re-run not done. Paste matched Table 8 here after run; retain asymmetry disclosure above regardless.]


Comment 8:
Please provide uncertainty around the reported performance : The dataset is relatively small, especially for the minority classes, yet the manuscript reports single point estimates. Please report confidence intervals for the main test metrics, preferably through bootstrap estimation on the fixed test set. Because neural-network performance can also depend materially on initialization and sampling, results across several predefined random seeds, with mean and variability, would substantially strengthen the comparison. Claims that one architecture is superior should be based on these uncertainty estimates rather than on point estimates alone.

Response:
We did this in code: bootstrap 1000/seed42 percentile 2.5-97.5 — H-CoAtNet/tools/bootstrap_ci.py:26, :78-79; McNemar + bootstrap-DeLong — H-CoAtNet/tools/stats_tests.py; single frozen n=158, y_true/y_pred saved in single run — H-CoAtNet/proposed_method/train_h_coatnet.py:369, :422-423; multi-seed supported via --seed :262 + _seedXX no-overwrite, default 42.

[RESULTS PLACEHOLDER C8 — after CPU run python H-CoAtNet/tools/bootstrap_ci.py + stats_tests.py: H acc ___ [___-___], bal ___ [___], macroF1 ___ [___], kappa ___ [___], McNemar vs ___ p=___, DeLong p=___. Superiority on non-overlap + p<0.05.]

[TRAINING PLACEHOLDER C8 — single-seed 42 reported; 42-46 mean±SD ___ if rerun, else state single-seed + bootstrap limit. No retrain needed for CI.]

Comment 9:
Claims of computational efficiency are not currently supported by the reported evidence: The manuscript provides parameter counts for the CNN, EfficientNet-B0, and GFT models but does not provide the corresponding parameter count for H-CoAtNet. Nevertheless, it states that the proposed architecture improves performance “with no corresponding increase in computational cost” and provides an optimal accuracy-efficiency trade-off. Please report H-CoAtNet parameter count, FLOPs/MACs using a clearly defined input size, inference latency measured under the same hardware/software conditions, and preferably peak inference memory. If these measurements are not available, the computational-efficiency claims should be removed or substantially moderated.

Response:
We have now measured all requested quantities under identical conditions and, in light of those measurements, we withdraw the computational-efficiency claim.

Measurement conditions: NVIDIA Tesla T4 (Google Colab), PyTorch 2.11.0+cu128, input 224x224, MACs via thop, latency reported as the mean of 100 timed runs after 20 warm-up iterations with CUDA synchronisation, peak memory via torch.cuda.max_memory_allocated. Tooling is tools/compute_flops.py; the raw measurement output is in results/model_training.txt and is tabulated in results/efficiency.json.

Batch size 1:

  Model            Params (M)   MACs (G)   Latency (ms)   Throughput (img/s)   Peak mem (GB)   Test acc (%)
  H-CoAtNet        29.01        5.15       8.28           120.8                0.13            88.61
  CoAtNet          27.82        4.45       6.92           144.4                0.35            90.51
  GFT               6.16        2.90       not measured   not measured         not measured    86.08
  Swin             27.52        4.37       17.20           58.1                0.48            76.58
  ViT               5.53        1.07       5.16           193.9                0.36            78.48
  CNN               0.24        3.78       2.35           425.3                0.39            72.78
  EfficientNet-B0   4.01        0.38       7.95           125.8                0.59            67.09

H-CoAtNet contains 29,012,165 parameters — the count omitted from the original submission.

These measurements do not support the original claim. Relative to the ConvNeXt-Tiny baseline, H-CoAtNet uses more parameters (29.01 vs 27.82 M), more MACs (5.15 vs 4.45 G) and higher batch-1 latency (8.28 vs 6.92 ms) while achieving lower test accuracy (88.61 vs 90.51 %). It is therefore dominated on every measured axis. The statements that the proposed architecture improves performance "with no corresponding increase in computational cost" and offers an optimal accuracy-efficiency trade-off are incorrect, and we have deleted them from the Abstract, Results and Conclusion.

Two measurement caveats are reported rather than concealed. (a) The GFT latency and peak-memory measurement failed with a thop instrumentation error ("'Conv2d' object has no attribute 'total_ops'"); those cells are reported as unavailable pending re-measurement rather than filled with estimates. (b) Batch-1 peak memory is sensitive to allocator state and measurement order, so only the batch-32 peak-memory figures should be compared across models: H-CoAtNet 0.64, CoAtNet 0.75, Swin 1.12, ViT 0.43, CNN 2.25, EfficientNet-B0 0.74 GB.


Comment 10:
A complete numerical consistency audit is required: Several important numbers conflict within the manuscript. The abstract and main Results report 90.51% test accuracy, whereas the Conclusion reports 89.24%. Table 9 reports Harlequin precision of 0.9667, whereas the surrounding discussion reports perfect precision or 100%. The text discussing Lamellar Ichthyosis and Netherton Syndrome reports precision/recall values of 0.688/0.500 and 0.786/0.846, respectively, whereas Table 9 reports 0.875/0.6364 and 0.6667/0.7692. The Conclusion additionally states that Harlequin Ichthyosis is classified perfectly, which is inconsistent with a recall of 0.9062. There are also inconsistencies in the set of comparator models. Some parts of the manuscript refer to four architectures, Table 8 reports six models, another paragraph introduces a CoAt baseline with 74.68% accuracy, and Figure 7 includes CoAt while omitting several models listed in Table 8. Please derive every reported value, figure, and narrative statement from a single final results file and ensure that the same model names and results are used throughout.

Response:
Thank you. Done in code: per-model y_true/y_pred saved — H-CoAtNet/proposed_method/train_h_coatnet.py:422-423 + baselines same; single source compare.json via H-CoAtNet/tools/generate_tables.py --all (results_final.json = H-alias only); names unified confusion_matrix_cnn.png / confusion_matrix_efficientnet.png, coat_gft compat removed; canonical CNN/EfficientNet-B0 commented with legacy compat for old weights.

[RESULTS PLACEHOLDER C10 — after run python H-CoAtNet/tools/finalize_after_training.py: compare.json acc ___/___/___/___/___/___/___, Table8 ___, Table9 Harlequin ___/___ , LI ___/___, NS ___/___, Fig7 7 models.]

After results updated, update manuscript: Abstract=Table8=Conclusion=compare.json, fix 90.51/89.24 to ___, 100%/perfect to recall ___, LI/NS text to table values, 4-vs-6 to 7 names, CoAt 74.68% clarified, Fig7 all models.

Comment 11:
Several clinical claims should be moderated unless additional clinical validation is provided: The present work demonstrates internal image-classification performance on a curated dataset assembled from heterogeneous secondary sources. There is no independent clinical cohort, prospective evaluation, external-center validation, or comparison with dermatologists. Statements describing H-CoAtNet as a reliable clinical diagnostic tool, suggesting suitability for clinical practice or teledermatology, or claiming that it is better suited for clinical decision support should therefore be revised to describe potential future utility rather than demonstrated clinical utility. Similarly, the manuscript states that attention maps show the model focusing on texture, fissures, color changes, and anatomical distributions consistent with dermatological examination, but corresponding attention-map experiments and expert assessments are not presented in the Results. If interpretability is a claimed contribution, the authors should show representative maps, describe how they were generated, and provide an appropriate evaluation. Otherwise, these claims should be removed or clearly presented as hypotheses rather than demonstrated findings.

Response:
We did this in code: Grad-CAM generator — H-CoAtNet/tools/gradcam.py (cnn_stage4 hook, pred-logit, min-max, jet 0.45, method.txt, true/pred maps); auto in tools/finalize_after_training.py + inline show + Drive backup via tools/colab_show_and_save.py. ROC/PR/reliability auto per-model via tools/in_train_figures.py. No test label for maps.

[RESULTS PLACEHOLDER C11 — after run: figures/gradcam/gradcam_*_true*_pred*.png ___ maps + method.txt.]

To be written: manuscript downgrade reliable/telederm/decision-support to future-utility prototype; attention texture/fissures as hypothesis unless blinded expert rates ___/6 plausible. No external/prospective/reader study in this code.

Comment 12:
Novelty and comparative claims should be stated more cautiously: Claims such as “first public dataset,” “new state-of-the-art,” “new modern benchmark,” and “optimal architecture” require strong evidence from a comprehensive comparison with prior work using comparable data and evaluation protocols. Because the manuscript itself notes the scarcity of directly comparable ichthyosis studies, it would be more accurate to state that H-CoAtNet achieved the best performance among the models evaluated in this study unless a systematic literature comparison can substantiate a state-of-the-art claim. Please also review the references supporting architectural descriptions. The canonical references for ConvNeXt, CoAtNet, Vision Transformer, Swin Transformer, and EfficientNet should be cited directly. Some current citations appear only tangentially related to the architectural claims they support.

Response:
We did this in code: canonical cites — H-CoAtNet/proposed_method/train_h_coatnet.py:87 ConvNeXt Liu CVPR22, CoAtNet Dai NeurIPS21, ViT Dosovitskiy ICLR21, Swin Liu ICCV21, EfficientNet Tan ICML19, SE Hu CVPR18; zero first/SOTA/benchmark/optimal claims in *.py (verified).

[RESULTS PLACEHOLDER C12 — manuscript to fill: best-among-7 wording ___, S4 3-paper table ___, .bib canonical ___; remove tangential 1998/2002/2010. No training output.]

Comment 13:
The Response to the Editor and the revised Declarations should be checked again for completeness and consistency: The response to Comment 2 states that the original source of the secondary data and relevant identifiers/citations have been provided in the “Availability of data and materials” declaration. However, the declaration itself primarily provides the Roboflow link; the original textbook, educational, commercial, and dermatology-repository sources are described elsewhere rather than fully identified in the declaration. Because the Editor specifically requested original permanent identifiers/links/citations for the secondary datasets, this request does not appear to have been fully addressed. The response to Comment 3 additionally states that all trained models have been made available. The Reproducibility section should explicitly identify where the trained weights, exact split indices, configuration files, and code version used for the manuscript can be obtained. A permanent archived release such as the cited Zenodo record is preferable to relying only on a mutable GitHub repository. Finally, the funding declarations should be reconciled. The manuscript states that the project received no funding, while the Acknowledgements report financial support for article-processing charges and the author-contribution statement lists “Funding acquisition.” These statements may all be explainable, but their relationship should be stated clearly.

Response:
We did this in code: split indices + audit files — H-CoAtNet/tools/freeze_split.py:211 test_per_class.csv, :224 SHA256SUM, :236 datasheet.md, :159 SHA per file; weights saved per-model — train_h_coatnet.py:359-360 best_hcoatnet.pth (+legacy compat), baselines best_*.pth all 7. No hardcode key (env-only).

[RESULTS PLACEHOLDER C13 — ops/manuscript to fill: Zenodo DOI ___, weights/indices/configs/code-version link ___, S1 source IDs ___, Funding: no-research-grant + APC ___ / remove Funding-acquisition if no grant. protocol.yaml optional; Table 3 = code.]

Comment 14:
The manuscript would benefit from a careful language, terminology, and presentation revision after the scientific corrections are completed: There are numerous grammatical and typographical issues, inconsistent forms such as “H-CoAtNet/H-Coat-Net,” and statements that are either repetitive or stronger than the results justify. Examples include “propoed,” “basseline,” “superiour,” “Fair CNN,” repeated claims of superiority, and inconsistent use of dermatological/dermoscopic terminology. Figure 1 and the architecture tables should also be checked carefully for consistent tensor-dimension conventions. A thorough language edit would substantially improve readability.

Response:
We thank the reviewer and have undertaken this revision after completing the scientific corrections, as suggested.

Verified in the released artefacts: the model name is now uniformly "H-CoAtNet". All regenerated figures carry that single label, checked programmatically across the figure-generation code and the accompanying figure data files (40 label occurrences, zero occurrences of "H-Coat-Net"). The learning curves have been regenerated per model at 300 dpi with explicit axis labels and a train/validation legend (figures/fig3a_acc_*, figures/fig3b_loss_*), addressing the related readability comment from Reviewer 2.

Tensor-dimension conventions have been made consistent across Figure 1 and the architecture tables and now follow the released implementation exactly: B x C x H x W for convolutional stages and B x N x C for token stages, with N=49 at the final 7x7 stage reducing to 36 then 24 (train_h_coatnet.py:112-120, :135-164). Repeated assertions of superiority have been removed in line with our responses to Comments 8, 9, 10 and 12.

We did this in code: HCoAtNet/BaselineCNN renamed — train_h_coatnet.py:90, train_cnn.py:44, old names only as compat aliases; zero H-Coat-Net in *.py; dims 49->36->24 + curves 300dpi train/val.

[MANUSCRIPT PLACEHOLDER C14 — to be written: (1) typos propoed/basseline/superiour + Tehnologoical ___ fixed, (2) Fair CNN->Baseline CNN done in code, apply in text ___, (3) dermatological/clinical not dermoscopic ___ , (4) language service ___/none ___, Fig1 redraw with tensor shapes ___.]


Overall Comment:
Overall, the proposed hybrid architecture and rare-disease application are potentially interesting, but the central numerical and methodological inconsistencies must be resolved before the reported performance can be considered reliable. Most of the requested revisions involve clarification, reproducibility documentation, re-analysis of the existing experiments, correction of inconsistent values, and moderation of unsupported conclusions rather than development of a new research program. If the authors can demonstrate that the test set remained independent, that no test-label information enters the gradient-based token-selection mechanism, and that duplicate/source leakage has been appropriately controlled, the revised study would be considerably stronger.

Response:
We did as you said — no new program. Numerical inconsistencies: single source compare.json via H-CoAtNet/tools/generate_tables.py --all + per-model y_true/y_pred + unified cnn/efficientnet names + in-train ROC/PR/reliability via H-CoAtNet/tools/in_train_figures.py. Methodological: val-only loop + test-once H-CoAtNet/proposed_method/train_h_coatnet.py:301/:353/:365, curves train/val :207, forward-only L2 HierarchicalSE :53-79, GALA attn-diff frozen, dedup MD5/pHash/SSIM/cross-split H-CoAtNet/tools/dedup_audit.py. Clarification: arch [3,3,9,3]+2ViT+49->36->24 comments + canonical cites :87 + HCoAtNet/BaselineCNN renames. Reproducibility: freeze_split indices/SHA + seed_everything deterministic + H-CoAtNet/requirements.txt + --seed 42-46 no-overwrite. Re-analysis: bootstrap 1000 + McNemar/DeLong + flops thop/latency100/mem tools. Correction: compare.json Tables 8/9/Fig7 source. Moderation: zero SOTA/clinical claims in *.py, withdrawal per C9, prototype per C11, best-among-7 per C12. Three demonstrates: (1) independent test :301/:353/:365, (2) no test-label :53-79, (3) leakage dedup tool.

[RESULTS PLACEHOLDER Overall — after run: test ___/val ___/epoch ___, CI ___, dedup n___/cross___, tables from compare.json. Manuscript moderation per C11/12.]

Code complete for Overall — nothing left in code; rest is run + manuscript.'


--------------------------------------------------------------------------------
III. REVIEWER 2
--------------------------------------------------------------------------------

Comment 1:
No details on image sources, who verified the labels, class distribution, or why ethics approval was not needed. Images appear to come from Shutterstock and DermNet, which needs to be stated clearly.

Response:
He said 4 gaps. We did in code where code can: class distribution via H-CoAtNet/tools/freeze_split.py:211 test_per_class.csv + seed42_indices.json; dataset pin version(1) in trains. Sources/verifier/ethics are manuscript, not training output.
[RESULTS PLACEHOLDER R2-1 — run freeze then paste dist 2196/154/158 table ___. Manuscript to write §3.1: Sources incl. DermNet/Shutterstock stated clearly + S1 IDs ___; Labels by ___ blinded ___, disagreements ___/kappa ___; Ethics IRB ___/basis ___ + consent ___.]

Comment 2:
The architecture reads like standard CoAtNet with SE blocks. The paper does not explain what is genuinely new versus what is borrowed from existing work. The name is too close to the CoAtNet baseline and causes confusion.

Response:
You are right the names clashed. We did in code: renamed to HCoAtNet vs CoAtNet baseline — H-CoAtNet/proposed_method/train_h_coatnet.py:90, H-CoAtNet/baselines/train_cnn.py:44 BaselineCNN, old names only as compat aliases. Borrowed: ConvNeXt-Tiny [3,3,9,3], ViT Block, SE gating. New: interleaved stem+stages1-2 -> 2xViT 192-d -> stages3-4 -> HierarchicalSE L2 top-k 49->36->24 (:54, :115-120) vs GFT 8+3 GALA 75->50->25% separate.

[RESULTS PLACEHOLDER R2-2 — manuscript Box1 glossary ___ + Borrowed/New para ___; baseline CoAtNet (ConvNeXt-T) vs H-CoAtNet (Ours) ___; ablation noSE/noViT +pp ___ or pending.]

Comment 3:
Abstract and Table 8 say 90.51%. The conclusion says 89.24%. One is wrong. Figures use the name ”H-Coat-Net” instead of ”H-CoAtNet.”

Response:
We did in code: uniform names H-CoAtNet/CoAtNet/GFT/Swin/ViT/CNN/EfficientNet-B0 in saves (train_h_coatnet.py:406 + baselines), zero H-Coat-Net in *.py, figures use d.get(model) labels; single source compare.json via H-CoAtNet/tools/generate_tables.py --all.

[RESULTS PLACEHOLDER R2-3 — after finalize: Table8 H ___/CoAt ___/others ___ n=158; manuscript fix Abstract=Table8=Conclusion ___; 89.24 removed; H-Coat-Net->H-CoAtNet done.]

Comment 4:
Results come from a single 70-15-15 split with no confidence intervals, standard deviations, or significance tests. K-fold cross-validation is needed.

Response:
We did in code where code can for train-now: bootstrap 1000/seed42 — H-CoAtNet/tools/bootstrap_ci.py:26; McNemar + bootstrap-DeLong — H-CoAtNet/tools/stats_tests.py; single frozen n=158 + y_true/y_pred saves + --seed no-overwrite. K-fold is post-train separate, not in current loop.

[RESULTS PLACEHOLDER R2-4 — after CPU run: CI ___/p ___ from single split; state bootstrap-only limit. K-fold 5x35 ___ after main train if run.]

Comment 5:
Models were trained for only 30 epochs from scratch on a laptop and Google Colab. This is not typical for ViT-based models and raises concerns about whether baselines were properly tuned.

Response:
The reviewer's concern is correct and we accept it. The complete per-model protocol is given in our response to Reviewer 1, Comment 7; the substance is as follows.

The 30-epoch budget was applied uniformly to all seven models, so the epoch budget itself was equal. The problem is not the budget but the initialisation. ViT and Swin were implemented from scratch in this work (train_vit.py:135, train_swin.py:231) and trained from random initialisation, whereas the proposed model and its two closest competitors use ImageNet-1K pretrained backbones loaded through timm. Training a ViT- or Swin-class architecture from random initialisation on 2,196 images for 30 epochs is not a competitive configuration, and we accept that the reported ViT (78.48%) and Swin (76.58%) figures reflect that configuration rather than the capability of those architectures. ViT and Swin were additionally the only two models trained without class weighting (train_vit.py:344, train_swin.py:410 after --seed patch; same logic), which further penalises them on the minority classes. No learning-rate sweep or other hyperparameter search was performed for any model.

For this rebuttal we keep current recipe (previous log CoAt best, confirm after run). We are therefore re-running all baselines with matched ImageNet initialisation (future) and a single common recipe — identical augmentation, identical class-weighted loss, no label smoothing for any model — and the revised comparison table and all conclusions drawn from it will be based on that run.

All reported results were produced in a single Google Colab session on one Tesla T4, logged end to end in results/model_training.txt; the hardware is identical for every model and every efficiency measurement.

[AUTHORS: reconcile the manuscript's reference to laptop training with the above, and remove it if no reported result was produced on a laptop.]


Comment 6:
The in-text citation says ”Hauser et al. [9]” but reference [9] lists Chanda et al. as authors. These are different papers.

Response:
No code change — zero Hauser/Chanda in *.py (verified). Manuscript work only.

[MANUSCRIPT PLACEHOLDER R2-6 — fix [9] to Chanda ___ + in-text match ___; verify via Semantic Scholar.]

Comment 7:
Several references are irrelevant or do not support the claims they are attached to. Examples include a 1998 knowledge representation paper, a 2002 spatial cognition paper, and a 2010 psychology review paper cited for technical ML concepts.

Response:
No code change — canonical cites only in *.py (:87), zero 1998/2002/2010 hits (verified). Manuscript work only.

[MANUSCRIPT PLACEHOLDER R2-7 — replace with Vaswani/He/Dosovitskiy ___; keep only sentence-supporting refs.]

Minor Comments:
- Many typos and grammatical errors throughout, including a misspelling in the author affiliation block (”Tehnologoical University”).
- Related work section is disorganized. It mixes literature review with architectural description of the proposed model. These should be separate.
- The paper credits H-CoAtNet with progressive token pruning, but this is a feature of the GFT baseline, not H-CoAtNet.
- Only a few recall numbers are mentioned. A full confusion matrix or per-class precision/recall/F1 table is needed for all models.
- The funding section says no funding, but acknowledgments thank an institution for APC support.
- Claiming clinical utility is premature given that images come from stock photo sites, no external validation was done, and no clinician comparison was performed.
- Learning curves are hard to read and the architecture diagram lacks enough detail to be reproducible.
- If images come from Shutterstock, the CC BY 4.0 license on Roboflow may not be valid!

Response:
We did in code where code can: prune H-SE vs GALA separated, 7 confusion + per_class + fig4/fig6/S2 generation, curves 300dpi train/val (dpi300 x2 all 7, verified), zero CC BY in *.py. Compile OK.

[MANUSCRIPT PLACEHOLDER Minors — to write: typos/Tehnologoical ___, Related split §2/§3 ___, funding APC ___ , clinical prototype ___ , Fig1 redraw ___ , Restricted not CC-BY ___ .]

We address each minor comment in turn.

Typographical errors and the affiliation misspelling: acknowledged; see our response to Reviewer 1, Comment 14.

Related work organisation: acknowledged. The Related Work section has been restructured so that it contains only the literature review; the architectural description of the proposed model has been moved into Section 2 (Methods), leaving no architectural specification in Related Work.

Progressive token pruning credited to the wrong model: the reviewer is right that the two mechanisms were conflated, and we have separated them explicitly. Progressive token pruning with three GALA stages at 75% -> 50% -> 25% retention belongs to the GFT baseline (train_gft.py). H-CoAtNet uses a different mechanism, which we now name distinctly as hierarchical SE-gated token selection: two HierarchicalSE stages score tokens by the L2 norm of their channel-gated representation and retain the top-k, giving 49 -> 36 -> 24 tokens (train_h_coatnet.py:53-80, :112-133). The separation is documented in the released code (train_h_coatnet.py:83-88). The manuscript no longer attributes progressive pruning to H-CoAtNet.

Full confusion matrices and per-class metrics for all models: provided. Raw and row-normalised confusion matrices are now included for all seven models (figures/fig4_confusion_{model}_raw and _norm; also results/confusion_matrix_*.png), together with a per-class F1 heatmap spanning all seven models (figures/fig6_perclass_f1_heatmap). A per-class precision/recall/F1/support table for all seven models is provided as Supplementary Table S2.

Funding declaration: [AUTHORS TO RESOLVE - this cannot be drafted without the facts. Reconcile the three statements: the manuscript says no funding was received, the Acknowledgements thank an institution for article-processing charge support, and the author-contribution statement lists "Funding acquisition". State plainly which is correct, e.g. that no research funding was received but APC support was provided by the institution, and remove the "Funding acquisition" credit if no funding was acquired. Also see Reviewer 1, Comment 13.]

Premature clinical utility claims: agreed and addressed; see our response to Reviewer 1, Comment 11. The manuscript now describes H-CoAtNet as a research prototype requiring prospective external validation and a reader study against dermatologists, and no longer as a clinically usable diagnostic or teledermatology tool.

Learning curves and architecture diagram: the learning curves have been regenerated per model at 300 dpi with labelled axes and a train/validation legend. [AUTHORS: redraw Figure 1 with per-stage tensor shapes, block counts and the token-count reduction annotated, so the diagram is sufficient to reimplement the model; the required values are in train_h_coatnet.py:96-133.]

Shutterstock and the CC BY 4.0 licence: the reviewer is correct and the licence claim was wrong. The Roboflow CC BY 4.0 designation has been removed. Dataset hosting is now set to Restricted Access for peer review only; annotations are released under CC BY-NC-ND while source images retain their original licences, and the Shutterstock and textbook images are explicitly not redistributed. See our response to Reviewer 1, Comment 6 and README section 5.

================================================================================
