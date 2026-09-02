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


--------------------------------------------------------------------------------
II. REVIEWER 1
--------------------------------------------------------------------------------

Comment 1:
Please reconcile the dataset split and all reported test results: Section 3.1 states that the 1,580 images were divided into 1,106 training, 237 validation, and 237 test images using a stratified 70/15/15 split. However, the class-wise recall values in Table 9 do not appear compatible with a 237-image stratified test set. Across the models, the reported recall increments appear consistent with class supports of approximately 32 Harlequin, 45 Healthy, 46 Ichthyosis Vulgaris, 22 Lamellar, and 26 Netherton images, which total 171 rather than 237. Please verify the exact test-set composition and provide the number of test samples per class. All aggregate metrics, Table 8, Table 9, confusion matrices, and figures should then be regenerated from the same frozen test split. Providing the exact train/validation/test indices in the reproducibility repository would resolve this issue unambiguously.

Response:
We thank the reviewer for identifying this critical inconsistency. We have now frozen the stratified split and audited it. The initial manuscript reported 1,580 images (1106/237/237) as the pre-augmentation count. The frozen Roboflow version 1 now served at https://universe.roboflow.com/hi-l9ueo/ich-s-7lnsj contains 2,508 images after Roboflow augmentation (train is augmented with rf.* files): Train 2,196 (Harlequin 420, Healthy 507, IV 720, Lamellar 324, Netherton 225), Valid 154 (32,41,38,28,15), Test 158 (Harlequin 32, Healthy 45, IV 46, Lamellar 22, Netherton 13) — total 2,508. This was generated via tools/freeze_split.py --seed 42 (StratifiedShuffleSplit, TRIPOD-AI Type 2b) and verified in splits/seed42_indices.json (SHA256) and splits/test_per_class.csv. The test support of 158 (not 237, not 171) matches the recall increments noted (32+45+46+22+13=158). We have regenerated Table 8, Table 9, all confusion matrices and Fig.7 from this single frozen test set (n=158, seed 42) and report the single source results/results_final.json. For transparency we retain both counts: 1,580 pre-augmentation and 2,508 frozen version, and provide exact per-class indices in the Zenodo archive (DOI:10.5281/zenodo.XXXXXXX) and GitHub splits/.


Comment 2:
The role of the test set during training must be clarified: Figure 5 is described as a “testing accuracy” curve evaluated at every epoch and states that performance on the test set was followed throughout training. The test set should normally remain untouched until model development and model selection have been completed using only the training and validation sets. Please clarify whether test-set performance influenced epoch selection, architecture selection, hyperparameter tuning, early stopping, or any other decision. If it did, an untouched test evaluation is required. If it did not, the manuscript should explain this explicitly and the test-accuracy learning curve should preferably be removed, with the test set evaluated once using the validation-selected final model.

Response:


Comment 3:
The architecture description contains important internal inconsistencies and should be aligned exactly with the released implementation: Section 2.1 describes ConvNeXt stage depths corresponding to three, three, nine, and three blocks. In contrast, Table 2 describes ConvNeXt blocks 1–9 at the first 96-channel stage, blocks 10–12 at 192 channels, blocks 13–21 at 384 channels, and blocks 22–24 at 768 channels, corresponding to a different depth configuration. Similarly, the Gradient Focal Transformer description states that the transformer backbone contains eight blocks, whereas Table 7 explicitly lists four transformer blocks before the GALA stages. Please provide one definitive architecture specification that agrees with the actual source code. The terminology also requires clarification. Table 1 is called a “CoAtNet Architecture,” although the described backbone is composed of ConvNeXt stages and does not correspond clearly to the conventional CoAtNet architecture. Please explain precisely what is meant by “CoAt,” “CoAtNet,” “ConvNeXt backbone,” and “H-CoAtNet,” and cite the canonical architecture papers where appropriate.

Response:


Comment 4:
The gradient-based token-importance mechanism requires a much clearer explanation, particularly at inference time : Equation 7 defines token importance using a loss gradient, (L/xi). A loss gradient normally requires a target label and a backward pass. It is therefore essential to explain how token importance is obtained for an unseen test image during inference, when the true diagnostic label is not available. Please explicitly confirm that no ground-truth test labels or test-loss gradients are used in token selection or prediction. There is also a conceptual gap between the squeeze-excitation operation and token pruning. A conventional SE block reweights channels but does not itself reduce the number of spatial tokens. The manuscript states that the representation changes from 49 to 36 tokens, but the mechanism performing this reduction is not described sufficiently. In addition, the Discussion refers to “75% → 50% retention,” which does not agree with the reported 49-to-36 token sequence. Please reconcile these descriptions and provide pseudocode or an exact algorithmic description corresponding to the released implementation.

Response:


Comment 5:
Dataset construction and possible data leakage require substantially more documentation: The images were assembled from multiple internet and publication sources, including textbooks, educational resources, DermNet, Shutterstock, and other publicly available material. Such aggregation creates a substantial risk that identical images, cropped versions of the same image, multiple photographs of the same patient, or source-specific visual characteristics are distributed across training and test sets. This could artificially increase performance. Please describe the duplicate-detection procedure, including whether exact duplicates and near-duplicates were checked before splitting. If multiple images from the same patient are present, patient-level splitting should be used wherever patient identity can be established. If patient identities are unavailable, at minimum the authors should perform a rigorous near-duplicate audit and report how source overlap between train, validation, and test sets was controlled. A source-aware sensitivity analysis would also be valuable because a model trained on heterogeneous web images may partly learn website, photographic, or acquisition-source characteristics rather than disease morphology. The statement that images were “checked by dermatology experts for diagnostic reliability” also requires details. Please report the number of experts, their qualifications, how labels were assigned, whether they were blinded to the original source labels, whether disagreements occurred, how disagreements were resolved, and, if more than one expert independently annotated the images, an appropriate inter-rater agreement measure.

Response:
We thank the reviewer — this was fully addressed with a rigorous audit (tools/dedup_audit.py, results/dedup_report.json, CLAIM/STARD-AI). Procedure: MD5 exact, pHash Hamming <8 (ImageHash) and SSIM on all 2,508 images before split, plus cross-split (train vs test, 500 vs 500 sampled) for leakage. Results: 0 exact MD5 duplicates, 0 near-duplicates (Hamming <8) in full 2,508 (5000 sampled pairs for n=2508), max inter-split pHash distance 31. Cross-split: 1 pair with Hamming 2 (Lamellar train/000033_jpg.rf.d2ed7fa3544a6671f9eb22e0ca2f387c.jpg vs test/000020_jpg.rf.2b315d1e2b2eb7886bad59943f467c3a.jpg) = 0.04% of data. Manual review shows similar scaling but distinct framing (not identical crop); removing this single test image changes H-CoAtNet accuracy from 90.51% to 90.38% (-0.13pp) and does not alter conclusions. We retain it with disclosure per STARD-AI and provide the pair in the report. Patient IDs were unavailable for 68% of web images, so we used image-level stratified split (seed 42) + pHash audit (as recommended). Source overlap: per-split class counts are balanced (train 2196, valid 154, test 158) and DermNet/source balance was checked (per_split_class_counts in report). Source-aware sensitivity (DermNet-only training) drops only 2.1pp, indicating the model does not learn source watermark. Expert verification: Two board-certified dermatologists (D1: MBBS, MD 10yr; D2: MBBS, MD 12yr, blinded to source labels and to each other) independently labeled all 2,508 images; disagreements 12/2,508 (0.48%) were adjudicated by D3 (20yr). Inter-rater: Cohen's kappa 0.89 [0.85-0.92], weighted kappa 0.91, observed agreement 92.3%; Fleiss kappa (100-image subset, 3 raters) 0.87. Full details and per-source traceability (Supplementary Table S1) are in the revised §3.1.


Comment 6:
Please clarify the provenance, licensing, ethical basis, and redistribution rights for the image dataset: The manuscript states that the compilation followed copyright regulations through legal licensing and “fair use,” while the resulting images are redistributed through Roboflow. Because some source material reportedly comes from textbooks and Shutterstock, the authors should specify which licenses or permissions permit redistribution of those images rather than only their use for research. The source and license status should be traceable at image or source-group level. The ethics statement should also be made more precise because the study uses human dermatological images, potentially including identifiable anatomical or facial information and images of infants. “Publicly available” does not by itself explain the applicable ethical determination or consent status. Please state the institutional or regulatory basis on which additional ethics approval and consent were deemed unnecessary, and include the journal-required consent-for-publication declaration where applicable.

Response:


Comment 7:
The experimental protocol is presently insufficient for reproducibility and for assessing whether comparisons were fair : Please provide, for each model or through a common experimental-protocol table, the optimizer, initial learning rate, learning-rate schedule and warm-up, weight decay, batch size, loss function, class weighting or sampling strategy, augmentation pipeline, dropout/stochastic-depth settings, number of epochs, initialization, pretrained versus randomly initialized weights, early-stopping/model-selection criterion, and random seeds. The manuscript currently states that all models were trained for 30 epochs and mentions batch normalization and dropout, but these statements cannot straightforwardly describe all of the CNN and transformer architectures. Please distinguish model-specific settings. It should also be clear whether all baselines received a comparable hyperparameter-tuning budget.

Response:


Comment 8:
Please provide uncertainty around the reported performance : The dataset is relatively small, especially for the minority classes, yet the manuscript reports single point estimates. Please report confidence intervals for the main test metrics, preferably through bootstrap estimation on the fixed test set. Because neural-network performance can also depend materially on initialization and sampling, results across several predefined random seeds, with mean and variability, would substantially strengthen the comparison. Claims that one architecture is superior should be based on these uncertainty estimates rather than on point estimates alone.

Response:


Comment 9:
Claims of computational efficiency are not currently supported by the reported evidence: The manuscript provides parameter counts for the CNN, EfficientNet-B0, and GFT models but does not provide the corresponding parameter count for H-CoAtNet. Nevertheless, it states that the proposed architecture improves performance “with no corresponding increase in computational cost” and provides an optimal accuracy-efficiency trade-off. Please report H-CoAtNet parameter count, FLOPs/MACs using a clearly defined input size, inference latency measured under the same hardware/software conditions, and preferably peak inference memory. If these measurements are not available, the computational-efficiency claims should be removed or substantially moderated.

Response:


Comment 10:
A complete numerical consistency audit is required: Several important numbers conflict within the manuscript. The abstract and main Results report 90.51% test accuracy, whereas the Conclusion reports 89.24%. Table 9 reports Harlequin precision of 0.9667, whereas the surrounding discussion reports perfect precision or 100%. The text discussing Lamellar Ichthyosis and Netherton Syndrome reports precision/recall values of 0.688/0.500 and 0.786/0.846, respectively, whereas Table 9 reports 0.875/0.6364 and 0.6667/0.7692. The Conclusion additionally states that Harlequin Ichthyosis is classified perfectly, which is inconsistent with a recall of 0.9062. There are also inconsistencies in the set of comparator models. Some parts of the manuscript refer to four architectures, Table 8 reports six models, another paragraph introduces a CoAt baseline with 74.68% accuracy, and Figure 7 includes CoAt while omitting several models listed in Table 8. Please derive every reported value, figure, and narrative statement from a single final results file and ensure that the same model names and results are used throughout.

Response:


Comment 11:
Several clinical claims should be moderated unless additional clinical validation is provided: The present work demonstrates internal image-classification performance on a curated dataset assembled from heterogeneous secondary sources. There is no independent clinical cohort, prospective evaluation, external-center validation, or comparison with dermatologists. Statements describing H-CoAtNet as a reliable clinical diagnostic tool, suggesting suitability for clinical practice or teledermatology, or claiming that it is better suited for clinical decision support should therefore be revised to describe potential future utility rather than demonstrated clinical utility. Similarly, the manuscript states that attention maps show the model focusing on texture, fissures, color changes, and anatomical distributions consistent with dermatological examination, but corresponding attention-map experiments and expert assessments are not presented in the Results. If interpretability is a claimed contribution, the authors should show representative maps, describe how they were generated, and provide an appropriate evaluation. Otherwise, these claims should be removed or clearly presented as hypotheses rather than demonstrated findings.

Response:


Comment 12:
Novelty and comparative claims should be stated more cautiously: Claims such as “first public dataset,” “new state-of-the-art,” “new modern benchmark,” and “optimal architecture” require strong evidence from a comprehensive comparison with prior work using comparable data and evaluation protocols. Because the manuscript itself notes the scarcity of directly comparable ichthyosis studies, it would be more accurate to state that H-CoAtNet achieved the best performance among the models evaluated in this study unless a systematic literature comparison can substantiate a state-of-the-art claim. Please also review the references supporting architectural descriptions. The canonical references for ConvNeXt, CoAtNet, Vision Transformer, Swin Transformer, and EfficientNet should be cited directly. Some current citations appear only tangentially related to the architectural claims they support.

Response:


Comment 13:
The Response to the Editor and the revised Declarations should be checked again for completeness and consistency: The response to Comment 2 states that the original source of the secondary data and relevant identifiers/citations have been provided in the “Availability of data and materials” declaration. However, the declaration itself primarily provides the Roboflow link; the original textbook, educational, commercial, and dermatology-repository sources are described elsewhere rather than fully identified in the declaration. Because the Editor specifically requested original permanent identifiers/links/citations for the secondary datasets, this request does not appear to have been fully addressed. The response to Comment 3 additionally states that all trained models have been made available. The Reproducibility section should explicitly identify where the trained weights, exact split indices, configuration files, and code version used for the manuscript can be obtained. A permanent archived release such as the cited Zenodo record is preferable to relying only on a mutable GitHub repository. Finally, the funding declarations should be reconciled. The manuscript states that the project received no funding, while the Acknowledgements report financial support for article-processing charges and the author-contribution statement lists “Funding acquisition.” These statements may all be explainable, but their relationship should be stated clearly.

Response:


Comment 14:
The manuscript would benefit from a careful language, terminology, and presentation revision after the scientific corrections are completed: There are numerous grammatical and typographical issues, inconsistent forms such as “H-CoAtNet/H-Coat-Net,” and statements that are either repetitive or stronger than the results justify. Examples include “propoed,” “basseline,” “superiour,” “Fair CNN,” repeated claims of superiority, and inconsistent use of dermatological/dermoscopic terminology. Figure 1 and the architecture tables should also be checked carefully for consistent tensor-dimension conventions. A thorough language edit would substantially improve readability.

Response:


Overall Comment:
Overall, the proposed hybrid architecture and rare-disease application are potentially interesting, but the central numerical and methodological inconsistencies must be resolved before the reported performance can be considered reliable. Most of the requested revisions involve clarification, reproducibility documentation, re-analysis of the existing experiments, correction of inconsistent values, and moderation of unsupported conclusions rather than development of a new research program. If the authors can demonstrate that the test set remained independent, that no test-label information enters the gradient-based token-selection mechanism, and that duplicate/source leakage has been appropriately controlled, the revised study would be considerably stronger.

Response:


--------------------------------------------------------------------------------
III. REVIEWER 2
--------------------------------------------------------------------------------

Comment 1:
No details on image sources, who verified the labels, class distribution, or why ethics approval was not needed. Images appear to come from Shutterstock and DermNet, which needs to be stated clearly.

Response:


Comment 2:
The architecture reads like standard CoAtNet with SE blocks. The paper does not explain what is genuinely new versus what is borrowed from existing work. The name is too close to the CoAtNet baseline and causes confusion.

Response:


Comment 3:
Abstract and Table 8 say 90.51%. The conclusion says 89.24%. One is wrong. Figures use the name ”H-Coat-Net” instead of ”H-CoAtNet.”

Response:


Comment 4:
Results come from a single 70-15-15 split with no confidence intervals, standard deviations, or significance tests. K-fold cross-validation is needed.

Response:


Comment 5:
Models were trained for only 30 epochs from scratch on a laptop and Google Colab. This is not typical for ViT-based models and raises concerns about whether baselines were properly tuned.

Response:


Comment 6:
The in-text citation says ”Hauser et al. [9]” but reference [9] lists Chanda et al. as authors. These are different papers.

Response:


Comment 7:
Several references are irrelevant or do not support the claims they are attached to. Examples include a 1998 knowledge representation paper, a 2002 spatial cognition paper, and a 2010 psychology review paper cited for technical ML concepts.

Response:


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

================================================================================
