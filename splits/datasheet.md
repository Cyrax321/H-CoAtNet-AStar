# H-CoAtNet Dataset Datasheet (TRIPOD-AI, STARD-AI)

- **Source:** Roboflow hi-l9ueo/ich-s-7lnsj v1 (2,508 images after frozen augmentation; 1,580 pre-augmentation collected)
- **Split:** Frozen stratified seed 42, TRIPOD-AI Type 2b (test held-out, evaluated once)
- **Counts:** Train 2196 / Valid 154 / Test 158 (n=2508)
- **Per-class test (n=158):** Harlequin 32, Healthy 45, Ichthyosis Vulgaris 46, Lamellar 22, Netherton 13 (see test_per_class.csv)
- **Manifest SHA256:** f5a86fc2390bfd3d7a36cc36e28e24f83ee8303c93b88c555fe37b82b653b341
- **Dedup:** MD5 0 exact, pHash<8 0 near (5k sampled pairs), cross-split 6 IV pairs Hamming 0-2 disclosed (see results/dedup_report.json)
- **Note:** Patient IDs unavailable for 68% web images; image-level split + audit. Shutterstock/textbook images not redistributed (Restricted Access).
- **Generated:** 2026-09-09
