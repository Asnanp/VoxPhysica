# Experiment Matrix

| stage | mode | variant | description |
| --- | --- | --- | --- |
| stage0_baseline_truth | replay | baseline_truth | Replay the current strict frontier with legacy aggregation only. |
| stage1_aggregation_only | replay | aggregation_only | Enable omega pooling with handcrafted reliability and no retraining. |
| stage2_speaker_structured_no_physics | train | speaker_structured_no_physics | Grouped-speaker batching with no-physics V2. |
| stage2b_hybrid_speaker_structured | train | hybrid_speaker_structured_no_physics | Hybrid speaker batching with higher batch diversity and legacy aggregation. |
| stage3_speaker_alignment | train | speaker_alignment_no_physics | Add pooled speaker loss, consistency, and ranking on the no-physics line. |
| stage3b_height_focused_control | train | height_focused_no_physics | Return to plain shuffled batches and simplify the objective around height. |
| stage3c_height_only_regularized | train | height_only_regularized_no_physics | Height-only no-physics control with stronger regularization and a lower learning rate. |
| stage3d_height_only_slice_aligned | train | height_only_slice_aligned_no_physics | Stage 3c plus slice-aware speaker sampling, pooled speaker losses, and gentle stability smoothing. |
| stage3d_height_only_slice_aligned_stable | train | height_only_slice_aligned_stable_no_physics | Stage 3d with the restart-heavy schedule replaced by monotonic cosine decay; every other Stage 3d component is preserved (slice-aware sampling, pooled speaker losses, smoothing). |
| stage3e_height_only_stable_bin_weighted | train | height_only_stable_bin_weighted_no_physics | Stage 3c backbone with gentle hard-slice weighting, feature smoothing, and monotonic cosine decay. |
| stage3f_height_only_long_stable | train | height_only_long_stable_no_physics | Stage 3c objective and plain shuffled batches, extended to 50 epochs with monotonic cosine decay. |
| stage4_learned_reliability | train | learned_reliability_no_physics | Switch from handcrafted reliability to the learned metadata tower. |
| stage5_physics_smart | train | physics_smart | Reintroduce the approved physics components only after the no-physics frontier improves. |
| stage6_flagship | train | flagship_candidate | Combined flagship candidate using only individually promoted Omega components. |
