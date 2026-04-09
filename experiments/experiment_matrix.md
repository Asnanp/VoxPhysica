# Experiment Matrix

| stage | mode | variant | description |
| --- | --- | --- | --- |
| stage0_baseline_truth | replay | baseline_truth | Replay the current strict frontier with legacy aggregation only. |
| stage1_aggregation_only | replay | aggregation_only | Enable omega pooling with handcrafted reliability and no retraining. |
| stage2_speaker_structured_no_physics | train | speaker_structured_no_physics | Grouped-speaker batching with no-physics V2. |
| stage2b_hybrid_speaker_structured | train | hybrid_speaker_structured_no_physics | Hybrid speaker batching with higher batch diversity and legacy aggregation. |
| stage3_speaker_alignment | train | speaker_alignment_no_physics | Add pooled speaker loss, consistency, and ranking on the no-physics line. |
| stage4_learned_reliability | train | learned_reliability_no_physics | Switch from handcrafted reliability to the learned metadata tower. |
| stage5_physics_smart | train | physics_smart | Reintroduce the approved physics components only after the no-physics frontier improves. |
| stage6_flagship | train | flagship_candidate | Combined flagship candidate using only individually promoted Omega components. |
