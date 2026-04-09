# Failure Mode Table

| rank | failure mode | evidence | intervention |
| --- | --- | --- | --- |
| 1 | Clip-to-speaker objective mismatch | train speaker error can be much lower than val speaker error while training is still mostly clip-level | add grouped-speaker batching and speaker-pooled losses |
| 2 | Weak last-mile pooling | the first omega replay on the matched no-physics seed-11 checkpoint lost to legacy pooling (`4.645` vs `4.611` val speaker MAE, `6.312` vs `6.157` test speaker MAE) | keep legacy as primary and redesign pooling before trying to re-promote omega |
| 3 | Reliability underuse | audited feature store already has SNR, speech ratio, clipping, distance, voiced ratio, and duration, but the current replay did not convert that into a better pooled result | keep reliability signals, but do not trust the current pooling recipe as a gain source |
| 4 | Plain shuffle batching wastes speaker structure, but fully grouped batches were too aggressive | Stage 2 showed that `8 speakers x 2 clips` reduced apparent gap but degraded the frontier, suggesting identity diversity was over-compressed | test a hybrid sampler that preserves some same-speaker pairing without collapsing the batch to only a few speakers |
| 5 | Physics is unearned | no strict 3-seed proof that physics beats the no-physics line | reintroduce physics only as a challenger after the no-physics line improves |
