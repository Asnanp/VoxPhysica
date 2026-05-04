# Final Verdict

No Omega stage has been promoted.

## Current Honest Position
- Stage 0 (`baseline_truth`) replay: killed
- Stage 1 (`aggregation_only`) replay: killed
- Stage 2 (`speaker_structured_no_physics`): killed
- Stage 3 one-epoch exploratory speaker-alignment run: killed
- The replay frontier remains the best current no-physics reference:
  - validation speaker MAE: `4.611 cm`
  - test speaker MAE: `6.157 cm`
- `1-2 cm` remains rejected.
- `2.0-2.5 cm` remains aspirational, not evidenced.

## What This Means
- Omega pooling did not help.
- The first grouped-speaker batching line did not help.
- The first quick speaker-alignment line was substantially worse.
- The current route to improvement is redesign, not promotion of the existing Omega variants.
