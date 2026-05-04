"""
VocalMorph Differentiable Acoustic Frontend.

This package implements fully differentiable signal processing layers
that replace traditional LPC/Praat-based formant extraction, enabling
end-to-end gradient flow from height prediction error back to feature extraction.

Modules:
    - differentiable_lpc: Neural LPC layer for formant/VTL estimation
    - neural_filterbanks: Learnable gamma-tonal filterbanks
    - source_features: Differentiable F0, jitter, shimmer, HNR extraction
"""
