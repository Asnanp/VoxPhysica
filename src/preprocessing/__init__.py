from importlib import import_module

from .augmentation import AUDIOMENTATIONS_AVAILABLE, AugmentationConfig, apply_augmentations, build_augmenter
from .audio_enhancement import (
    MicrophoneEnhancementConfig,
    SpeechEnhancementReport,
    WEBRTCVAD_AVAILABLE,
    enhance_microphone_audio,
)
from .feature_extractor import (
    FeatureConfig,
    extract_all_features,
    extract_mfcc,
    extract_praat_features,
    extract_spectral,
    load_audio,
    process_audio_file,
    process_dataset,
)

__all__ = [
    "FeatureConfig",
    "load_audio",
    "extract_mfcc",
    "extract_spectral",
    "extract_praat_features",
    "extract_all_features",
    "process_audio_file",
    "process_dataset",
    "VocalMorphDataset",
    "collate_fn",
    "build_dataloaders",
    "build_dataloaders_from_dirs",
    "AUDIOMENTATIONS_AVAILABLE",
    "AugmentationConfig",
    "build_augmenter",
    "apply_augmentations",
    "MicrophoneEnhancementConfig",
    "SpeechEnhancementReport",
    "WEBRTCVAD_AVAILABLE",
    "enhance_microphone_audio",
]

_LAZY_DATASET_EXPORTS = {
    "VocalMorphDataset",
    "collate_fn",
    "build_dataloaders",
    "build_dataloaders_from_dirs",
}


def __getattr__(name: str):
    if name in _LAZY_DATASET_EXPORTS:
        dataset_module = import_module(".dataset", __name__)
        value = getattr(dataset_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
