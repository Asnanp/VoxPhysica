"""Compatibility entry point for the strict VoxPhysica experiment.

The former script mixed train, validation, and test speakers in cross-validation
and included an in-sample neural prediction feature. It is preserved under
archive/final_ensemble_all_data_cv_legacy.py for audit history, but its 1.683 cm
number is not a valid held-out-test result.
"""

if __package__:
    from .run_strict_3cm_research import main
else:
    from run_strict_3cm_research import main


if __name__ == "__main__":
    main()
