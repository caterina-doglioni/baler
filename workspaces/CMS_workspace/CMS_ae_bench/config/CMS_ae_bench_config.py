# === AE convergence benchmark ===
#
# Baseline for the VQ-VAE backend comparison. This is a COPY of
# CMS_project_v1_config.py with only the deviations listed below.
#
# CMS_project_v1 itself must not be edited: tests/roundtrip/thresholds.json is
# calibrated against that config at exactly 25 epochs over 20 CI runs, so
# changing it would silently invalidate the round-trip CI gate.
#
# Deviations from CMS_project_v1, and why:
#   epochs            25 -> 500   train to convergence rather than to a fixed
#                                 short budget.
#   test_size         0 -> 0.2    CMS_project_v1 uses test_size=0, which makes
#                                 validation loss a literal copy of training
#                                 loss (see training.train: val_epoch_loss =
#                                 train_epoch_loss when test_size is falsy).
#                                 A real held-out split is needed for
#                                 "converged" to mean anything beyond "training
#                                 loss stopped moving". The split uses
#                                 train_test_split(random_state=1) inside
#                                 helper.process, so it is identical for every
#                                 model benchmarked here.
#   min_delta         0 -> 1e-7   with min_delta=0 any improvement, however
#                                 small, resets EarlyStopping's counter (see
#                                 utils.EarlyStopping), so early stopping can
#                                 effectively never fire. A small positive
#                                 value lets the criterion actually trigger.
#   early_stopping_patience
#                     100 -> 50   paired with the convergence criterion used in
#                                 the benchmark report (relative improvement in
#                                 best validation loss over a trailing 50-epoch
#                                 window below 0.1%).
#   lr_scheduler_patience
#                     50 -> 20    must sit below early_stopping_patience, or
#                                 ReduceLROnPlateau would never fire before
#                                 training stops: at equal patience the run
#                                 ends on the same epoch the first LR reduction
#                                 would have happened, so the anneal never
#                                 contributes. 20 gives roughly two reductions
#                                 (factor 0.5) before the stopping criterion.
#   intermittent_model_saving
#                     False -> True (patience 50)
#                                 crash resilience over a multi-hour run.
#   activation_extraction
#                     True -> False  the forward hooks cost time every batch and
#                                 feed only --mode diagnose, which this
#                                 benchmark does not use.
#   deterministic_algorithm
#                     False -> True  reproducibility, and identical batch
#                                 ordering between the AE and VQ-VAE runs.
#
# Everything else - input_path, compression_ratio (1.6 -> latent 15), lr,
# batch_size, normalization, type_list - is identical to CMS_project_v1.


def set_config(c):
    c.input_path = "workspaces/CMS_workspace/data/example_CMS_data.npz"
    c.data_dimension = 1
    c.compression_ratio = 1.6
    c.apply_normalization = True
    c.model_name = "AE"
    c.epochs = 500
    c.lr = 0.001
    c.batch_size = 512
    c.early_stopping = True
    c.lr_scheduler = True
    c.save_error_bounded_deltas = False
    c.error_bounded_requirement = 10

    # === Additional configuration options ===

    c.early_stopping_patience = 50
    c.min_delta = 1e-7
    c.lr_scheduler_patience = 20
    c.custom_norm = False
    c.reg_param = 0.001
    c.RHO = 0.05
    c.test_size = 0.2
    c.extra_compression = False
    c.intermittent_model_saving = True
    c.intermittent_saving_patience = 50
    c.mse_avg = False
    c.mse_sum = True
    c.emd = False
    c.l1 = True
    c.activation_extraction = False
    c.deterministic_algorithm = True
    c.type_list = [
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "int",
        "float64",
        "float64",
        "float64",
        "int",
        "int",
    ]
    c.convert_to_blocks = False
    c.separate_model_saving = False
