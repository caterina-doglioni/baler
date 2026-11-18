# === VQ-VAE convergence benchmark ===
#
# Controlled comparison against CMS_ae_bench. This config is IDENTICAL to
# CMS_ae_bench_config.py except for the three lines marked below, so that any
# difference in the results is attributable to the bottleneck (continuous
# latent vs vector-quantized latent) and not to the training budget, the
# data split, the optimizer or the architecture.
#
# Differences from CMS_ae_bench:
#   model_name          "AE" -> "VQVAE"
#   vq_num_embeddings   (new) codebook size K = 512
#   vq_commitment_cost  (new) beta = 0.25
#
# K = 512 and beta = 0.25 are the values used throughout van den Oord,
# Vinyals & Kavukcuoglu, "Neural Discrete Representation Learning",
# NeurIPS 2017 (arXiv:1711.00937); the paper reports results are stable for
# beta in 0.1-2.0.
#
# Both options are read by data_processing.model_kwargs_from_config and
# applied at BOTH training time and compress/decompress time. They must not be
# changed between those steps: the codebook is a saved parameter, so a
# different K would fail to load.
#
# Everything else - 500 epochs, test_size 0.2 with the same
# train_test_split(random_state=1), min_delta 1e-7, early stopping patience
# 50, LR scheduler patience 20, compression_ratio 1.6 (latent 15),
# batch_size 512, lr 0.001, deterministic_algorithm True - matches the AE
# benchmark exactly. With deterministic_algorithm True and shuffle=False in
# training.train, both models also see identical batch ordering.


def set_config(c):
    c.input_path = "workspaces/CMS_workspace/data/example_CMS_data.npz"
    c.data_dimension = 1
    c.compression_ratio = 1.6
    c.apply_normalization = True
    c.model_name = "VQVAE"  # <-- differs from CMS_ae_bench
    c.epochs = 500
    c.lr = 0.001
    c.batch_size = 512
    c.early_stopping = True
    c.lr_scheduler = True
    c.save_error_bounded_deltas = False
    c.error_bounded_requirement = 10

    # === VQ-VAE specific (arXiv:1711.00937) ===

    c.vq_num_embeddings = 512  # <-- codebook size K
    c.vq_commitment_cost = 0.25  # <-- beta

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
