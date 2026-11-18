# === VQ-VAE rate-distortion sweep: M = 5 codes/row ===
#
# Generated for the product-quantization sweep. Identical to the other sweep
# configs except vq_num_groups, so the comparison isolates the bit rate.
#
#   bits/row = M * log2(K) = 5 * 9 = 45
#
# K = 512 and beta = 0.25 are the paper's values (arXiv:1711.00937).
# EMA codebook updates (appendix A.1) and dead-entry restarts are enabled to
# counter the codebook collapse measured in the first experiment (16.2%
# utilisation at M=1 without them).
# vq_store_indices stores the discrete codes rather than the quantized vector,
# which is what makes the bit rate above an actual file size.


def set_config(c):
    c.input_path = "workspaces/CMS_workspace/data/example_CMS_data.npz"
    c.data_dimension = 1
    c.compression_ratio = 1.6
    c.apply_normalization = True
    c.model_name = "VQVAE"
    c.epochs = 150
    c.lr = 0.001
    c.batch_size = 512
    c.early_stopping = True
    c.lr_scheduler = True
    c.save_error_bounded_deltas = False
    c.error_bounded_requirement = 10

    # === VQ-VAE specific (arXiv:1711.00937) ===

    c.vq_num_embeddings = 512
    c.vq_commitment_cost = 0.25
    c.vq_num_groups = 5
    c.vq_use_ema = True
    c.vq_restart_dead = True
    c.vq_store_indices = True

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
