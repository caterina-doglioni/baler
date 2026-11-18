# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import functional as F


import torch.utils.data
from torch.nn import functional as F
from torch.autograd import Function
from ..modules import helper


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size(), device=inputs.device) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    """

    def __init__(
        self,
        ch,
        inverse=False,
        beta_min=1e-6,
        gamma_init=0.1,
        reparam_offset=2**-18,
    ):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.device = helper.get_device()
        self.reparam_offset = torch.tensor([reparam_offset], device=self.device)

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2) ** 0.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=self.device) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=self.device)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class AE(nn.Module):
    # This class is a modified version of the original class by George Dialektakis found at
    # https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021
    # Released under the Apache License 2.0 found at https://www.apache.org/licenses/LICENSE-2.0.txt
    # Copyright 2021 George Dialektakis

    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(AE, self).__init__(*args, **kwargs)

        self.activations = {}

        # encoder
        self.en1 = nn.Linear(n_features, 200, dtype=torch.float64)
        self.en2 = nn.Linear(200, 100, dtype=torch.float64)
        self.en3 = nn.Linear(100, 50, dtype=torch.float64)
        self.en4 = nn.Linear(50, z_dim, dtype=torch.float64)
        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64)
        self.de2 = nn.Linear(50, 100, dtype=torch.float64)
        self.de3 = nn.Linear(100, 200, dtype=torch.float64)
        self.de4 = nn.Linear(200, n_features, dtype=torch.float64)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    # Implementation of activation extraction using the forward_hook method

    def get_hook(self, layer_name):
        def hook(model, input, output):
            self.activations[layer_name] = output.detach()

        return hook

    def get_layers(self) -> list:
        return [self.en1, self.en2, self.en3, self.de1, self.de2, self.de3]

    def store_hooks(self) -> list:
        layers = self.get_layers()
        hooks = []
        for i in range(len(layers)):
            hooks.append(layers[i].register_forward_hook(self.get_hook(str(i))))
        return hooks

    def get_activations(self) -> dict:
        for kk in self.activations:
            self.activations[kk] = F.leaky_relu(self.activations[kk])
        return self.activations

    def detach_hooks(self, hooks: list) -> None:
        for hook in hooks:
            hook.remove()


class CFD_dense_AE(nn.Module):
    # This class is a modified version of the original class by George Dialektakis found at
    # https://github.com/Autoencoders-compression-anomaly/Deep-Autoencoders-Data-Compression-GSoC-2021
    # Released under the Apache License 2.0 found at https://www.apache.org/licenses/LICENSE-2.0.txt
    # Copyright 2021 George Dialektakis

    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(CFD_dense_AE, self).__init__(*args, **kwargs)

        self.activations = {}

        # encoder
        self.en1 = nn.Linear(n_features, 200, dtype=torch.float)
        self.en2 = nn.Linear(200, 100, dtype=torch.float)
        self.en3 = nn.Linear(100, 50, dtype=torch.float)
        self.en4 = nn.Linear(50, z_dim, dtype=torch.float)
        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float)
        self.de2 = nn.Linear(50, 100, dtype=torch.float)
        self.de3 = nn.Linear(100, 200, dtype=torch.float)
        self.de4 = nn.Linear(200, n_features, dtype=torch.float)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    # Implementation of activation extraction using the forward_hook method

    def get_hook(self, layer_name):
        def hook(model, input, output):
            self.activations[layer_name] = output.detach()

        return hook

    def get_layers(self) -> list:
        return [self.en1, self.en2, self.en3, self.de1, self.de2, self.de3]

    def store_hooks(self) -> list:
        layers = self.get_layers()
        hooks = []
        for i in range(len(layers)):
            hooks.append(layers[i].register_forward_hook(self.get_hook(str(i))))
        return hooks

    def get_activations(self) -> dict:
        for kk in self.activations:
            self.activations[kk] = F.leaky_relu(self.activations[kk])
        return self.activations

    def detach_hooks(self, hooks: list) -> None:
        for hook in hooks:
            hook.remove()


class AE_Dropout_BN(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(AE_Dropout_BN, self).__init__(*args, **kwargs)

        # encoder
        self.enc_nn = nn.Sequential(
            nn.Linear(n_features, 200, dtype=torch.float64),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(200,dtype=torch.float64),
            nn.Linear(200, 100, dtype=torch.float64),
            nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(100,dtype=torch.float64),
            nn.Linear(100, 50, dtype=torch.float64),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(50,dtype=torch.float64),
            nn.Linear(50, z_dim, dtype=torch.float64),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(z_dim,dtype=torch.float64)
        )

        # decoder
        self.dec_nn = nn.Sequential(
            nn.Linear(z_dim, 50, dtype=torch.float64),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, dtype=torch.float64),
            nn.Linear(50, 100, dtype=torch.float64),
            # nn.Dropout(p=0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, dtype=torch.float64),
            nn.Linear(100, 200, dtype=torch.float64),
            # nn.Dropout(p=0.4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, dtype=torch.float64),
            nn.Linear(200, n_features, dtype=torch.float64),
            # nn.Dropout(p=0.5),
            nn.BatchNorm1d(n_features, dtype=torch.float64),
            nn.ReLU(),
        )

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        out = self.enc_nn(x)
        return out

    def decode(self, z):
        out = self.dec_nn(z)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class Conv_AE(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(Conv_AE, self).__init__(*args, **kwargs)

        self.q_z_mid_dim = 2000
        self.q_z_output_dim = 128
        self.conv_op_shape = None

        # Encoder

        # Conv Layers
        self.q_z_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 5), stride=(1), padding=(1)),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3), stride=(1), padding=(1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.q_z_lin = nn.Sequential(
            nn.Linear(self.q_z_output_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_output_dim),
            nn.Linear(self.q_z_mid_dim, z_dim),
            nn.ReLU(),
        )

        # Decoder

        # Linear layers
        self.p_x_lin = nn.Sequential(
            nn.Linear(z_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_output_dim),
            nn.Linear(self.q_z_mid_dim, self.q_z_output_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(42720)
        )
        # Conv Layers
        self.p_x_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(3), stride=(1), padding=(1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=(2, 5), stride=(1), padding=(1)),
        )

    def encode(self, x):
        # Conv
        out = self.q_z_conv(x)
        self.conv_op_shape = out.shape
        self.q_z_output_dim = out.shape[1] * out.shape[2] * out.shape[3]

        # Flatten
        out = self.flatten(out)
        # Dense
        out = self.q_z_lin(out)
        return out

    def decode(self, z):
        # Dense
        out = self.p_x_lin(z)
        # Unflatten
        out = out.view(
            self.conv_op_shape[0],
            self.conv_op_shape[1],
            self.conv_op_shape[2],
            self.conv_op_shape[3],
        )
        # Conv transpose
        out = self.p_x_conv(out)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def get_final_layer_dims(self):
        return self.conv_op_shape

    def set_final_layer_dims(self, conv_op_shape):
        self.conv_op_shape = conv_op_shape


class FPGA_prototype_model(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(FPGA_prototype_model, self).__init__(*args, **kwargs)

        # encoder
        self.en1 = nn.Linear(n_features, 20, dtype=torch.float64)
        self.en_act1 = nn.ReLU()
        self.en2 = nn.Linear(20, 10, dtype=torch.float64)
        self.en_act2 = nn.ReLU()
        self.en3 = nn.Linear(10, z_dim, dtype=torch.float64)

        # decoder
        self.de1 = nn.Linear(z_dim, 10, dtype=torch.float64)
        self.de_act1 = nn.ReLU()
        self.de2 = nn.Linear(10, 20, dtype=torch.float64)
        self.de_act2 = nn.ReLU()
        self.de3 = nn.Linear(20, n_features, dtype=torch.float64)

        self.n_features = n_features
        self.z_dim = z_dim

    def encode(self, x):
        s1 = self.en1(x)
        s2 = self.en_act1(s1)
        s3 = self.en2(s2)
        s4 = self.en_act2(s3)
        s5 = self.en3(s4)
        return s5

    def decode(self, z):
        s6 = self.de1(z)
        s7 = self.de_act1(s6)
        s8 = self.de2(s7)
        s9 = self.de_act2(s8)
        s10 = self.de3(s9)
        return s10

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def get_layers(self) -> list:
        return [
            self.en1,
            self.en_act1,
            self.en2,
            self.en_act2,
            self.en3,
            self.de1,
            self.de_act1,
            self.de2,
            self.de_act2,
            self.de3,
        ]


class Conv_AE_3D(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(Conv_AE_3D, self).__init__(*args, **kwargs)

        self.q_z_mid_dim = 2000
        self.q_z_output_dim = 612864
        self.compress_to_latent_space = True
        self.debug = False

        # Encoder

        # Conv Layers
        self.q_z_conv = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(1, 3, 5), stride=(1), padding=(0)),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv3d(4, 8, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.q_z_lin = nn.Sequential(
            nn.Linear(self.q_z_output_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_mid_dim),
            nn.Linear(self.q_z_mid_dim, z_dim),
            nn.ReLU(),
        )

        # Decoder

        # Linear layers
        self.p_x_lin = nn.Sequential(
            nn.Linear(z_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_mid_dim),
            nn.Linear(self.q_z_mid_dim, self.q_z_output_dim),
            nn.ReLU(),
        )

        # Conv Layers
        self.p_x_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 3, 3), stride=(1), padding=(0)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 8, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 4, kernel_size=(3), stride=(1), padding=(0)),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.ConvTranspose3d(4, 1, kernel_size=(1, 3, 5), stride=(1), padding=(0)),
        )

    def encode(self, x):
        # Conv

        out = self.q_z_conv(x)
        # Flatten
        out = self.flatten(out)

        if self.compress_to_latent_space:
            # Dense
            out = self.q_z_lin(out)

        return out

    def decode(self, out):
        if self.compress_to_latent_space:
            out = self.p_x_lin(out)

        out = out.view(4, 64, 7, 38, 36)

        # Conv transpose
        out = self.p_x_conv(out)

        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def set_compress_to_latent_space(self, compress_to_latent_space):
        self.compress_to_latent_space = compress_to_latent_space


class Conv_AE_GDN(nn.Module):
    def __init__(self, n_features, z_dim, *args, **kwargs):
        super(Conv_AE_GDN, self).__init__(*args, **kwargs)

        self.q_z_mid_dim = 2000
        self.q_z_output_dim = 128
        self.conv_op_shape = None

        # Encoder

        # Conv Layers
        self.q_z_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2, 5), stride=(1), padding=(1)),
            # nn.BatchNorm2d(8),
            GDN(8),
            nn.Conv2d(8, 16, kernel_size=(3), stride=(1), padding=(1)),
            nn.BatchNorm2d(16),
            GDN(16),
            nn.Conv2d(16, 32, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(32),
            GDN(32),
        )
        # Flatten
        self.flatten = nn.Flatten(start_dim=1)

        # Linear layers
        self.q_z_lin = nn.Sequential(
            nn.Linear(self.q_z_output_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_output_dim),
            nn.Linear(self.q_z_mid_dim, z_dim),
            nn.ReLU(),
        )

        # Decoder

        # Linear layers
        self.p_x_lin = nn.Sequential(
            nn.Linear(z_dim, self.q_z_mid_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(self.q_z_output_dim),
            nn.Linear(self.q_z_mid_dim, self.q_z_output_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(42720)
        )
        # Conv Layers
        self.p_x_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=(3), stride=(1), padding=(0)),
            # nn.BatchNorm2d(16),
            GDN(16, inverse=True),
            nn.ConvTranspose2d(16, 8, kernel_size=(3), stride=(1), padding=(1)),
            # nn.BatchNorm2d(8),
            GDN(8, inverse=True),
            nn.ConvTranspose2d(8, 1, kernel_size=(2, 5), stride=(1), padding=(1)),
        )

    def encode(self, x):
        # Conv
        out = self.q_z_conv(x)
        self.conv_op_shape = out.shape
        self.q_z_output_dim = out.shape[1] * out.shape[2] * out.shape[3]

        # Flatten
        out = self.flatten(out)
        # Dense
        out = self.q_z_lin(out)
        return out

    def decode(self, z):
        # Dense
        out = self.p_x_lin(z)
        # Unflatten
        out = out.view(
            self.conv_op_shape[0],
            self.conv_op_shape[1],
            self.conv_op_shape[2],
            self.conv_op_shape[3],
        )
        # Conv transpose
        out = self.p_x_conv(out)
        return out

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def get_final_layer_dims(self):
        return self.conv_op_shape

    def set_final_layer_dims(self, conv_op_shape):
        self.conv_op_shape = conv_op_shape


class PJ_Conv_AE(nn.Module):
    def __init__(self, n_features, z_dim=10, *args, **kwargs):
        super(PJ_Conv_AE, self).__init__(*args, **kwargs)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 20, kernel_size=5, stride=2, padding=2
            ),  # Adjust input channels to 1 for grayscale images
            nn.LeakyReLU(0.2),
            nn.Conv2d(20, 50, kernel_size=5, stride=2, padding=2),
            nn.Flatten(),
            nn.Linear(50 * 7 * 7, 500),  # Adjust input size based on your data
            nn.Linear(500, z_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 500),
            nn.LeakyReLU(0.2),
            nn.Linear(500, 2450),
            nn.Unflatten(1, (50, 7, 7)),
            nn.ConvTranspose2d(
                50, 20, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ConvTranspose2d(
                20, 1, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.LeakyReLU(0.2),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out

    def get_final_layer_dims(self):
        return list(self.decoder.children())[-1]

    def set_final_layer_dims(self, conv_op_shape):
        self.conv_op_shape = conv_op_shape


class TransformerAE(nn.Module):
    """Autoencoder mixed with the Transformer Encoder layer

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        in_dim,
        h_dim=256,
        n_heads=1,
        latent_size=50,
        activation=torch.nn.functional.gelu,
    ):
        super(TransformerAE, self).__init__()

        self.transformer_encoder_layer_1 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=in_dim,
            activation=activation,
            dim_feedforward=h_dim,
            nhead=n_heads,
        )

        self.transformer_encoder_layer_2 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=256,
            activation=activation,
            dim_feedforward=256,
            nhead=n_heads,
        )
        self.transformer_encoder_layer_3 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            norm_first=True,
            d_model=128,
            activation=activation,
            dim_feedforward=128,
            nhead=n_heads,
        )

        self.encoder_layer_1 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(in_dim, 256),
            torch.nn.LeakyReLU(),
        )

        self.encoder_layer_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),
        )

        self.encoder_layer_3 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(128, latent_size),
            torch.nn.LeakyReLU(),
        )

        self.decoder_layer_3 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(latent_size, 128),
            torch.nn.LeakyReLU(),
        )
        self.decoder_layer_2 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(), torch.nn.Linear(128, 256), torch.nn.LeakyReLU()
        )
        self.decoder_layer_1 = torch.nn.Sequential(
            torch.nn.LazyBatchNorm1d(),
            torch.nn.Linear(256, in_dim),
            torch.nn.LeakyReLU(),
        )

        self.transformer_decoder_layer_3 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=128,
            activation=activation,
            dim_feedforward=128,
            nhead=n_heads,
        )

        self.transformer_decoder_layer_2 = torch.nn.TransformerEncoderLayer(
            batch_first=True,
            d_model=256,
            activation=activation,
            dim_feedforward=256,
            nhead=n_heads,
        )

        self.transformer_decoder_layer_1 = torch.nn.TransformerEncoderLayer(
            d_model=in_dim,
            dim_feedforward=h_dim,
            activation=activation,
            nhead=n_heads,
        )

    def encoder(self, x: torch.Tensor):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        z = self.transformer_encoder_layer_1(x)
        z = self.encoder_layer_1(z)
        z = self.transformer_encoder_layer_2(z)
        z = self.encoder_layer_2(z)
        z = self.transformer_encoder_layer_3(z)
        z = self.encoder_layer_3(z)

        return z

    def decoder(self, z: torch.Tensor):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.decoder_layer_3(z)
        x = self.transformer_decoder_layer_3(x)
        x = self.decoder_layer_2(x)
        x = self.transformer_decoder_layer_2(x)
        x = self.decoder_layer_1(x)
        x = self.transformer_decoder_layer_1(x)
        return x

    def forward(self, x: torch.Tensor):
        """_summary_

        Args:
            z (_type_): _description_

        Returns:
            _type_: _description_
        """
        z = self.encoder(x)
        x = self.decoder(z)
        return x


class VectorQuantizer(nn.Module):
    """Vector-quantization bottleneck (van den Oord, Vinyals & Kavukcuoglu,
    "Neural Discrete Representation Learning", NeurIPS 2017, arXiv:1711.00937).

    Maps each continuous encoder output z_e(x) to its nearest entry in a
    learned codebook of `num_embeddings` vectors, each of `embedding_dim`
    dimensions. Equation 3 of the paper gives the two bottleneck loss terms
    returned here (the reconstruction term is supplied by the caller):

        ||sg[z_e(x)] - e||^2  +  beta * ||z_e(x) - sg[e]||^2
        \\______________/         \\___________________/
          codebook loss              commitment loss

    where sg[.] is the stop-gradient operator. The codebook loss moves the
    selected codebook entries towards the encoder output; the commitment loss
    keeps the encoder from growing its output arbitrarily to escape the
    codebook. The paper reports results are stable for beta in 0.1-2.0 and
    uses beta = 0.25 throughout, which is the default here.

    argmin is not differentiable, so gradients are passed from the decoder
    input back to the encoder output unchanged via the straight-through
    estimator (section 3.2 of the paper): z_q = z_e + (z_q - z_e).detach()
    is numerically equal to z_q in the forward pass, while d(z_q)/d(z_e) = 1
    in the backward pass.

    **Product quantization (`num_groups`).** With num_groups = M the latent is
    split into M contiguous sub-vectors of embedding_dim/M dims, each quantized
    against its own codebook, giving M indices per row. This is the lever that
    sets the bit rate:

        bits/row = M * log2(K)

    K is only a logarithmic lever and the codebook grows linearly in K, so M is
    how the compression ratio is actually tuned. M = 1 reproduces the original
    whole-vector behaviour, which on a 1-D tabular latent is an extremely tight
    bottleneck (one symbol for the entire row).

    **EMA codebook updates (`use_ema`).** Appendix A.1 of the paper replaces the
    codebook loss term with an exponential moving average of the encoder outputs
    assigned to each entry, which is generally more robust against codebook
    collapse. When enabled the codebook receives no gradient and the returned
    loss is the commitment term only.

    **Dead-entry restarts (`restart_dead`).** Entries unused for
    `restart_after` consecutive training calls are re-seeded from encoder
    outputs in the current batch. This complements EMA: an entry that is never
    selected is never updated by either rule, so it would otherwise stay dead
    permanently.

    Args:
        num_embeddings (int): codebook size K.
        embedding_dim (int): full latent width D. Must be divisible by
            num_groups.
        commitment_cost (float): beta in the equation above.
        num_groups (int): M, number of codes stored per row.
        use_ema (bool): use EMA codebook updates instead of the codebook loss.
        ema_decay (float): EMA decay (the paper uses 0.99).
        ema_epsilon (float): Laplace smoothing for the EMA cluster counts.
        restart_dead (bool): enable dead-entry restarts.
        restart_after (int): calls an entry may go unused before restarting.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        num_groups=1,
        use_ema=False,
        ema_decay=0.99,
        ema_epsilon=1e-5,
        restart_dead=False,
        restart_after=50,
    ):
        super(VectorQuantizer, self).__init__()
        if embedding_dim % num_groups != 0:
            divisors = [
                d for d in range(1, embedding_dim + 1) if embedding_dim % d == 0
            ]
            raise ValueError(
                f"num_groups={num_groups} does not divide embedding_dim="
                f"{embedding_dim}. Valid num_groups for this latent width: "
                f"{divisors}. The latent width follows from the config's "
                f"compression_ratio, so changing that changes the options."
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.num_groups = num_groups
        self.sub_dim = embedding_dim // num_groups
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.restart_dead = restart_dead
        self.restart_after = restart_after

        # One codebook per group: (M, K, sub_dim). float64 to match baler's
        # data_dimension == 1 path, which feeds the model float64 tensors.
        codebook = torch.empty(
            num_groups, num_embeddings, self.sub_dim, dtype=torch.float64
        )
        # Uniform over 1/K, as in the reference implementation.
        codebook.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.codebook = nn.Parameter(codebook)

        if use_ema:
            # EMA updates the codebook in place; no gradient should reach it.
            self.codebook.requires_grad_(False)
            # cluster_size starts at ONE, not zero, so that the implied
            # codebook (ema_w / cluster_size) equals the initial codebook. With
            # a zero start, an entry that is never selected divides a decaying
            # ema_w by a near-zero count and diverges instead of standing
            # still -- observed as an unused entry running off to ~2e4.
            self.register_buffer(
                "ema_cluster_size",
                torch.ones(num_groups, num_embeddings, dtype=torch.float64),
            )
            self.register_buffer("ema_w", self.codebook.data.clone())
        if restart_dead:
            self.register_buffer(
                "steps_unused",
                torch.zeros(num_groups, num_embeddings, dtype=torch.long),
            )

    def _split(self, z_e):
        """(N, D) -> (M, N, sub_dim)."""
        n = z_e.shape[0]
        return z_e.view(n, self.num_groups, self.sub_dim).permute(1, 0, 2)

    def _merge(self, z_g):
        """(M, N, sub_dim) -> (N, D)."""
        n = z_g.shape[1]
        return z_g.permute(1, 0, 2).reshape(n, self.embedding_dim)

    def lookup(self, indices):
        """Codebook vectors for stored indices, without running the encoder.

        This is what an index-based decompression path needs: the stored
        integers are all that is required to rebuild the quantized latent.

        Args:
            indices (torch.Tensor): shape (N, M), dtype long.

        Returns:
            torch.Tensor: (N, D) reassembled quantized latent.
        """
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        idx_g = indices.t()
        g = torch.arange(self.num_groups, device=idx_g.device).unsqueeze(1)
        return self._merge(self.codebook[g, idx_g])

    def forward(self, z_e):
        """Quantize a batch of encoder outputs.

        Args:
            z_e (torch.Tensor): continuous encoder output, shape (N, D).

        Returns:
            tuple: (z_q, loss, indices) with z_q of shape (N, D) carrying the
                straight-through estimator, loss a scalar, and indices of shape
                (N, M), dtype long.
        """
        z_g = self._split(z_e)
        # Squared euclidean distance ||z - e||^2 expanded as
        # ||z||^2 - 2 z.e^T + ||e||^2, batched over groups, so no
        # (M, N, K, sub) difference tensor is ever materialised.
        d = (
            z_g.pow(2).sum(dim=-1, keepdim=True)
            - 2 * torch.bmm(z_g, self.codebook.transpose(1, 2))
            + self.codebook.pow(2).sum(dim=-1).unsqueeze(1)
        )
        idx_g = d.argmin(dim=-1)  # (M, N)
        g = torch.arange(self.num_groups, device=z_e.device).unsqueeze(1)
        q_g = self.codebook[g, idx_g]  # (M, N, sub)
        z_q = self._merge(q_g)

        if self.use_ema:
            if self.training:
                self._ema_update(z_g, idx_g)
            # Appendix A.1: the codebook is maintained by EMA, so the codebook
            # loss term disappears and only the commitment term remains.
            loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())
        else:
            # Equation 3. mse_loss averages, which rescales both terms
            # identically, so beta keeps its meaning.
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commitment_loss = F.mse_loss(z_e, z_q.detach())
            loss = codebook_loss + self.commitment_cost * commitment_loss

        if self.restart_dead and self.training:
            self._restart_dead_entries(z_g, idx_g)

        # Straight-through estimator: identical to z_q numerically, but routes
        # the decoder's gradient straight back to the encoder.
        z_q = z_e + (z_q - z_e).detach()
        return z_q, loss, idx_g.t().contiguous()

    @torch.no_grad()
    def _ema_update(self, z_g, idx_g):
        """EMA codebook update, appendix A.1 of arXiv:1711.00937."""
        k = self.num_embeddings
        onehot = F.one_hot(idx_g, k).to(self.codebook.dtype)  # (M, N, K)
        counts = onehot.sum(dim=1)  # (M, K)
        dw = torch.bmm(onehot.transpose(1, 2), z_g)  # (M, K, sub)

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            counts, alpha=1 - self.ema_decay
        )
        self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Laplace smoothing so a zero-count entry does not divide by zero,
        # while preserving the total mass per group.
        n = self.ema_cluster_size.sum(dim=1, keepdim=True)
        smoothed = (
            (self.ema_cluster_size + self.ema_epsilon) / (n + k * self.ema_epsilon) * n
        )
        self.codebook.data.copy_(self.ema_w / smoothed.unsqueeze(-1))

    @torch.no_grad()
    def _restart_dead_entries(self, z_g, idx_g):
        """Re-seed entries unused for `restart_after` calls from this batch.

        An entry that is never selected is never updated by either the codebook
        loss or the EMA, so without this it stays dead permanently.
        """
        used = torch.zeros_like(self.steps_unused, dtype=torch.bool)
        used.scatter_(1, idx_g, True)
        self.steps_unused.add_(1)
        self.steps_unused[used] = 0

        dead = self.steps_unused > self.restart_after
        if not bool(dead.any()):
            return
        n = z_g.shape[1]
        for m in range(self.num_groups):
            dm = dead[m]
            n_dead = int(dm.sum())
            if n_dead == 0:
                continue
            pick = torch.randint(0, n, (n_dead,), device=z_g.device)
            self.codebook.data[m, dm] = z_g[m, pick]
            if self.use_ema:
                self.ema_w[m, dm] = z_g[m, pick]
                self.ema_cluster_size[m, dm] = 1.0
            self.steps_unused[m, dm] = 0

    def perplexity(self, indices):
        """Codebook perplexity exp(-sum_k p_k log p_k), averaged over groups.

        Equals the number of distinct entries under uniform use and collapses
        towards 1 when the model routes everything to a few entries (the
        characteristic VQ-VAE failure mode). Bounded by 1 <= perplexity <= K.

        Args:
            indices (torch.Tensor): (N,) or (N, M) selected entries.

        Returns:
            torch.Tensor: scalar perplexity.
        """
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        onehot = F.one_hot(indices, self.num_embeddings).to(self.codebook.dtype)
        probs = onehot.mean(dim=0)  # (M, K)
        # Guard log(0) for unused entries; they contribute 0 to the entropy.
        per_group = torch.exp(-(probs * torch.log(probs + 1e-12)).sum(dim=-1))
        return per_group.mean()

    def utilisation(self, indices):
        """Fraction of entries used at least once, averaged over groups."""
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        onehot = F.one_hot(indices, self.num_embeddings)
        used = (onehot.sum(dim=0) > 0).to(self.codebook.dtype)  # (M, K)
        return used.mean()


class VQVAE(nn.Module):
    """Vector-Quantized autoencoder backend (arXiv:1711.00937).

    Same dense encoder/decoder topology as `AE` so that the comparison between
    the two backends isolates the bottleneck: the only difference is that the
    latent is snapped to a learned codebook before decoding.

        n_features -> 200 -> 100 -> 50 -> z_dim  -[quantize]->  z_dim -> 50 -> 100 -> 200 -> n_features

    `encode` returns the *quantized* latent, so the existing
    `helper.compress` / `helper.decompress` plumbing works unchanged. Note that
    this stores the continuous codebook vector (z_dim float64 values per row),
    not the integer codebook index -- see `encode_indices` for the discrete
    representation, which is what an entropy-coding compression backend would
    actually store.

    The quantization loss from the last forward pass is exposed as `self.vq_loss`
    for `training.fit` / `training.validate` to add to the reconstruction loss.

    Args:
        n_features (int): input/output width.
        z_dim (int): latent width, set by baler from the compression ratio.
        num_embeddings (int): codebook size K. Config: `vq_num_embeddings`.
        commitment_cost (float): beta. Config: `vq_commitment_cost`.
        num_groups (int): M, codes stored per row (product quantization).
            Must divide z_dim. Config: `vq_num_groups`.
        use_ema (bool): EMA codebook updates. Config: `vq_use_ema`.
        restart_dead (bool): dead-entry restarts. Config: `vq_restart_dead`.
    """

    def __init__(
        self,
        n_features,
        z_dim,
        num_embeddings=512,
        commitment_cost=0.25,
        num_groups=1,
        use_ema=False,
        restart_dead=False,
        *args,
        **kwargs,
    ):
        super(VQVAE, self).__init__(*args, **kwargs)

        self.activations = {}

        # encoder
        self.en1 = nn.Linear(n_features, 200, dtype=torch.float64)
        self.en2 = nn.Linear(200, 100, dtype=torch.float64)
        self.en3 = nn.Linear(100, 50, dtype=torch.float64)
        self.en4 = nn.Linear(50, z_dim, dtype=torch.float64)

        # Quantizer is registered BEFORE the decoder on purpose. helper.decompress
        # infers the reconstructed width from the LAST entry of the state dict:
        #     number_of_columns = len(model_dict[list(model_dict.keys())[-1]])
        # so the final registered parameter must stay the output-layer bias
        # (length n_features). Registering the codebook after the decoder would
        # make that read the embedding weight instead (length num_embeddings)
        # and silently build a wrongly-shaped model at decompression time.
        # tests/test_models_vqvae.py pins this ordering.
        self.quantizer = VectorQuantizer(
            num_embeddings,
            z_dim,
            commitment_cost,
            num_groups=num_groups,
            use_ema=use_ema,
            restart_dead=restart_dead,
        )

        # decoder
        self.de1 = nn.Linear(z_dim, 50, dtype=torch.float64)
        self.de2 = nn.Linear(50, 100, dtype=torch.float64)
        self.de3 = nn.Linear(100, 200, dtype=torch.float64)
        self.de4 = nn.Linear(200, n_features, dtype=torch.float64)

        self.n_features = n_features
        self.z_dim = z_dim

        # Populated on every forward/encode; read by training.fit and
        # training.validate. A zero default keeps the attribute present (and
        # the hasattr check meaningful) before the first forward pass.
        self.vq_loss = torch.zeros((), dtype=torch.float64)
        self.last_indices = None

    def encode_continuous(self, x):
        """Encoder output before quantization, shape (N, z_dim)."""
        h1 = F.leaky_relu(self.en1(x))
        h2 = F.leaky_relu(self.en2(h1))
        h3 = F.leaky_relu(self.en3(h2))
        return self.en4(h3)

    def encode(self, x):
        """Quantized latent, shape (N, z_dim).

        Matches `AE.encode`'s contract so helper.compress needs no changes.
        Side effect: refreshes `self.vq_loss` and `self.last_indices`.
        """
        z_e = self.encode_continuous(x)
        z_q, loss, indices = self.quantizer(z_e)
        self.vq_loss = loss
        self.last_indices = indices
        return z_q

    def encode_indices(self, x):
        """Discrete codebook indices for a batch, shape (N, M), dtype long.

        This is the representation that makes VQ-VAE interesting for
        compression: M * log2(K) bits per row instead of z_dim float64 values.
        Used by `helper.compress` when the config sets `vq_store_indices`.
        """
        z_e = self.encode_continuous(x)
        _, _, indices = self.quantizer(z_e)
        return indices

    def decode_indices(self, indices):
        """Reconstruct from stored codebook indices alone.

        The counterpart to `encode_indices`: an index-based decompression path
        never sees the encoder, only the saved integers and the codebook.

        Args:
            indices (torch.Tensor): (N, M) long.

        Returns:
            torch.Tensor: (N, n_features) reconstruction.
        """
        return self.decode(self.quantizer.lookup(indices))

    def decode(self, z):
        h4 = F.leaky_relu(self.de1(z))
        h5 = F.leaky_relu(self.de2(h4))
        h6 = F.leaky_relu(self.de3(h5))
        out = self.de4(h6)
        return out

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def codebook_perplexity(self, x):
        """Codebook perplexity for a batch of inputs. See VectorQuantizer."""
        return self.quantizer.perplexity(self.encode_indices(x))

    # Activation extraction, matching AE's interface.

    def get_hook(self, layer_name):
        def hook(model, input, output):
            self.activations[layer_name] = output.detach()

        return hook

    def get_layers(self) -> list:
        return [self.en1, self.en2, self.en3, self.de1, self.de2, self.de3]

    def store_hooks(self) -> list:
        layers = self.get_layers()
        hooks = []
        for i in range(len(layers)):
            hooks.append(layers[i].register_forward_hook(self.get_hook(str(i))))
        return hooks

    def get_activations(self) -> dict:
        for kk in self.activations:
            self.activations[kk] = F.leaky_relu(self.activations[kk])
        return self.activations

    def detach_hooks(self, hooks: list) -> None:
        for hook in hooks:
            hook.remove()
