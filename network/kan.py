import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,    # in_features: Number of input features.
        out_features,   # out_features: Number of output features.
        grid_size=5,    # grid_size: Grid size, used to control the accuracy of calculations.
        spline_order=3, # spline_order: The order of splines affects the smoothness of the smoothing function.
        scale_noise=0.1,    # scale_noise: Scaling factor of noise.
        scale_base=1.0,     # scale_base: Scaling factor for basic weights.
        scale_spline=1.0,   # scale_spline: Scaling factor for spline weights.
        enable_standalone_scale_spline=True,    # enable_standalone_scale_spline: Whether to enable independent spline scale parameters.
        base_activation=torch.nn.SiLU,  # base_activation: Basic activation function, default is SiLU (Sigmoid Linear Unit).
        grid_eps=0.02,  # grid_eps: The epsilon value of the grid is used for numerical stability.
        grid_range=[-1, 1], # grid_range: The default range for grid values is [-1, 1].
    ):
        super(KANLinear, self).__init__()  # super(KANLinear, self).__init__() calls the base class constructor.
        self.in_features = in_features
        self.out_features = out_features  # Define class attributes in_features and out_features.
        self.grid_size = grid_size  # Calculate grid points and register them as a buffer using register_buffer. Buffers are tensors that do not participate in gradient computation.
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Define base_weight as a trainable parameter.
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # Define spline_weight as a trainable parameter. If standalone spline scaling is enabled, also define spline_scaler.
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )
        # Define other attributes, including noise, base and spline scaling factors, activation function, and whether standalone spline scaling is enabled.
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # self.reset_parameters() calls a method to initialize parameters
        self.reset_parameters()
    # This method is used to initialize the parameters of the KANLinear class.
    # It first uses the Kaiming initialization method (also known as He initialization) to initialize the base_weight,
    # then uses the no-gradient context manager torch.no_grad() to avoid computing gradients.
    def reset_parameters(self):
        # torch.nn.init.kaiming_uniform_: Used to initialize tensors, here for initializing base_weight.
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # noise: Computes a random noise tensor for initializing the spline_weight. The noise is obtained by sampling from a uniform distribution and scaling.
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            # self.spline_weight.data.copy_: Uses the computed noise to update the spline weight tensor.
            # If standalone spline scaling is enabled, use self.scale_spline as the scaling factor, otherwise use 1.0.
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            # If standalone spline scaling is enabled, also use the Kaiming initialization method to initialize spline_scaler.
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
    # This method is used to compute the B-spline basis functions for the given input tensor x.
    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        # First, check if the dimensions and size of the input tensor x meet the expectations.
        assert x.dim() == 2 and x.size(1) == self.in_features

        # grid: Use the class attribute grid, which is a registered buffer containing the grid points.
        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        # x.unsqueeze(-1): Add a dimension to the input tensor x to match the dimension requirements for subsequent operations.
        x = x.unsqueeze(-1)
        # bases: Initialize the B-spline basis function matrix using logical AND operator & to compute whether each point is between two grid points.
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # Loop: Iterate through the spline orders in a loop to update the bases tensor and compute the B-spline basis functions.
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        # assert: Ensure that the size of the computed B-spline basis function matrix meets the expectations.
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        # Return value: Return the contiguous B-spline basis function tensor.
        return bases.contiguous()

    # This code defines the curve2coeff method in the KANLinear class, which is used to compute the coefficients of the curve that interpolates the given data points.
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        # assert statements are used to ensure that the dimensions and sizes of the input tensors x and y meet the expectations.
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # A = self.b_splines(x).transpose(0, 1)
        # Call the b_splines method to compute the B-spline basis functions and transpose the result to match the dimension of the coefficient matrix for the linear system.
        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        # B = y.transpose(0, 1) Transpose the output tensor y to match the right-hand side vector of the linear system.
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        # solution = torch.linalg.lstsq(A, B).solution Use linear least squares (torch.linalg.lstsq)
        # to solve the linear system Ax = B, where x is the required curve coefficients.
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        # result = solution.permute(2, 0, 1) Rearrange the solution tensor to obtain the shape of the final coefficients tensor.
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        # assert result.size() == (...) Ensure that the size of the computed coefficients tensor meets the expectations.
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        # return result.contiguous() Return the contiguous coefficients tensor.
        return result.contiguous()

    # Use the @property decorator to define a property named scaled_spline_weight.
    # This property returns the scaled spline weights based on whether the standalone spline scaling parameter enable_standalone_scale_spline is enabled.
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            # If standalone scaling is enabled, use the self.spline_scaler tensor and expand it to a new dimension for broadcasting.
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            # If not enabled, simply return the spline weights self.spline_weight multiplied by 1.0.
            else 1.0
        )

    # The forward method defines the forward propagation logic of the KANLinear layer.
    def forward(self, x: torch.Tensor):
        # Print the shape of the input tensor
        # print(f"Input shape before assert: {x.shape}")
        # First, check the dimension of the input tensor x.
        assert x.size(-1) == self.in_features
        # Reshape the input x to (-1, in_features) to match the input of the linear layer.
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        # print(f"Input shape after view: {x.shape}")

        # Compute the base output base_output using the activation function self.base_activation and the base weights self.base_weight.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # Compute the spline output spline_output using the B-spline basis functions computed by self.b_splines and the scaled spline weights self.scaled_spline_weight.
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        # Add the base output and spline output to get the final output.
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        # Reshape the output back to the original shape, but with the last dimension as out_features.
        # print(f"Output shape: {output.shape}")
        return output

    # This method is defined in a no-gradient context torch.no_grad() and is used to update the spline grid points.
    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        # Check the dimensions and size of the input tensor x.
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        # Compute the B-spline basis functions splines.
        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        # Perform matrix multiplication between the B-spline basis functions and the original spline weights orig_coeff to get the unreduced spline output unreduced_spline_output.
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        # Sort the input x to collect data distribution information.
        x_sorted = torch.sort(x, dim=0)[0]
        # Use the sorted x and the given margin to compute the adaptive grid points grid_adaptive.
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # Compute the uniform step uniform_step and generate uniformly distributed grid points grid_uniform.
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        # Combine the adaptive grid and uniform grid to form the final grid.
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        # Update the class's grid buffer and spline_weight data.
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        # self.grid.copy_(grid.T) and self.spline_weight.data.copy_(...)
        # These two lines directly modify the class attributes, which is a common practice in PyTorch for updating model parameters or buffers.
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
        # The purpose of this method is to adjust the grid points based on the distribution of the input data to better adapt to the data.
        # By combining adaptive and uniform grid points, it can improve the accuracy and flexibility of spline interpolation.
        # Compute the regularization loss for a single KANLinear layer.
        def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
            """
            Compute the regularization loss.

            This is a dumb simulation of the original L1 regularization as stated in the
            paper, since the original one requires computing absolutes and entropy from the
            expanded (batch, in_features, out_features) intermediate tensor, which is hidden
            behind the F.linear function if we want an memory efficient implementation.

            The L1 regularization is now computed as mean absolute value of the spline
            weights. The authors implementation also includes this term in addition to the
            sample-based regularization.
            """
            # It first simulates the original L1 regularization by computing the mean absolute value of the spline weights.
            l1_fake = self.spline_weight.abs().mean(-1)
            # Then, compute the loss for the activation term based on L1 regularization regularization_loss_activation.
            regularization_loss_activation = l1_fake.sum()
            p = l1_fake / regularization_loss_activation
            # Next, compute the entropy based on the weight distribution regularization_loss_entropy.
            regularization_loss_entropy = -torch.sum(p * p.log())
            # Finally, weight and sum the activation and entropy losses according to the provided weight parameters to get the final regularization loss.
            return (
                    regularize_activation * regularization_loss_activation
                    + regularize_entropy * regularization_loss_entropy
            )

    class KAN(torch.nn.Module):
        # The __init__ method receives a layers_hidden parameter, which is a list containing the number of features in the hidden layers. The length of layers_hidden determines the number of KANLinear layers in the KAN model.
        # In the initialization method, by iterating through the layers_hidden list, create the corresponding number of KANLinear layer instances and add them to the self.layers module list.
        def __init__(
                self,
                layers_hidden,
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1],
        ):
            super(KAN, self).__init__()
            self.grid_size = grid_size
            self.spline_order = spline_order

            self.layers = torch.nn.ModuleList()
            for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
                self.layers.append(
                    KANLinear(
                        in_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
                )

        # The forward method defines the forward propagation logic of the KAN model.
        def forward(self, x: torch.Tensor, update_grid=False):
            # It receives the input tensor x and an optional boolean parameter update_grid.
            for layer in self.layers:
                # If update_grid is True, call the update_grid method in each layer to update the grid points of the spline weights.
                if update_grid:
                    layer.update_grid(x)
                # For each layer, perform forward propagation using layer(x) and pass the result to the next layer.
                x = layer(x)
            return x

        def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
            # It receives two parameters regularize_activation and regularize_entropy to control the weights of the activation term and entropy term in the regularization loss.
            return sum(
                # The regularization loss is accumulated by iterating through all layers and calling the regularization_loss method for each layer.
                layer.regularization_loss(regularize_activation, regularize_entropy)
                for layer in self.layers
            )