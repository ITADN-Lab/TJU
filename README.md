# TJU Optimizers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

This repository provides PyTorch implementations of advanced optimizers from the TJU family, inspired by recent advancements in adaptive optimization techniques. These optimizers integrate approximate Hessian information with gradient-based updates for improved training stability and efficiency in deep learning models.

- **TJU_v1**: An implementation of the TJU algorithm, which combines approximate Hessian with EMA of gradients, supporting various weight decay strategies and warmup schedules.
- **TJU_v3**: An implementation of the AdTJU algorithm, an improved variant that integrates Adam-like updates with scaled approximate Hessian addition to prevent small steps, with optional cosine annealing. 
- **TJU_v4**: An implementation of the NdaTJU algorithm, which fixes decoupled weight decay (AdamW-style) while maintaining the precision of TJU_v3. It ensures correct weight decay scaling with learning rate and restores update stability.

These optimizers are designed for flexibility in training neural networks, particularly in scenarios requiring robust convergence and handling of complex loss landscapes.

## Key Features
- **Approximate Hessian Integration**: Enhances second-moment estimates for better adaptation to curvature.
- **Weight Decay Options**: Supports 'L2', 'decoupled' (AdamW-style), and 'stable' modes.
- **Learning Rate Scheduling**: Built-in warmup and optional cosine annealing for smooth training progression.
- **Numerical Stability**: Includes bias correction, clamping, and epsilon for safe denominator handling.
- **Based on Research**: TJU_v3 draws from the AdTJU algorithm described in the CVPR paper (see [References](#references)). TJU_v4 extends to NDatju with improvements in weight decay handling.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/tju-optimizers.git
cd tju-optimizers
pip install -r requirements.txt
```

**requirements.txt** (minimal):
```
torch>=1.10.0
numpy
```

You can also copy the optimizer classes directly into your PyTorch project.

## Usage

Import and use the optimizers like any PyTorch optimizer. Below are examples for each variant.

### TJU_v1 (TJU)
```python
import torch
from tju_optimizers import TJU_v1  # Assume you have a file tju_optimizers.py with the classes

model = YourModel()
optimizer = TJU_v1(model.parameters(), lr=1e-3, beta=0.9, eps=1e-4, rebound='constant', warmup=500, weight_decay=1e-5, weight_decay_type='decoupled')

for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### TJU_v3 (AdTJU)
```python
from tju_optimizers import TJU_v3

optimizer = TJU_v3(model.parameters(), lr=1e-3, betas=(0.9, 0.999), beta_h=0.85, eps=1e-8, rebound='constant', warmup=100, weight_decay=0.0, weight_decay_type='L2', hessian_scale=0.05, total_steps=10000, use_cosine_scheduler=True)
```

### TJU_v4 (NdaTJU)
```python
from tju_optimizers import TJU_v4

optimizer = TJU_v4(model.parameters(), lr=1e-3, betas=(0.9, 0.999), beta_h=0.85, eps=1e-8, rebound='constant', warmup=100, weight_decay=0.05, weight_decay_type='AdamW', hessian_scale=0.05, total_steps=10000, use_cosine_scheduler=True)
```

For full parameter details, refer to the docstrings in the code.

### RUN
```python
 python .\cifar10\cifar10_train.py
```

## Performance Notes
- **When to Use**: TJU_v1 (TJU) for scenarios needing robust gradient EMA and Hessian guidance. TJU_v3 (AdTJU) for improved Adam-like updates with Hessian scaling. TJU_v4 (NDatju) for tasks requiring precise decoupled weight decay, such as fine-tuning large models.
- **Hyperparameters**: Tune `hessian_scale` (e.g., 0.05-0.1) to balance Hessian influence. Use 'belief' rebound for adaptive bounding in noisy gradients.
- **Compatibility**: Tested with PyTorch 1.10+. Works with standard training loops, including distributed training.

## References
This implementation is based on the AdTJU algorithm from the CVPR paper:
- [AdTJU: Adaptive Optimization with Approximate Hessian](link-to-paper-if-available) (PDF attachment: CVPR_AdTJU_F1 (1).pdf)
  - TJU_v3 directly implements AdTJU.
  - TJU_v4 extends to NDatju with improvements in weight decay handling.

If you use this code in your research, please cite the original paper:
```
@article{AdTJU_CVPR,
  title={AdTJU: Adaptive Optimization with Approximate Hessian},
  author={Authors from PDF},
  journal={CVPR},
  year={Year from PDF},
  url={Link if available}
}
```

For NDatju specifics in TJU_v4, refer to the improvements noted in the code comments.

## Contributing
Contributions are welcome! Please open an issue or pull request for bug fixes, new features, or improvements. Ensure code follows PEP8 and includes tests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

If you encounter issues or have questions, feel free to open an issue! 🚀
