# Enforcing Hidden Physics in Physics-Informed Neural Networks

This repository implements a novel framework for training Physics-Informed Neural Networks (PINNs) that explicitly enforces **hidden irreversibility constraints** in physical systems. Our approach addresses a critical gap in conventional PINNs by incorporating the fundamental directionality implied by the Second Law of Thermodynamics into the training process.

## Key Contributions

- A systematic identification of irreversibility as a hidden physical principle that is often violated in conventional PINNs
- A simple yet robust regularization strategy that explicitly enforces irreversibility constraints during PINN training
- Comprehensive validation across five challenging benchmarks, demonstrating significant improvements in accuracy and physical consistency with minimal implementation effort

## Methodology

### Irreversibility Constraint Formulation

Our approach augments the standard PINN loss function with an irreversibility regularization term:

```
L_total = w_g * L_g + w_b * L_b + w_i * L_i + w_irr * L_irr
```

Where the irreversibility loss is defined as:

**For forward irreversible processes** (quantities that should only increase):
```python
L_irr = mean(ReLU(-∂φ/∂t))  # or ∂φ/∂x for spatial irreversibility
```

**For backward irreversible processes** (quantities that should only decrease):
```python
L_irr = mean(ReLU(∂φ/∂t))   # or ∂φ/∂x for spatial irreversibility
```

The ReLU activation ensures that only violations of the irreversibility constraint contribute to the loss, making the approach both elegant and computationally efficient.

## Benchmark Results

We evaluate our framework across five challenging applications spanning two types of irreversibility:

### Directional Irreversibility
- **Traveling Wave Propagation**: Fisher-type reaction-diffusion equation with spatial irreversibility
- **Steady Combustion**: One-dimensional combustion front with directional constraints

### Dissipative Irreversibility  
- **Corrosion Evolution**: 2D pitting corrosion with temporal irreversibility (φ can only decrease)
- **Ice Melting**: Stefan problem with advancing melting front
- **Fracture Mechanics**: Phase-field fracture with irreversible damage accumulation

### Performance Comparison

The following table summarizes the relative errors L2 (%) achieved by our irreversibility-regularized PINNs compared to conventional baseline PINNs:

| Problem | Baseline PINN| IRR. PINN |
|---------|-------------------|-----------|
| Traveling Wave | 100 | 1.02 |
| Combustion | 54.9 | 0.464 |
| Ice Melting | 0.696 | 0.164 |
| Corrosion | 4.07 | 0.118 |
| Fracture | 7.28 | 2.15 |

*Relative errors calculated as L² norm relative to finite element method (FEM) reference solutions*


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{chen2024enforcing,
  title={Enforcing hidden physics in physics-informed neural networks},
  author={Chen, Nanxi and Cui, Chuanjie and others},
  journal={arXiv preprint},
  year={2024}
}
```
