# RONEK

**Reduced Order modeling for Non-Equilibrium Kinetics**

---

RONEK is a Python library designed to perform model reduction for non-equilibrium kinetics, based on balanced truncation techniques. It currently supports mixtures consisting of two species: one atom and one molecule, with a single chemical component. For instance, it can manage mixtures like $\mathcal{S}=\left[\text{N},\text{N}_2\right]$ or $\mathcal{S}=\left[\text{O},\text{O}_2\right]$.

It leverages PyTorch for GPU acceleration. This implementation excels in performance, making it well-suited for high-dimensional problems.

## Installation and Usage

To install RONEK, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/ivanZanardi/ronek.git
cd ronek
```

2. Create a Conda environment:

```bash
conda env create -f conda/env.yml
conda activate ronek
```

3. Install the package:

```bash
pip install .
```

Activate the Conda environment whenever using RONEK:

```bash
conda activate ronek
```

## Citation

If you use this code or find this work useful in your research, please cite us:

```bibtex
@article{Zanardi_2025_RONEK,
  title={Petrov-Galerkin model reduction for thermochemical nonequilibrium gas mixtures},
  journal={Journal of Computational Physics},
  volume={533},
  pages={113999},
  month={4},
  year={2025},
  issn={0021-9991},
  doi={https://doi.org/10.1016/j.jcp.2025.113999},
  url={https://www.sciencedirect.com/science/article/pii/S0021999125002827},
  author={Ivan Zanardi and Alberto Padovan and Daniel J. Bodony and Marco Panesi},
  keywords={Reduced-order modeling, Thermochemistry, Nonequilibrium, Hypersonics}
}
@inbook{Zanardi_2025_RONEK_N3,
  title={Petrov-Galerkin Model Reduction for Thermochemical Nonequilibrium Gas Mixtures: Application to the N$_2$+N System},
  author={Ivan Zanardi and Alberto Padovan and Daniel J. Bodony and Marco Panesi},
  booktitle={AIAA SCITECH 2025 Forum},
  chapter={},
  pages={},
  doi={10.2514/6.2025-2524},
  url={https://arc.aiaa.org/doi/abs/10.2514/6.2025-2524}
}
```

## Explore

Check out the [examples](https://github.com/ivanZanardi/ronek/tree/main/examples) provided in the repository to see RONEK in action.

## License

RONEK is distributed under the [Apache-2.0 license](https://github.com/ivanZanardi/ronek/blob/main/LICENSE). You are welcome to utilize, modify, and contribute to this project in accordance with the terms outlined in the license.
