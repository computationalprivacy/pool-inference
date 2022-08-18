# Pool Inference Attacks on Local Differential Privacy
This repository contains the source code for the paper _Pool Inference Attacks on Local Differential Privacy: Quantifying the Privacy Guarantees of Apple’s Count Mean Sketch in Practice_ by A. Gadotti, F. Houssiau, M.S.M.S. Annamalai, Y.-A. de Montjoye, presented August 2022 at USENIX Security 2022 [[link](https://www.usenix.org/conference/usenixsecurity22/presentation/gadotti)]. 

## Installation
Please make sure that [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed. After that, the dependencies for this project can be installed by running the following command:
```bash
$ conda env create --file conda_deps.yml
```
Additionally, if you would like to achieve the same plot styles, make sure that [LaTeX](https://www.latex-project.org/get/) and the `LinLibertine` font is installed.

## Usage
All the experiments are given in the `experiments.ipynb` notebook which can be run individually. Each function has been annotated with a docstring for further use and modification.

Please note that the Twitter dataset used in the paper cannot be redistributed by us and therefore it is not included in this repository.

## License
GNU General Public License v3.0
