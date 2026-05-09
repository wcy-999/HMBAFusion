# HMBAFusion

Official implementation of:

**"Multi-Scale Global-Regional-Local Attention Fusion for Enhanced Medical Image Visualization and Integration"**

[![DOI](https://zenodo.org/badge/1227232906.svg)](https://doi.org/10.5281/zenodo.20098163)

---

## Introduction

This repository provides the implementation of HMBAFusion, a hierarchical multi-branch attention framework for medical image fusion.

The proposed method integrates global contextual information, regional structural representations, and local visual details for multi-modal medical image integration tasks.

The framework is evaluated on:

- MRI-CT fusion
- MRI-PET fusion
- MRI-SPECT fusion

---

## Platform

- Python 3.7
- PyTorch >= 1.0

---

## Dataset

The training and testing datasets are constructed based on the publicly available Harvard Medical School dataset:

http://www.med.harvard.edu/AANLIB/home.html

### Experimental Settings

#### SPECT-MRI Fusion
- 333 image pairs for training
- 24 image pairs for testing

#### PET-MRI Fusion
- 245 image pairs for training
- 24 image pairs for testing

#### CT-MRI Fusion
- 160 image pairs for training
- 24 image pairs for testing

---

## Code Availability

GitHub:
https://github.com/wcy-999/HMBAFusion

DOI:
10.5281/zenodo.20098164

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{xu2026hmbafusion,
  title={Multi-Scale Global-Regional-Local Attention Fusion for Enhanced Medical Image Visualization and Integration},
  author={Xu, Yi and Wang, Chenyu and Wu, Shoucai},
  journal={The Visual Computer},
  year={2026}
}
```

---

## Contact

For questions regarding the code, please contact:

xuyi1023@126.com
