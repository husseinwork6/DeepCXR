# DeepCXR: ResNet-Based Chest X-Ray Analysis
This repository contains the implementation of DeepCXR, a ResNet-based model for chest X-ray analysis. The model is designed to assist in the diagnosis of various thoracic diseases by analyzing chest X-ray images.
## Performance
Image-level AUROC: 0.95  

F1-Score: 0.90  

Backbone: ResNet-18  

Sampling Ratio: 0.005 (Coreset)

## Methodology
1. The framework follows the PatchCore methodology, focusing on mid-level feature representations:  

2. Feature Extraction: Uses ResNet-18 to extract high-dimensional patches from chest radiographs.  

3. Memory Bank Construction: Stores historical feature representations of healthy (Normal) lungs.  

4. Coreset Sampling: Implements a greedy sub-sampling ratio of 0.005 to keep only the most diverse 0.5% of normal features, ensuring rapid diagnostic comparisons.  

5. Anomaly Scoring: New scans are compared against the memory bank via Nearest Neighbor Search; significant deviations flag potential abnormalities with heatmaps for localization.

## Dependencies
This project uses **uv** for fast, reliable dependency management. The primary libraries required are:
*   **anomalib**: Core framework for unsupervised anomaly detection.
*   **torch / torchvision**: Deep learning engine and computer vision utilities.
*   **lightning**: High-level interface for PyTorch used by the engine.
*   **pillow / matplotlib**: Image processing and result visualization.

## Usage
### 1. Installation
Ensure you have uv installed, then sync the environment:
```bash
uv sync
```
### 2. Data Preparation

Place your X-ray images in a folder named dataset in the root directory following this structure:

dataset/train/NORMAL: Healthy images for memory bank construction.

dataset/test/: Folders containing DEFECT, TB, or NORMAL images for evaluation.

### 4. Evaluation
5. After training, evaluate the model on the test set:
```bash
uv run python -m notebooks.DeepCXR
```
## Team and Acknowledgments
Hussein Hodroj - Middle East University  

Thabit Bustanji - Middle East University  

Supervisor: Dr. Maria Yousef (Middle East University)

## Publication
This research was presented at the ICETES 2026 conference. The work is indexed in IEEE Xplore and Scopus.

## Contact
For further information or collaboration opportunities, please contact Hussein Hodroj at [husseinwork6@gmail.com]

## Citation
If you use this code in your research, please cite our paper:
```
@inproceedings{hodroj2026deepcxr,
  title={DeepCXR: A Deep Learning Framework for Automated Chest X-Ray Analysis Using ResNet Architectures},
  author={Hodroj, Hussein and Bustanji, Thabit and Yousef, Dr. Maria},
  booktitle={2026 International Conference on Emerging Technologies in Engineering and Sciences (ICETES)},
  year={2026},
  publisher={IEEE},
  note={Indexed in Scopus and IEEE Xplore}
}
```

