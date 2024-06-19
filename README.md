# itmAFA

---

### Introduction


This repository contains the implementations of [Enhancing Image-Text Matching with Adaptive Feature Aggregation](https://ieeexplore.ieee.org/document/10446913), accepted by ICASSP 2024.


### Usage

0. Put the data files in `data` folder before running the following commands. For more details on the data files, please refer to [SCAN](https://github.com/kuanghuei/SCAN), [vsepp](https://github.com/fartashf/vsepp), and [vse_infty](https://github.com/woodfrog/vse_infty).


1. To train the models, use
```bash
./train_{DATASET}.sh
```

2. To evaluate the models, use
```bash
./inference_{DATASET}.sh
```

- Notes: 
  - `{DATASET}` can be `f30k` or `coco`.
  - Checkpoints are saved in the `models` folder.


### Acknowledgements

Our implementations are based on [SCAN](https://github.com/kuanghuei/SCAN), [vsepp](https://github.com/fartashf/vsepp), [vse_infty](https://github.com/woodfrog/vse_infty), and other repositories. We give credit to all these researchers and sincerely appreciate their contributions.



### Citation

If you find the paper and the code useful, please cite our paper as follows:

@INPROCEEDINGS{10446913,
  author={Wang, Zuhui and Yin, Yunting and Ramakrishnan, I.V.},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Enhancing Image-Text Matching with Adaptive Feature Aggregation}, 
  year={2024},
  volume={},
  number={},
  pages={8245-8249},
  keywords={Training;Deep learning;Adaptation models;Codes;Aggregates;Speech enhancement;Signal processing;triplet ranking loss;feature enhancement;cross-modal retrieval;image-text matching},
  doi={10.1109/ICASSP48485.2024.10446913}
}

