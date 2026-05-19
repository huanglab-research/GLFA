# GLFA: Global-Local Frequency-Aware Network for Sea Surface Temperature Super-Resolution

This is the implementation of paper: Global-Local Frequency-Aware Network for Sea Surface Temperature Super-Resolution.

> ⚠ **Note:** The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.
---

## Datasets
Experiments on OISST and OSTIA:

|#| Datasets     |Download|
|---|--------------|-----|
|1| OISST        |[Link](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr)|
|2| OSTIA        |[Link](https://ghrsst-pp.metoffice.gov.uk/ostia-website/index.html)



### Environment
- Python 3.10
- PyTorch 2.4

Installation

```bash
conda create -n GLFA python=3.10
conda activate GLFA

pip install -r requirements.txt
python setup.py develop
```

## Training

```python
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 basicsr/train.py -opt options/train/train_GLFA.yml
```

## Test

```python
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 basicsr/test.py -opt options/test/test_GLFA.yml
```

## Acknowledgments

We are very grateful for these excellent works: [BasicSR](https://github.com/XPixelGroup/BasicSR), [CATANet](https://github.com/EquationWalker/CATANet). Please follow their respective licenses for usage and redistribution. Thanks for their awesome works.

## Contact

Feel free to contact me if there is any question. (Wenyu Xu: [xwy3885@stu.ouc.edu.cn](mailto:xwy3885@stu.ouc.edu.cn), Lei Huang: [huangl@ouc.edu.cn](mailto:huangl@ouc.edu.cn))

---