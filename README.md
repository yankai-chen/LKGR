# LKGR
This is our PyTorch implementation for our WSDM 2022 paper:

>Yankai Chen, Menglin Yang, Yingxue Zhang, Mengchen Zhao, Ziqiao Meng, Jianye Hao, Irwin King. 2022. Modeling Scale-free Graphs with Hyperbolic Geometry for
Knowledge-aware Recommendation, [arXiv](https://arxiv.org/pdf/2108.06468.pdf).

## Environment Requirement

The code runs well under python 3.7.7. The required packages are as follows:

- PyTorch-gpu == 1.4.0
- numpy == 1.21.1

## Datasets
**First**, please refer to [link](https://drive.google.com/file/d/1rnhNBNgiN76Gjd81vXEn34PCESrkrwQv/view?usp=sharing) to download the orginal rating data and put it under "/LKGR/data/". Then goto each main_xxx.py to create datasplit under the ratio 6:2:2.

For example, uncomment the following line in main_movie.py:
```
data_split(args)
```
**then** uncomment the following line to conduct the experiments:

```
Exp_run(args)
```
**Finally** run [main_xxx.py] as: 
```bash
python main_movie.py
```
