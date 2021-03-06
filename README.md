# LKGR
This is the PyTorch implementation for paper "Modeling Scale-free Graphs with Hyperbolic Geometry for
Knowledge-aware Recommendation", [arXiv](https://arxiv.org/pdf/2108.06468.pdf). Yankai Chen, Menglin Yang, Yingxue Zhang, Mengchen Zhao, Ziqiao Meng, Jianye Hao, Irwin King. 2022. 

## Environment Requirement

The code runs well under python 3.8.0. The required packages are as follows:

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

You can also download our five experimental random splits for each datasets via [link](https://drive.google.com/file/d/1YlMqQl4pxnV2cwfJnsmAw_4tfmfDYNcN/view?usp=sharing).


## Citation
Please kindly cite our paper if you find this code useful for your research:

```
@inproceedings{chen2021modeling,
  title     = {Modeling Scale-free Graphs with Hyperbolic Geometry for Knowledge-aware Recommendation},
  author    = {Chen, Yankai and Yang, Menglin and Zhang, Yingxue and Zhao, Mengchen and Meng, Ziqiao and Hao, Jianye and King, Irwin},
  booktitle = {{WSDM} '22: The Fifteenth {ACM} International Conference on Web Search
               and Data Mining, Virtual Event / Tempe, AZ, USA, February 21 - 25,
               2022},
  pages     = {94--102},
  publisher = {{ACM}},
  year      = {2022},
}
```
