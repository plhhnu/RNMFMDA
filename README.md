# RNMFMDA

Supporting Information for the paper "[RNMFMDA:A microbe-disease association identification method based on reliable negative sample selection and logistic matrix factorization with neighborhood regularization](https://www.frontiersin.org/)"

by Lihong Peng, Ling Shen, Longjie Liao, Guangyi Liu, Liqian Zhou, Frontiers in Microbiology, section Systems Microbiology.

## dataset
Data is available at [HMDAD](http://www.cuilab.cn/hmdad).

## Usage
Install python3.* for runing this code. And these packages should be satisified:
+ numpy
+ scikit-learn
+ scipy

To run the model, default 5 fold cross validation and 1 negative sample scale
```shell
python main.py
```

set 5 fold cross validaion to run the model
```shell
python main.py --Kfold_num 5
```

set negative sample scale to run the model
```shell
python main.py --rn_scale 1
```