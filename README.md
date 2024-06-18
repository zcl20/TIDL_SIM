# Fast_DL_SIM-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of TIDL_SIM

## Table of contents

- [TIDL_SIM-PyTorch](#srgan-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test-srganx4)
        - [Train](#train-srresnetx4)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)

## Download weights

Download all available model weights.
Download `` weights to `/Model_output/out1`

## Generate datasets
```shell
python Generate_Rawimage.py
```

## How Test and Train

Both training and testing only need to modify yaml file.

### Test 

```shell
python Test.py --config_path Model_output/out1/tidl_sim_div2k.yaml
```

### Train 

```shell
python  Train.py --config_path  Configs/tidl_sim_div2k.yaml
```

## Result

Source of original paper results:

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## 

**Abstract** <br>

