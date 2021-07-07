# Camouflaged Object Detection (CVPR2020-Oral)

> Authors:
> [Deng-Ping Fan](https://dpfan.net/), 
> [Ge-Peng Ji](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en), 
> [Guolei Sun](https://github.com/GuoleiSun), 
> [Ming-Ming Cheng](https://mmcheng.net/), 
> [Jianbing Shen](http://iitlab.bit.edu.cn/mcislab/~shenjianbing), 
> [Ling Shao](http://www.inceptioniai.org/).


## 0. Preface

- Welcome to joint the COD community! We create a group chat in WeChat, you can join it via adding contact 
(WeChat ID: CVer222). Please attach your affiliations.

- This repository includes detailed introduction, strong baseline 
(Search & Identification Net, SINet), and one-key evaluation codes for 
**_Camouflaged Object Detection (COD)_**.

- For more information about Camouflaged Object Detection, please visit our [Project Page](http://dpfan.net/Camouflage/) 
and read the [Manuscript (PDF)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.pdf) 
/ [Chinese Version (PDF)](http://dpfan.net/wp-content/uploads/CamouflagedObjectDetection_Chinese-1.pdf).  

<!--http://dpfan.net/wp-content/uploads/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.pdf -->

- If you have any questions about our paper, feel free to contact [Deng-Ping Fan](dengpfan@gmail.com) 
or [Ge-Peng Ji](gepengai.ji@gmail.com) via E-mail. And if you are using SINet or evaluation toolbox for your research, 
please cite this paper ([BibTeX](http://dpfan.net/wp-content/uploads/Camouflage.txt)) 

### 0.1. :fire: NEWS :fire:
- [2021/07/07] :boom: **The latest enhanced version of SINet is coming**, which is accepted at IEEE TPAMI 2021 ([Paper](https://ieeexplore.ieee.org/document/9444794) | [GitHub](https://github.com/GewelsJI/SINet-V2)). The SINet-V2 can surpass the performance of existing COD methods by a large margin, while maintaining real-time inference.
- [2020/10/22] :boom: **Training code could be avaliable via email (dengpfan@gmail.com).** Please provide your Name & Institution. Please note the code can be only used for research purpose.
- [2020/11/21] Upadted evaluated tool: Bi_cam(cam>threshold)=1 -> Bi_cam(cam>=threshold)=1;
- [2020/10/22] For eq (4): j = k+1, M -> j = m, k-1. (note that m is a specific layer, in our paper it should be equal to 1).
- [2020/09/09] SINet is the best method on the open benchmark website (https://paperswithcode.com/task/camouflaged-object-segmentation).
- [2020/08/27] Updated the describtion in Table 3 (Baseline models are trained using the training setting (iii) rather than (iv)). 
- [2020/08/05] Online demo has been released! (http://mc.nankai.edu.cn/cod).
- [2020/06/11] We re-organize the training set, listed in [2.2. Usage](#22-usage) section, please download it again.
<!-- - [2020/06/11] :boom: Release train code.
- [2020/05/15] training code is coming soon ... -->
- [2020/05/05] :boom: Release testing code.
- [2020/04/25] Training/Testing code will be updated soon ...

### 0.2. Table of Contents

- [Camouflaged Object Detection (CVPR2020)](#camouflaged-object-detection--cvpr2020-)
  * [0. Preface](#0-preface)
    + [0.1. :fire: NEWS :fire:](#01--fire--news--fire-)
    + [0.2. Table of Contents](#02-table-of-contents)
    + [0.3. File Structure](#03-file-structure)
  * [1. Task Relationship](#1-task-relationship)
  * [2. Proposed Baseline](#2-proposed-baseline)
    + [2.1. Overview](#21-overview)
    + [2.2. Usage](#22-usage)
  * [3. Results](#3-results)
    + [3.1. Qualitative Comparison](#31-qualitative-comparison)
    + [3.2. Quantitative Comparison (Overall/Sub-class)](#32-quantitative-comparison--overall-sub-class-)
    + [3.3. Results Download](#33-results-download)
  * [4. Proposed COD10K Datasets](#4-proposed-cod10k-datasets)
  * [5. Evaluation Toolbox](#5-evaluation-toolbox)
  * [6. Potential Applications](#6-potential-applications)
  * [7. User Study Test](#7-user-study-test)
  * [8. Citation](#8-citation)
  * [9. LICENSE](#9-license)
  * [10. Acknowledgements](#10-acknowledgements)
  * [11. TODO LIST](#11-todo-list)
  * [12. FAQ](#12-faq)

### 0.3. File Structure
	SINet
	├── EvaluationTool
	│   ├── CalMAE.m
	│   ├── Enhancedmeasure.m
	│   ├── Fmeasure_calu.m
	│   ├── main.m
	│   ├── original_WFb.m
	│   ├── S_object.m
	│   ├── S_region.m
	│   └── StructureMeasure.m
	├── Images
	│   ├── CamouflagedTask.png
	│   ├── CamouflagingFromMultiView.png
	│   ├── CmpResults.png
	│   ├── COD10K-2.png
	│   ├── COD10K-3.png
	│   ├── COVID'19-Infection.png
	│   ├── locust detection.png
	│   ├── new_score_1.png
	│   ├── PolypSegmentation.png
	│   ├── QuantitativeResults-new.png
	│   ├── SampleAquaticAnimals.png
	│   ├── Search-and-Rescue.png
	│   ├── SINet.png
	│   ├── SubClassResults-1.png
	│   ├── SubClassResults.png
	│   ├── Surface defect Detection2.png
	│   ├── TaskRelationship.png
	│   ├── Telescope.png
	│   └── UnderwaterEnhancment.png
	├── MyTest.py
	├── README.md
	├── requirement.txt
	└── Src
	    ├── backbone
	    ├── __init__.py
	    ├── SearchAttention.py
	    ├── SINet.py
	    └── utils

## 1. Task Relationship

<p align="center">
    <img src="./Images/Relationship.png"/> <br />
    <em> 
    Figure 1: Task relationship. Given an input image (a), we present the ground-truth for (b) panoptic segmentation
    (which detects generic objects including stuff and things), (c) salient object detection (which detects isolated 
    objects that grasp human attention), and (d) the proposed concealed object detection task, where the goal is to 
    detect objects that have a similar pattern to the natural habitat. In this example, the boundaries of the two 
    butterflies are blended with the bananas, making them difficult to identify..
    </em>
</p>

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/CamouflagedTask.png"/> <br />
    <em> 
    Figure 2: Given an input image (a), we present the ground-truth for (b) panoptic segmentation 
    (which detects generic objects including stuff and things), (c) salient instance/object detection 
    (which detects objects that grasp human attention), and (d) the proposed camouflaged object detection task, 
    where the goal is to detect objects that have a similar pattern (e.g., edge, texture, or color) to the natural habitat. 
    In this case, the boundaries of the two butterflies are blended with the bananas, making them difficult to identify. 
    This task is far more challenging than the traditional salient object detection or generic object detection.
    </em>
</p>

> References of Salient Object Detection (SOD) benchmark works<br>
> [1] Video SOD: Shifting More Attention to Video Salient Object Detection. CVPR, 2019. ([Project Page](http://dpfan.net/davsod/))<br>
> [2] RGB SOD: Salient Objects in Clutter: Bringing Salient Object Detection to the Foreground. ECCV, 2018. ([Project Page](https://dpfan.net/socbenchmark/))<br>
> [3] RGB-D SOD: Rethinking RGB-D Salient Object Detection: Models, Datasets, and Large-Scale Benchmarks. TNNLS, 2020. ([Project Page](http://dpfan.net/d3netbenchmark/))<br>
> [4] Co-SOD: Taking a Deeper Look at the Co-salient Object Detection. CVPR, 2020. ([Project Page](http://dpfan.net/CoSOD3K/))

## 2. Proposed Baseline

### 2.1. Overview

<p align="center">
    <img src="Images/SINet.png"/> <br />
    <em> 
    Figure 3: Overview of our SINet framework, which consists of two main components: the receptive field (RF) 
    and partial decoder component (PDC). The RF is introduced to mimic the structure of RFs in the human visual system. 
    The PDC reproduces the search and identification stages of animal predation. 
    SA = search attention function described in [71]. See x 4 for details.
    </em>
</p>

### 2.2. Usage

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single GeForce RTX TITAN GPU of 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size
(~419 MB per image in `apex-mode=O1`, and ~305 MB per image in `apex-mode=O2`)


1. Configuring your environment (Prerequisites):
   
    Note that SINet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n SINet python=3.6`.
    
    + Installing necessary packages: `pip install -r requirements.txt`.
    
    + (Optional: only for training) Installing [NVIDIA-Apex](https://github.com/NVIDIA/apex) 
    for accelerate training process with mixed precision. 
    [(Instructions)](https://github.com/NVIDIA/apex#linux) (Under CUDA-10.0 and Cudnn-7.4).

<!--2. Downloading Testing Sets: -->
2. Downloading Training and Testing Sets:
    + downloading _**NEW testing dataset**_ (COD10K-test + CAMO-test + CHAMELEON) and move it into `./Dataset/TestDataset/`, 
    which can be found in this [Google Drive link](https://drive.google.com/file/d/1QEGnP9O7HbN_2tH999O3HRIsErIVYalx/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/143yHFLAabMBT7wgXA0LrMg) with the fetch code: z83z.
    
    + download **_NEW training dataset_** (COD10K-train) which can be found in this [Google Drive link](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing) or [Baidu Pan link](https://pan.baidu.com/s/1XL6OjpDF-MVnXOY6-bdaBg) with the fetch code:djq2.  Please refer to our original paper for other training data. 
    
<!-- + download **_NEW training dataset_** (COD10K-train + CAMO-train) and move it into `./Dataset/TrainDataset/`, which can be found in this [download link](https://drive.google.com/file/d/1D9bf1KeeCJsxxri6d2qAC7z6O1X_fxpt/view?usp=sharing). -->
    

<!--3. Training Configuration:

    + Assigning your customed path, like `--save_model`, `--train_img_dir`, and `--train_gt_dir` in `MyTrain.py`.
    
    + Just run it! -->

3. Testing Configuration:

    + After you download all the pre-trained model and testing data, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--model_path`) and assign your the save directory of the inferred mask (`--test_save`)
    
    + Note that we re-trained our model (marked as $\diamondsuit$ in the following figure) equipped with mixed training 
    strategy of Apex lib (`mode=O1`) and get better performance in 40 epoch. Here we provide a new pre-trained model 
    ([Baidu Drive [fetch code:2pp2]](https://pan.baidu.com/s/1wcUZA9wmM6grAL3BvofBbQ)/[Google Drive](https://drive.google.com/open?id=1iCkHXU55Or4gEtSqesijea6A5r8lVEYv)) here. Later, We will try different backbones based SINet to improve performance 
    and provide more comprehensive comparison.  

    <p align="center">
        <img src="http://dpfan.net/wp-content/uploads/new_score_1.png"/> <br />
    </p>

4. Evaluation your trained model:

    + One-key evaluation is written in MATLAB code (revised from [link](https://github.com/DengPingFan/CODToolbox)), 
    please follow this the instructions in `main.m` and just run it to generate the evaluation results in 
    `./EvaluationTool/EvaluationResults/Result-CamObjDet/`.

## 3. Results

### 3.1. Qualitative Comparison
<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/CmpResults.png"/> <br />
    <em> 
    Figure 4: Qualitative results of our SINet and two top-performing baselines on COD10K. Refer to our paper for details.
    </em>
</p>

### 3.2. Quantitative Comparison (Overall/Sub-class)

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/QuantitativeResults-new.png"/> <br />
    <em> 
    Table 1: Quantitative results on different datasets. The best scores are highlighted in bold.
    </em>
</p>


<p align="center">
    <img width="860" height="1060" src="http://dpfan.net/wp-content/uploads/SubClassResults.png"/> <br />
    <em> 
    Table 2: Quantitative results of Structure-measure (Sα) for each sub-class in our COD10K dataset-(1/2). The best
    score of each category is highlighted in bold.
    </em>
</p>


<p align="center">
    <img width="850" height="1050" src="http://dpfan.net/wp-content/uploads/SubClassResults-1.png"/> <br />
    <em> 
    Table 3: Quantitative results of Structure-measure (Sα) for each sub-class in our COD10K dataset-(2/2). The best
    score of each category is highlighted in bold.
    </em>
</p>

### 3.3. Results Download 

1. Results of our SINet can be found in this [download link](https://drive.google.com/open?id=1fHAwcUwCjBKSw8eJ9OaQ9_0kW6VtDZ6L).

2. Performance of competing methods can be found in this [download link](https://drive.google.com/open?id=1jGE_6IzjGw1ExqxteJ0KZSkM4GaEVC4J).

## 4. Proposed COD10K Datasets

<p align="center">
    <img width="850" height="750" src="http://dpfan.net/wp-content/uploads/SampleAquaticAnimals.png"/> <br />
    <em> 
    Figure 5: The extraction of individual samples including 20 sub-classes from our COD10K (2/5)–Aquatic animals.
    </em>
</p>

<p align="center">
    <img width="850" height="680" src="http://dpfan.net/wp-content/uploads/COD10K-2.png"/> <br />
    <em> 
    Figure 6: Annotation diversity and meticulousness in the proposed COD10K dataset. Instead of only providing coarse-grained
    object-level annotations with the three major types of bias (e.g., Watermark embedded, Coarse annotation, and Occlusion) 
    in prior works, we offer six different annotations, which include edge-level (4rd row), object-level (5rd row), 
    instance-level (6rd row), bounding boxes (7rd row), and attributes (8rd row). Refer to the manuscript for more attribute 
    details.
    </em>
</p>

<p align="center">
    <img width="850" height="440" src="http://dpfan.net/wp-content/uploads/COD10K-3.png"/> <br />
    <em> 
    Figure 7: Regularized quality control during our labeling reverification stage. Strictly adheres to the 
    four major criteria of rejection or acceptance to near the ceiling of annotation accuracy.
    </em>
</p>

> COD10K datasets: [Baidu](https://pan.baidu.com/s/15GnPrMd6TGzMufEQ9IC66Q) aq4i | [Google](https://drive.google.com/file/d/1pVq1rWXCwkMbEZpTt4-yUQ3NsnQd_DNY/view?usp=sharing)

## 5. Evaluation Toolbox

We provide complete and fair one-key evaluation toolbox for benchmarking within a uniform standard. 
Please refer to this link for more information:
Matlab version: https://github.com/DengPingFan/CODToolbox
Python version: https://github.com/lartpang/PySODMetrics

## 6. Potential Applications

1. Medical (Polyp Segmentation and COVID-19 Infection Segmentation Diagnose)
Please refer to this page (https://github.com/DengPingFan/Inf-Net) for more details.
<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/PolypSegmentation.png"/> <br />
    <em> 
    Figure 8: Lung Infection Segmentation.
    </em>
</p>
​    

<p align="center">
    <img width="600" height="230" src="http://dpfan.net/wp-content/uploads/COVID19-Infection.png"/> <br />
    <em> 
    Figure 9: Example of COVID-19 infected regions in CT axial slice, where the red and green regions denote the GGO, 
    and consolidation, respectively. The images are collected from here. 
    (COVID-19 CT segmentation dataset (link: https://medicalsegmentation.com/covid19/, accessed: 2020-04-11).)
    </em>
</p>


2. Agriculture (locust detection to prevent invasion)

<p align="center">
    <img width="600" height="230" src="http://dpfan.net/wp-content/uploads/locust-detection.png"/> <br />
    <em> 
    Figure 10: Locust disaster detection.
    </em>
</p>

3. Art (e.g., for photorealistic blending, or recreational art)

<p align="center">
    <img width="600" height="230" src="http://dpfan.net/wp-content/uploads/CamouflagingFromMultiView.png"/> <br />
    <em> 
    Figure 11: The answer can be found at here (Camouflaging an Object from Many Viewpoints, CVPR 2014.)
    </em>
</p>

4. Computer Vision (e.g., for search-and-rescue work, or rare species discovery)

<p align="center">
    <img width="600" height="230" src="http://dpfan.net/wp-content/uploads/Search-and-Rescue.png"/> <br />
    <em> 
    Figure 13: Search and Rescue for saving lives.
    </em>
</p>

5. Underwater Image Enhancement

<p align="center">
    <img width="2014" height="320" src="http://dpfan.net/wp-content/uploads/UnderwaterEnhancment.png"/> <br />
    <em> 
    Figure 14: Please refer to "An Underwater Image Enhancement Benchmark Dataset and Beyond, TIP2019" for more details.
    </em>
</p>

6. Surface defect Detection

<p align="center">
    <img width="1000" height="320" src="http://dpfan.net/wp-content/uploads/Surface-defect-Detection2.png"/> <br />
    <em> 
    Figure 15: Please refer to "A review of recent advances in surface defect detection using texture analysis techniques, 2008" for more details.
    </em>
</p>
## 7. User Study Test

[--> Click here to explore more interest things (YouTube Link) <--](https://youtu.be/Ovv_leSGKDw)

## 8. Citation

Please cite our paper if you find the work useful: 

	@inproceedings{fan2020Camouflage,
  	title={Camouflaged Object Detection},
  	author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  	booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  	year={2020}
	}

## 9. LICENSE

- The COD10K Dataset is made available for non-commercial purposes only.

- You will not, directly or indirectly, reproduce, use, or convey the COD10K Dataset 
or any Content, or any work product or data derived therefrom, for commercial purposes.

This code is for academic communication only and not for commercial purposes. 
If you want to use for commercial please contact me.

Redistribution and use in source with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
  
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in
  the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 	
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## 10. Acknowledgements

We would like to thank authors of CHAMELEON, and CAMO dataset for their work. 
They provide tremendous efforts in these dataset to boost this field. 
We also appreciate image annotators and 
[Wenguan Wang](https://scholar.google.com/citations?user=lZQRrrkAAAAJ&hl=zh-CN), 
[Geng Chen](https://www.researchgate.net/profile/Geng_Chen13), 
[Hongsong Wang](https://scholar.google.com/citations?hl=zh-CN&user=LzQnGacAAAAJ) for insightful feedback and discussion.

## 11. TODO LIST

> If you want to improve the usability or any piece of advice, please feel free to contact me directly ([E-mail](gepengai.ji@gmail.com)).

- [ ] Support `NVIDIA APEX` training.

- [ ] Support different backbones (
VGGNet, 
ResNet, 
[ResNeXt](https://github.com/facebookresearch/ResNeXt)
[Res2Net](https://github.com/Res2Net/Res2Net-PretrainedModels), 
[iResNet](https://github.com/iduta/iresnet), 
and 
[ResNeSt](https://github.com/zhanghang1989/ResNeSt) 
etc.)

- [ ] Support distributed training.

- [ ] Support lightweight architecture and real-time inference, like MobileNet, SqueezeNet.

- [ ] Support distributed training

- [ ] Add more comprehensive competitors.

## 12. FAQ

1. If the image cannot be loaded in the page (mostly in the domestic network situations).

    [Solution Link](https://blog.csdn.net/weixin_42128813/article/details/102915578)
    
---

**[⬆ back to top](#0-preface)**
