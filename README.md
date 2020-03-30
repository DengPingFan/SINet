# SINet (CVPR2020)
Search and Identification Net (SINet) for Camouflaged Object Detection (code will be updated soon)<br>
![alt text](./CamouflagedTask.png)
Figure 1: Given an input image (a), we present the ground-truth for (b) panoptic segmentation (which detects generic objects including stuff and things), (c) salient instance/object detection (which detects objects that grasp human attention), and (d) the proposed camouflaged object detection task, where the goal is to detect objects that have a similar pattern (e.g., edge, texture, or color) to the natural habitat. In this case, the boundaries of the two butterflies are blended with the bananas, making them difficult to identify. This task is far more challenging than the traditional salient object detection or generic object detection. <br>
![alt text](./CmpResults.png)
Figure 2: Qualitative results of our SINet and two top-performing baselines on COD10K. Refer to our paper for details.

## Task Relationship
![alt text](./TaskDifference.png)

## Results
Results of our SINet on four datasets (e.g., CHAMELEON[1], CPD1K-Test[2], CAMO-Test[3], and COD10K-Test[4]) can be found:<br> 

https://drive.google.com/open?id=1fHAwcUwCjBKSw8eJ9OaQ9_0kW6VtDZ6L

Performance of competing methods can be found:

https://drive.google.com/open?id=1jGE_6IzjGw1ExqxteJ0KZSkM4GaEVC4J

References of datasets<br>
[1] Animal camouflage analysis: Chameleon database. Unpublished Manuscript, 2018. <br>
[2] Detection of people with camouflage pattern via dense deconvolution network. IEEE SPL, 2018.<br>
[3] Anabranch network for camouflaged object segmentation. CVIU, 2019.<br>
[4] Camouflaged Object Detection, CVPR, 2020.

## Datasets
Our training dataset is:

https://drive.google.com/open?id=1aH9_0w3zCVoh9ttrU10xjCYcjuvPuzWY

Our testing dataset is:

https://drive.google.com/open?id=1AeJBD-FemHSVdprC8_6BOi41Wt5KgMIt

## Applications
1. Medical (polyp segmentation)
![alt text](./PolypSegmentation.png)
PraNet: Parallel Reverse Attention Network for Polyp Segmentation, MICCAI 2020 (submitted).
2. Agriculture (locust detection to prevent invasion)
3. Art (e.g., for photorealistic blending, or recreational art)
4. Military (for discriminating enemies)
![alt text](./Telescope.png)
5. Computer Vision (e.g., for search-and-rescue work, or rare species discovery),
![alt text](./Search-and-Rescue.png)
## Paper

http://dpfan.net/wp-content/uploads/COD_CVPR20-OmittedModel.pdf

## Citation
Please cite our paper if you find the work useful: 

	@inproceedings{fan2020Camouflage,
  	title={Camouflaged Object Detection},
  	author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  	booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  	year={2020}
	}
  
