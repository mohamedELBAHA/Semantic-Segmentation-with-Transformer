# Transformers for multi-organ segmentation in abdominal imaging

### Context
The rapid development of non-invasive medical imaging technologies has opened new horizons in the study of
anatomy. The segmentation of 3D images by magnetic resonance imaging (MRI) has become a
a crucial step for many applications: computer-aided diagnosis, surgical planning, image-guided interventions, extraction of quantitative indices...
However, the analysis of complex and heterogeneous imaging data sets is tedious for radiologists, clinicians and
researchers. Moreover, the techniques developed to date are not entirely robust to low contrast
fully robust to low contrast, anatomical variability and noise intrinsic to medical images.
medical images. Although time consuming and subject to significant intra/inter-expert variability, segmentation of anatomical
structures is still largely performed manually, slice by slice.
Methodological developments allowing an automated and precise delimitation of structures of interest in
medical imaging are required, in particular for the segmentation of abdominal organs (liver
abdomen (liver, kidneys, spleen, pancreas...).

![image](pictures/Picture1.png)


----
In this project, we implemented different models for medical image semantic segmentation using pytorch, this repository hosts the code for the following Networks :
1. Unet ( Full convolutional neural network ).
2. Segmenter ( Full transformers ).
3. Medical Transfomer & TransUnet( Hybrid architectures: CNN + Transformers ). 

### Implementation:

Please prepare an environment with python, and then use the command "pip install -r requirements.txt" for the dependencies.

Please follow the instructions in the README.md in the ![/data](data/README.md) directory to be able to use the code in this github correctly.


If you want to use the transunet model where the encoder weights have already been trained on imagenet, please follow the instructions in the README in the "segmentation_models/transunet/vit_checkpoint" directory

### Acknowledgement:
The work was supervised by Mr Pierre-Henri Conze and Mr Vincent Jaouen.

### Citations

@article{chen2021transunet,
  title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L., and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
