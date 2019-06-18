# HomoInterpGAN
Homomorphic Latent Space Interpolation for Unpaired Image-to-image Translation (CVPR 2019, oral)

# Installation

The implementation is based on [pytorch](pytorch.org). Our model is trained and tested on version 1.0.1.post2. Please install relevant packages based on your own environment. 

All other required packages are listed in "requirements.txt". Please run 

```bash
pip install -r requirements.txt
```

to install these packages. 

# Dataset

Download the "Align&Cropped Images" of the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. 
If the original link is unavailable, you can also download it [here](https://www.kaggle.com/jessicali9530/celeba-dataset).

# Training

Firstly, cd to the project directory and run
```bash
export PYTHONPATH=./:$PYTHONPATH
```

before executing any script. 

To train a model on CelebA, please run

```bash
python run.py train --data_dir CELEBA_ALIGNED_DIR -sp checkpoints/CelebA -bs 128 -gpu 0,1,2,3 
```
Key arguments

```bash
--data_dir: The path of the celeba_aligned images. 
-sp: The trained model and logs, intermediate results are stored in this directory.
-bs: Batch size.
-gpu: The GPU index.
--attr: This specifies the target attributes. Note that we concatenate multiple attributes defined in CelebA as our grouped attribute. We use "@" to group multiple multiple attributes to a grouped one (e.g., Mouth_Slightly_Open@Smiling forms a "expression" attriute). We use "," to split different grouped attributes. See the default argument of "run.py" for details. 
```



# Testing

```bash
python run.py attribute_manipulation -mp checkpoints/CelebA -sp checkpoints/CelebA/test/Smiling  --filter_target_attr Smiling -s 1 --branch_idx 0 --n_ref 5 -bs 8
```
This conducts attribute manipulation with reference samples selected in CelebA dataset. The reference samples are selected based on their attributes (--filter_target_attr), and the interpolation path should be chosen accordingly. 

Key arguments:

```bash
-mp: the model path. The checkpoints of encoder, interpolator and decoder should be stored in this path.
-sp: the save path of the results.
--filter_target_attr: This specifies the attributes of the reference images. The attribute names can be found in "info/attribute_names.txt". We can specify one attribute (e.g., "Smiling") or several attributes (e.g., "Smiling@Mouth_Slightly_Open" will filter mouth open smiling reference images). To filter negative samples, add "NOT" as prefix to the attribute names, such as "NOTSmiling", "NOTSmiling@Mouth_Slightly_Open".
--branch_idx: This specifies the branch index of the interpolator. Each branch handles a group of attribute. Note that the physical meaning of each branch is specified by "--attr" during testing. 
-s: The strength of the manipulation. Range of [0, 2] is suggested. If s>1, the effect is exaggerated.
-bs: the batch size of the testing images. 
-n_ref: the number of images used as reference. 
```

## Testing on unaligned images

Note the the performance could degenerate if the testing image is not well aligned. Thus we also provide a tool for face alignment. Please place all your testing images to a folder (e.g., examples/original), then run 

```bash
python facealign/align_all.py examples/original examples/aligned
```

to align testing images to an samples in CelebA. Then you can run manipulation by 

```bash
python run.py attribute_manipulation -mp checkpoints/CelebA -sp checkpoints/CelebA/test/Smiling  --filter_target_attr Smiling -s 1 --branch_idx 0 --n_ref 5 -bs 8 --test_folder examples/aligned
```

Note that an additional argument "--test_folder" is specified. 

# Pretrained model

We have also provided a pretrained model [here](https://www.dropbox.com/sh/31dki21jqaifzjj/AACPH11iwBK-38rFy8oKqeraa?dl=0). It is trained with default parameters. The meaning of each branch of the interpolator is listed bellow.

| Branch index | Grouped attribute |        Corresponding labels on CelebA         |
| :----------: | :---------------: | :-------------------------------------------: |
|      1       |    Expression     |         Mouth_Slightly_Open, Smiling          |
|      2       |   Gender trait    |  Male, No_Beard, Mustache, Goatee, Sideburns  |
|      3       |    Hair color     | Black_Hair, Blond_Hair, Brown_Hair, Gray_Hair |
|      4       |    Hair style     |        Bald, Receding_Hairline, Bangs         |
|      5       |        Age        |                     Young                     |

# Reference

[Ying-Cong Chen](http://appsrv.cse.cuhk.edu.hk/~ycchen/), Xiaogang Xu, Zhuotao Tian, [Jiaya Jia](http://jiaya.me/), "Homomorphic Latent Space Interpolation for Unpaired Image-to-image Translation" , *Computer Vision and Pattern Recognition (CVPR), 2019* [PDF](<http://appsrv.cse.cuhk.edu.hk/~ycchen/pdfFiles/HomoInterp.pdf>)

```
@inproceedings{chen2019Homomorphic,
  title={Homomorphic Latent Space Interpolation for Unpaired Image-to-image Translation},
  author={Chen, Ying-Cong and Xu, Xiaogang and Tian, Zhuotao and Jia, Jiaya},
  booktitle={CVPR},
  year={2019}
}
```

# Contect

Please contact [yingcong.ian.chen@gmail.com](mailto:yingcong.ian.chen@gmail.com) if you have any question or suggestion.