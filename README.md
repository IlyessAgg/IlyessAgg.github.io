# Machine Learning Portfolio
> Portfolio of machine learning projects completed for academic & self-learning purposes.

<p align='center'>
  <img src="imgs/undraw_about_me_wa29.png" width="600" height="462"/>
</p>

## Academics

*to be done*






## Personal


#### Detecting breast cancer metastases <a href="https://github.com/IlyessAgg/Metastasis_Detection" align="right"><img align="right" src="imgs/GitHub-Mark-32px.png"></a>

The goal of the challenge is to detect **lymph node metastases** in histological images of patients diagnosed with breast cancer. Each patient is described by **1,000** small images (tiles) extracted from one whole-slide image. 

<figure align='center'>
   <img src="https://i.postimg.cc/7LBRCxWF/Capture-d-e-cran-2019-01-25-a-10-45-05.png" alt='Tumor_images'/>
   <figcaption><i>Example of two tiles (non-tumoral on the left, tumoral on the right)</i></figcaption>
</figure>

Additionally, we have **11** patients for which each tile was annotated by a pathologist (total of **10,024** annotated tiles). Based on these annotated patients, I built a model that predicts the **probability of metastases in the tile**. Next, I computed the probabilities for every tile of the non-annotated patients. Based on these predictions, I built a model that predicts whether or not **a patient has any metastases** in its slide.

------


#### Predicting lung cancer survival time <a href="https://github.com/IlyessAgg/Owkin_Challenge" align="right"><img align="right" src="imgs/GitHub-Mark-32px.png"></a><a href="https://nbviewer.jupyter.org/github/IlyessAgg/Owkin_Challenge/blob/master/Challenge_Second_Try.ipynb" align="right"><img align="right" src="imgs/jupyter_icon.png"></a>


The goal of the challenge is to predict **survival time** of patients diagnosed with lung cancer, based on 3-dimensional radiology images (CT scans). Clinical data of patients and radiomics (quantitative features extracted from the scan) are also provided.

<div class='images' align='center'>
    <figure style='display:inline-block;vertical-align:top;'>
        <img src="imgs/lung_ct.png" alt='Lung_ct_Image' width='235' height='100'/>
        <figcaption><i>CT Scan</i></figcaption>
    </figure>
    <figure style='display:inline-block;vertical-align:top;'>
        <img src="imgs/lung_ct_and_mask.png" alt='Mask_Image' width='235' height='100'/>
        <figcaption><i>Mask of tumor (green part)</i></figcaption>
    </figure>
</div>

Based on the imaging modality (scan & mask), I re-computed radiomics features using [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/) (the radiomics data provided was only a subset of what could be computed using that library). After a feature selection process using ***VIF***, I built a **Cox Proportional Hazard** model to predict the survival time of the patients.

------

<div>
    <h4 style="display: inline;">Segmentation of liver tumor &nbsp;</h4>
    <img src="imgs/loadingcircles_icon.png" width='28' style="vertical-align:center;">
    <span><i>ongoing</i>
</div>

The goal of this [challenge](https://competitions.codalab.org/competitions/17094) is to segment liver lesions in contrast­-enhanced abdominal CT scans. Liver segmentation and tumor burden estimation are also evaluated. The data and segmentations are provided by various clinical sites around the world.

<figure align='center'>
   <img src="imgs/liver_segmentation.png" alt='Tumor_images' width='750'/>
   <figcaption><i>Example of a CT scan</i></figcaption>
</figure>


The first step of the process was to create a function that generates sub-volumes from our images. I also made sure that the sub-volumes extracted were relevant (i.e. at least 5% of something other than the background class). I am now building a U-Net architecture to complete the task of segmentation.
