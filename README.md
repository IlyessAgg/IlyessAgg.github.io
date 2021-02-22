# Machine Learning Portfolio
> Portfolio of machine learning projects completed for academic & self-learning purposes.

<p align='center'>
  <img src="imgs/undraw_about_me_wa29.png" width="600" height="462"/>
</p>

## Academics

*A project with 📜 means I had some external help (teacher / project as part of a course).*

------

#### Predicting hotel review ratings based on text review

The goal of the project is to be able to predict the rating of a review based on text review.

|     | Reviews Text | Reviews Rating |
| :-: | :----------: | :------------: |
| 1 | This hotel was nice and quiet. Did not know, ... | 3 |
| 2 | Not cheap but excellent location. Price is ... | 5 |
| 3 | Parking was horrible, somebody ran into my ... | 3 |

After preprocessing the text (*tokenization, stopwords, lemmatization, etc.*), I used *SentimentAnalyzer* in the nltk library to rate the sentiment of a review  (~ 1 means positive, ~ -1 means negative). From that grade I build a classifier that would predict the rating of the review.

Another method I implemented was to use [fasttext](https://fasttext.cc/docs/en/supervised-tutorial.html), a library for learning of word embeddings and text classification created by Facebook's AI Research lab.

------

#### *Grand Débat* synthesis 

The goal of this project is to generate a synthesis of the data created by French citizens to answer some political open questions. 

<div align="center">
    <img src='imgs/TSNE_environment.png' height='600'>
    <p><i>TSNE Visualization of corpus</i></p>
</div>

My approach focused on the environmental questions. I also used N_Grams and LDA to pick up the subjects that came up most often in the answers. The results were issues like overconsumption, greenhouse gas emissions, extinction of species, fossil/nuclear energy, etc. Finally, I looked at whether those subjects emerged in the summary of Emmanuel Macron's appearance in the ***Grand Débat*** emission.

------

#### Scrapping & Classification of 2 dog breeds 📜<a href="https://nbviewer.jupyter.org/github/IlyessAgg/TDs_ESILV/blob/master/TD6_CNN.ipynb" align="right"><img align="right" src="imgs/jupyter_icon.png"></a>

This project was an introduction to the use of CNNs. Using *Selenium*, we scrapped images of both akita inu and giant schnauzer.
<div align="center">
    <img src='imgs/dogs.png' width='600'>
    <p><i>Example of images (giant schnauzer on the left, akita inu on the right)</i></p>
</div>
We built a basic CNN structure for this classification task. We then built another network using a pretrained model (VGG19 with Imagenet) and compared the results. Finally, we looked at the feature maps to gain a better understanding of the features detected by our model.

------

#### Activity prediction based on physiological and motion data <a href="https://github.com/IlyessAgg/PPG-DaLiA-Dataset-Analysis" align="right"><img align="right" src="imgs/GitHub-Mark-32px.png"></a><a href="https://nbviewer.jupyter.org/github/IlyessAgg/PPG-DaLiA-Dataset-Analysis/blob/master/Project.ipynb" align="right"><img align="right" src="imgs/jupyter_icon.png"></a><a href="https://github.com/IlyessAgg/PPG-DaLiA-Dataset-Analysis/blob/master/Report.pdf" align="right"><img align="right" src="imgs/PowerPoint_icon.png"></a>

**[PPG-DaLiA](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA)** is a multimodal dataset featuring physiological and motion data, recorded from both a wrist- and a chest-worn device, of 15 subjects while performing a wide range of activities under close to real-life conditions. This dataset is designed for PPG-based heart rate estimation but our instructor challenged us to predict the activity of the subjects.

<div align="center">
    <img src='imgs/heartrate_chart.png'>
    <p><i>Plot of the heartrate during the different activities.</i></p>
</div>
The purpose of this project was to deal with diverse sources of data and determine how to make use of them for our task. For example, each sensor had a different frequency so understanding and standardizing the different attributes was the essential part of the project. 

Since the dataset provides the data for only 15 subjects, I aimed my study at looking how the model developed would perform on unseen data.

------





*still things to add* 





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
