# How to use the dataset

## Instrcution

This readme file is for **DIPA** (Dataset with image privacy annotations).

For the corresponding paper, please refer to **DIPA: An Image Dataset with Cross-cultural Privacy Concern Annotations** in IUI'23 Open Science Track.

Our work is licensed under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. All DIPA dataset images come from [LVIS dataset](https://www.lvisdataset.org/) (extended from [COCO dataset](https://cocodataset.org/#home)) and  [OpenImages V6 dataset](https://storage.googleapis.com/openimages/web/index.html). We welcome you to use or expand DIPA.

If you have any question, please send emails to anran[at]iis-lab.org.

## File structure

- DIPA_lvis_map.csv --- the map from 22 categories of privacy-threatening content we identified with original categories in [LVIS dataset](https://www.lvisdataset.org/) 

- DIPA_openimages_map.csv --- the map from 22 categories of privacy-threatening content we identified with original categories in [OpenImages V6 dataset](https://storage.googleapis.com/openimages/web/index.html)

- oidv6-class-descriptions.csv --- Provided by [OpenImages v6](https://storage.googleapis.com/openimages/web/download_v6.html). It maps the the code of category to the name of it. 

- img_annotation_map.json --- the map of file name from images to privacy-oriented annotations.

- images --- store all image data in their original file names provided by OpenImages and LVIS.

- original labels --- original labels from OpenImages and LVIS dataset in unified format.

- annotations --- store all privacy-oriented labels and worker information collected from [CrowdWorks](https://crowdworks.jp/) and [Prolific](https://www.prolific.co/), which are two crowdsourcing platforms.

  

## Data structure 

### DIPA_lvis_map.csv

The two-column csv are written in DIPA Category, LVIS Category.

One DIPA category is corresponded with multiple categories in LVIS. 

Each LVIS Category is splited by '|'.

### DIPA_openimages_map.csv

The two-column csv are written in DIPA Category, OpenImages Category.

One DIPA category is corresponded with multiple categories in OpenImages. 

Each OpenImages Category is splited by '|'. 

Please note that the OpenImages categories are written in code.

### oidv6-class-descriptions.csv 

The two-column csv are written in  Code, Name.

You may correspond the code with its actual name when dealing annotations from OpenImages.

### img_annotation_map.json

Each element in this json file has the following key-value structure.

`{imgId: {"CrowdWorks": [labelId], "Prolific": [labelId]}}`

***imgId***: the ID of the target image. You can find the image in the folder "images" by adding suffix ".jpg".

***labelId***: the file name of privacy-oriented labels corresponded with the image.

If the key is "CrowdWorks", you can find the file in ./annotations/CrowdWorks/labels/

If the key is "Prolific", you can find the file in ./annotations/Prolific/labels/

If "CrowdWorks" or "Prolific" does not appear, it means no annotator from the specific platform successfully annotated it.



### Annotators Information

Worker information is stord in ./annotations/CrowdWorks/workerinfo/ and ./annotations/Prolific/workerinfo/

`{"age": "48", "gender": "Other", "nationality": "Japan", "workerId": "59422", "bigfives": {"Extraversion": 3, "Agreeableness": 5, "Conscientiousness": 7, "Neuroticism": 6, "Openness to Experience": 9}}`

***age***: age of the annotator.

***gender***: gender of the annotator.

***nationality***: nationality of the annotator

***workerid***: worker ID of the annotator from the platform we recruited him/her.

***bigfives***: Big-Five personality inventory results. 



### Privacy-oriented labels

The file name is structured in the following rules:

`{imgId}_{workerId}_label`

***imgId***: the ID of the target image. You can find the image in the folder "images" by adding suffix ".jpg". 

***workerId***: the ID of worker who created these labels. 

The file is a json file has the following structure for example. 

Please note that some workerId may contain extra `_`.

`{"source": "OpenImages", "workerId": "5070896", "defaultAnnotation": {"Clothing": {"category": "Clothing", "informationType": "", "informationTypeInput": "", "sharing": "", "sharingInput": "", "ifNoPrivacy": true, "informativeness": 4}, "Human body": {"category": "Human body", "informationType": "1", "informationTypeInput": "", "sharing": "3", "sharingInput": "", "ifNoPrivacy": false, "informativeness": "3"}, "Organ (Musical Instrument)": {"category": "Organ (Musical Instrument)", "informationType": "", "informationTypeInput": "", "sharing": "", "sharingInput": "", "ifNoPrivacy": true, "informativeness": 4}, "Piano": {"category": "Piano", "informationType": "", "informationTypeInput": "", "sharing": "", "sharingInput": "", "ifNoPrivacy": true, "informativeness": 4}, "Book": {"category": "Book", "informationType": "3", "informationTypeInput": "", "sharing": "3", "sharingInput": "", "ifNoPrivacy": false, "informativeness": "5"}, "Man": {"category": "Man", "informationType": "1", "informationTypeInput": "", "sharing": "2", "sharingInput": "", "ifNoPrivacy": false, "informativeness": "7"}}, "manualAnnotation": {"0": {"category": "lamp", "bbox": [663, 25, 130, 284], "informationType": "2", "informationTypeInput": "", "sharing": "2", "sharingInput": "", "informativeness": "4"}}}`ß

***source***: the original dataset of the image.

***workerId***: the ID of worker who created these labels. 

***defaultAnnotation***: the annotation corresponded with labels provided by its original dataset. 

Each element in this dictionary has a key-value structure. For example:

`"Book": {"category": "Book", "informationType": "3", "informationTypeInput": "", "sharing": "3", "sharingInput": "", "ifNoPrivacy": false, "informativeness": "5"}`

The key is the category name from it original dataset. 

- ***informationType***: the answer of question "Assuming you want to seek the privacy of the photo owner, what kind of information can this content tell?", provided in numeral format 1 to 5.
- ***informationTypeInput***: if the answer is "others" (value "5"), the information type will be stored here.
- ***sharing***: the answer of question "Assuming you are the photo owner, to what extent would you share this content at most?", provided in numeral format from 1 to 5.
- ***sharingInput***: if the answer is "others" (value "5"), the maximum sharing scope will be stored here.
- ***informativeness***: the rating results of question "How informative do you think about this privacy information for the photo owner?", provided in numeral format from 1 to 7. 
- ***ifNoPrivacy***: If true, it indicate the annotator did not think it is a privacy-threatening content. So, no need to read the above results.

***manualAnnotation***: the annotation provided by annotators manually.  

Each element in this dictionary has a key-value structure. For example:

`{"0": {"category": "lamp", "bbox": [663, 25, 130, 284], "informationType": "2", "informationTypeInput": "", "sharing": "2", "sharingInput": "", "informativeness": "0"}}`

The key is the stringified number of the manual annotation. 

Except the elements that default annotations have, manual annotation has two extra elements.

***category***: category of the annotated object by annotators' own descriptions.

***bbox***: the bounding box surrounded the object , in the format of `[x, y, width, height]`.



#### the map of metric value and metric content.

##### informationType

​			`{1: 'personal identity',` 

​			`2: 'location of shooting',`

​			`3:'personal habits',` 

​			`4: 'social circle',` 

​			`5: 'Others'}`

##### Informativeness

​			`{1: 'extremely uninformative',`

​            `2: 'moderately uninformative',`

​            `3: 'slightly uninformative',`

​            `4: 'neutral',`

​            `5: 'slightly informative',`

​            `6: 'moderately informative',`

​            `7: 'extremely informative'}`

##### Maximum sharing scope if photo owner

​			`{1: 'I won\'t share it',` 

​			`2: 'Family or friend',`

​			`3:'Public',` 

​			`4: 'Broadcast programme',` 

​			`5: 'Others'}`



### Original Labels

The file name is structured in the following rules:

`{imgId}_label`

Each line of the file is in a json dictionary format (you should read each line as json rather than the whole file as json).

Each line is one label on the target image, and at least contains the following key-value element.

***category***: category name from its original dataset. 

***bbox***: the bounding box surrounded the object , in the format of `[x, y, width, height]`.

***width***: width of the target image.

***height***: height of the target image.

***source***: the name of its original dataset.