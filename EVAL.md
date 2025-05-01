# Pipeline Evaluation

## Criteria

1. Obfuscate all license plates that are fully visible
2. Obfuscate all business signs containing legible store names
3. Obfuscate all addresses
4. Obfuscate all street signs with street names, and other text identifying locations
5. Obfuscate sensitive or private content (e.g., emails, social media) on electronic screens,
   but not screens in general
6. Obfuscate all personal documents and ID cards
7. Ensure all images have identifiable faces obfuscated, while non-human elements are preserved
8. No obfuscation on irrelevant objects (e.g., tickets, random objects)

In effect, 1. through 6. are about false negatives, and 7. is about false positives.

## Settings

The idea is that the default settings should work for all of these. These settings include

1. Privacy score threshold at 0.5
2. Ignoring objects detected by YOLO with a confidence score <= 0.4
3. The privacy score weightings given by `CLASS_FEATURE_WEIGHtS` in `pipeline.py`.
4. The following LLM prompt for privacy scoring text

> You are determining the privacy sensitivity of the list
> of text detected in a region of an image.

> Region class: {YOLO_CLASSES[int(class_id)]}
> Detected text: {text}

> Output a single number between 0.0 (completely non-sensitive text) and 1.0 (extremely private information text)
> with regards to the class.
> This number should be 1.0 or close to 1.0 if the text identifies a person or place,
> and if the text contains passwords, numerical ids, or financial information.
> Only output the number, nothing else.

Additionally, we are evaluating the pipeline with inpainting rather than Gaussian blurring,
except for applying blurring to faces, since inpainting more consistently obscures relevant information.
Gaussian blurring is the default, however, since the inpainting implementation with lama we're using
requires a GPU.

## Eval data

The evaluation data is the test set held out from training our YOLO model for object segmentation.
We also avoided tuning the various parameters for privacy scoring with reference to this test set,
to avoid cooking the books, so to speak, to make our application appear better than it is.
The train-val-test split is 85-10-5 (1740, 199, and 97 images respectively),
partitioned so as to have a proportional representation of each object class across the sets.
We removed images that were not particular relevant to evaluation,
and added some additional images to the test set
that were useful test cases for what we want the application to do.
In particular, we added images from
[DocXPand-25k](https://arxiv.org/html/2407.20662v1#S3),
which is large and diverse benchmark dataset for identity documents analysis,
and [Su (2023), “Screen Detection YOLOv8”](https://data.mendeley.com/datasets/kp89xh68p2/1).

After removing and adding images, the test set comes out to 103 images.

## Eval ground truth

| Criterion | Count |
| --------- | ----- |
| 1 lic.pl. | 16    |
| 2 bus.sn. | 21    |
| 3 address | 21    |
| 4 st.sn.  | 13    |
| 5 screens | 14    |
| 6 docs    | 23    |
| 7 faces   | 33    |

1. Obfuscate all license plates that are fully visible
2. Obfuscate all business signs containing legible store names
3. Obfuscate all addresses
4. Obfuscate all street signs with street names
5. Obfuscate all electronic screens displaying sensitive or private content (e.g., messages, emails)
6. Obfuscate all personal documents and ID cards
7. Ensure all group photos have identifiable faces obfuscated, while non-human elements are preserved
8. No obfuscation on irrelevant objects (e.g., tickets, random objects)

## Application performance

| Criterion | Count | Accuracy |
| :-------: | :---: | :------: |
| 1 lic.pl. |  15   |  0.9375  |
| 2 bus.sn. |  18   |  0.8571  |
| 3 address |  19   |  0.9048  |
| 4 st.sn.  |   8   |  0.6154  |
| 5 screens |  11   |  0.7857  |
|  6 docs   |  19   |  0.8261  |
|  7 faces  |  28   |  0.8484  |
|   8 FP    |  15   |   N/A    |

## False negatives

The small size of the dataset means there is significant variance, but some things
are clear enough, especially when the images involved are considered.
Some false negatives are clearly because of challenging images. For instance,
the relatively low performance for street signs (especially when compared to
metrics for the YOLO model itself) is largely due to the presence of a black
and white image containing several street signs.
In other cases, false negatives are due to something being amiss with our
privacy scoring. In particular, some of the faces missed by the application
were detected with high confidence by the YOLO model itself, which points
in the direction of our privacy scoring.

One significant kind of false negative reflected in these metrics
is cases where application obscures a photo id save for the picture
of the person's face. This seems to mainly be due to the quirks of our
privacy scoring for photos. To capture this, we counted any photo id as 2 objects,
one for the photo and one for the text of the document.

## False positives

We seem to have successfully controlled the false positive rate
(i.e. unwanted obfuscation) arising from completely haywire object detection.

False positives we do have arise for two reasons:

1. objects get we don't want to obscure that resemble objects we do want to obscure.
   This is a failure of the YOLO model, e.g. mistaking signage for street signage.
2. text that is wrongly scored as sensitive. This is a failure of the LLM and
   our privacy scoring on the basis of the score produced by the LLM in conjunction
   with other factors.

A much larger proportion of the false positives come from 2.
