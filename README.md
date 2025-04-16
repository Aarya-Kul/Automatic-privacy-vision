# Data

## object classes

At present, the mapping from object classes to numerical labels is

```json
{
  "address": 0,
  "advertisement": 1,
  "business_sign": 2,
  "electronicscreens": 3,
  "face": 4,
  "legible_text": 5,
  "license_plate": 6,
  "personal_document": 7,
  "photo": 8,
  "street_name": 9
}
```

## Data sources

Image were pulled from DIPA dataset and, selectively, from the Laion 400m dataset.
About 400 images of house addressed were drawn from the
[Stanford Street View House Numbers dataset] (<http://ufldl.stanford.edu/housenumbers/>).
Images were annotated for segmentation with [Roboflow] (<https://app.roboflow.com>).

## DIPA dataset

To download the DIPA dataset, run the script
seteup.sh in data/dipa.
