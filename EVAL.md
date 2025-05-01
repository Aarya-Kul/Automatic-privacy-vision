# Pipeline Evaluation

## Criteria

1. Obfuscate all license plates that are fully visible
2. Obfuscate all business signs containing legible store names
3. Obfuscate all addresses
4. Obfuscate all street signs with street names
5. Obfuscate all electronic screens displaying sensitive or private content (e.g., messages, emails)
6. Obfuscate all personal documents and ID cards
7. Ensure all group photos have identifiable faces obfuscated, while non-human elements are preserved
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
We added some additional images that were useful test cases for what we want the application to do.
