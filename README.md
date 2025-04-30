# Automatic Image Processing for Privacy

## Data

### object classes

At present, the mapping from object classes to numerical labels is

```json
{
  "address": 0,
  "business_sign": 1,
  "electronicscreens": 2,
  "face": 3,
  "license_plate": 4,
  "personal_document": 5,
  "photo": 6,
  "street_name": 7
}
```

Note that the mapping used to be

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

### Data sources

Image were pulled from DIPA dataset and, selectively, from the Laion 400m dataset.
About 400 images of house addressed were drawn from the
[Stanford Street View House Numbers dataset] (<http://ufldl.stanford.edu/housenumbers/>).
Images were annotated for segmentation with [Roboflow] (<https://app.roboflow.com>).

### DIPA dataset

To download the DIPA dataset, run the script
seteup.sh in data/dipa.

## LLM setup

### Install Ollama (you might need to install homebrew)

On macOS run

`brew install ollama`

On Linux, install with

`curl -fsSL https://ollama.com/install.sh | sh`

### (If needed) Fix PATH

```bash
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

### Start Ollama server

`ollama serve`

(If you want to shutdown Ollama server, run `sudo systemctl stop ollama`.)

### Download llama3 (only need to do this once, or if you want to manually use model)

`ollama run llama3`
