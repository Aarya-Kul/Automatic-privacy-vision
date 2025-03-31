#!/usr/bin/env python
# coding: utf-8

# Streaming the dataset is another option, but too slow in my experience.

# https://huggingface.co/datasets/laion/laion400m
# https://github.com/rom1504/img2dataset/issues/449
# https://www.kaggle.com/datasets/romainbeaumont/laion400m
# https://github.com/rom1504/img2dataset
# if streaming, use high performance dns resolver: https://github.com/rom1504/img2dataset#setting-up-a-high-performance-dns-resolver

# how to do text search through hf dataset: https://huggingface.co/docs/dataset-viewer/en/search

from pathlib import Path
import numpy as np

from img2dataset import download

# for my setup, at least, pyarrow needs to be installed from conda-forge rather than pip due to some discrepancy in the versions
# and compatibility with HuggingFace

from datasets import load_dataset
import datasets

rng = np.random.default_rng(seed=1234)

# uncomment below to sign in to HuggingFace
# login can be done with huggingface_hub.login() in Python or huggingface-cli login in CLI
# token = HfFolder.get_token()

# if token is None:
#    try:
#        subprocess.run("huggingface-cli login", shell=True)
#    except Exception as e:
#        print(f"Unable to call 'huggingface-cli login' in shell: {e}.")
# else:
#    try:
#        user_info = whoami()
#        print(f"Logged into HuggingFace as: {user_info['name']}")
#    except Exception as e:
#        print(f"Unable to identify user from HuggingFace token: {e}")
#
# assert HfFolder.get_token() is not None, "No authentication token for HuggingFace found, aborting."


def filter_function(to_search, search_terms) -> bool:
    if not to_search:
        return False
    to_search = to_search.lower()
    terminators = [" ", "\n.", ",", ":", ";"]
    for term in search_terms:
        term = term.lower()
        for t in terminators:
            if f" {term}{t}" in to_search:
                return True
            elif f" {term}s{t}" in to_search:
                return True
    return False


def search_dataset(
    dataset, search_terms, column="caption", start_from=0, up_to=10000
) -> datasets.Dataset:
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    if up_to == "full":
        up_to = dataset.num_rows
    elif up_to == "half":
        up_to = dataset.num_rows // 2
    elif up_to == "quarter":
        up_to = dataset.num_rows // 4
    elif up_to == "tenth":
        up_to = dataset.num_rows // 10
    up_to = min(start_from + up_to, dataset.num_rows)
    print(f"Filtering from row {start_from} to row {up_to} of the dataset")
    return dataset.select([i for i in range(up_to)]).filter(
        lambda row: filter_function(row[column], search_terms)
    )


def get_n(
    dataset: datasets.Dataset, n: int, exclusions=None
) -> tuple[datasets.Dataset, np.ndarray]:
    rand_indices = rng.choice(dataset.num_rows, size=n, replace=False)
    if exclusions:
        assert isinstance(exclusions, np.ndarray), (
            "excluded indices must be passed as np array"
        )
        while np.intersect1d(rand_indices, exclusions).size > 0:
            to_replace = np.intersect1d(rand_indices, exclusions, return_indices=True)[
                1
            ]
            rand_indices[to_replace] = rng.choice(
                dataset.num_rows, size=to_replace.size, replace=False
            )
            while np.unique(rand_indices).size != rand_indices.size:
                to_keep = np.unique(rand_indices, return_index=True)[1]
                mask = np.ones(rand_indices.shape, bool)
                mask[to_keep] = False
                rand_indices[mask] = rng.choice(
                    dataset.num_rows, size=rand_indices[mask].size, replace=False
                )
    return dataset.select(rand_indices), rand_indices


def img_from_dataset(dataset: datasets.Dataset, dest_path: str):
    # first save dataset to new parquet file to hidden version of specified path
    path = Path(dest_path)
    path.mkdir(exist_ok=True)
    (path / "selection").mkdir(exist_ok=True)
    metadata_path = path / "selection/01.parquet"
    dataset.to_parquet(metadata_path)
    output_path = path / "output"
    download(
        processes_count=8,
        thread_count=16,
        url_list=str(metadata_path),
        resize_mode="no",
        output_folder=str(output_path),
        output_format="files",
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        enable_wandb=True,
        number_sample_per_shard=1000,
        distributor="multiprocessing",
    )


if __name__ == "__main__":
    import argparse

    msg = """
    Module for taking local parquet files with urls to images, 
    searching captions, fetching images, and saving to {search_input}_images directory.
    Takes as positional arguments the search term and, optionally, the directory with parquet files. 
    Default directory is for laion400m database ("./laion400m"). 
    Required kwarg "-n" or "--n_images" to specify how many images to attempt to download. 
    Pass optional keyword argument '-d' or '--destination' in format '-d {destination directory}' to specify save directory, otherwise 
    saves in default location (search_term}_images). Optional kwarg "-s" or "--section" to specify how much of the dataset to search,
    options are "tenth", "quarter", "half", and "full", or a range of indices in format "0: 100". 
    You can also pass "--no_wandb" to prevent img2dataset from acccessing wandb.
    """
    parser = argparse.ArgumentParser(description=msg)
    parser.add_argument("search_term")
    parser.add_argument("source_path", nargs="?", default="./laion400m")
    parser.add_argument("-n", "--n_images", required=True)
    parser.add_argument("-s", "--section", default="tenth")
    parser.add_argument("-d", "--dest", default="replace")
    parser.add_argument("--no_wandb", required=False, action="store_false")
    args = parser.parse_args()

    search_term = args.search_term
    path = Path(args.source_path)
    n_images = int(args.n_images)
    dest = args.dest
    if dest == "replace":
        dest = f"./{search_term}_images"
    data_files = [str(file) for file in path.glob("part*.parquet")]
    dataset = load_dataset("parquet", data_files=data_files)
    if ":" in args.section:
        start = int(args.section.split(":")[0])
        stop = int(args.section.split(":")[1])
        search_results = search_dataset(
            dataset["train"], search_term, start_from=start, up_to=stop
        )
    else:
        search_results = search_dataset(
            dataset["train"], search_term, up_to=args.section
        )
    selection = get_n(search_results, n_images)
    img_from_dataset(selection[0], dest)
