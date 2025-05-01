Run search_parquet.py for taking local parquet files with urls to images,
searching captions, fetching images, and saving to {search_input}\_images directory.
Script takes as positional arguments the search term and, optionally, the
directory with parquet files.
Default directory is for laion400m database ("./laion400m").
Required kwarg "-n" or "--n_images" to specify how many images to attempt to download.
Pass optional keyword argument '-d' or '--destination' in format '-d
{destination directory}' to specify save directory, otherwise
saves in default location (search_term}\_images). Optional kwarg "-s" or "--section" to specify how much of the dataset to search,
options are "tenth", "quarter", "half", and "full", or a range of indices in format "0: 100".
You can also pass "--no_wandb" to prevent img2dataset from acccessing wandb.
