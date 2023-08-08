# Language_Clustering

## Installs

### Install `🤗 Transformers`

Follow the instructions from the Hugging Face website:
https://huggingface.co/docs/transformers/installation

# Download English Wikipedia pages

You can either download the full Wikipedia English data or the partial data.

(we will use the partial data).

## Download Options

### Partial Data

To download all the first 1 billion bytes of English Wikipedia, run the following command:

```
$ mkdir data
$ wget -c http://mattmahoney.net/dc/enwik9.zip -P data
$ unzip data/enwik9.zip -d data
```

### Full Data

To download all the English Wikipedia pages, run the following command:

```
$ wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

# Prepering The Data

## Pre-process Using `my-wikifil.pl`

A raw Wikipedia dump contains a lot of HTML / XML data.
We pre-process it with the wikifil.pl script
(originally developed by Matt Mahoney, we changed it a bit).

```
$ perl my-wikifil.pl data/enwik9 > data/fil9
```

## Split Sentences

The data must be split to sentences. We do it by running the scropt `split-lines.py` that saves the current input files
in the folder `split-output`.

# Generate ***WORD*** Vectors (to `.pkl` file)

To create ***word*** vectors from all words in the text, run the script `LaBSE-try.py` which will generate the word
vectors and will save them as a `.pkl` file.

The object saved to the `.pkl` file will be a dictionary with words as keys, and vectors of type `numpy.ndarray` as
values.

## Dimantion Reduction
