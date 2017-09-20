# Text Representations

## Bag of Vectors
Bag of Vectors (BoV) is a text representation based in Vector Space Model. More precisally, this representation use a pre-trained "word embeddings" model to generate an unique vector representation to each document, calculating the arithmetic mean of dataset words's vector representations found in model.

> BoV generation:
```sh
$ python3 text2bov.py --model models/Google/GoogleVectors_300.txt --input input/dataset/ --output output/bov/
```
> Convert a Doc-Term 'cat-pol' to Doc-Term 'cat' and 'pol' (use only if the classes are combined - e.g.: category_X-polarity_Y):
```sh
$ python3 bag2bag.py --input output/txt/ --output output/txt/
```
> Convert a Doc-Term matrix to ARFF (Weka file):
```sh
$ python3 bag2arff.py --weka weka.jar --input output/txt/ --output output/arff/
```

### Scripts
* [text2bov.py] - https://github.com/joao4ntunes/text-mining/blob/master/representations/bov/text2bov.py
* [bag2bag.py] - https://github.com/joao4ntunes/text-mining/blob/master/tools/bag2bag.py
* [bag2arff.py] - https://github.com/joao4ntunes/text-mining/blob/master/tools/bag2arff.py


### Observation
All generated files use TAB character as separator.
