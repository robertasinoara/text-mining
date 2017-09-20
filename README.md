# Text Representations

## Bag of Vectors
Bag of Vectors (BoV) is a text representation based in Vector Space Model. More precisally, this representation use a pre-trained "word embeddings" model to generate an unique vector representation to each document, calculating the arithmetic mean of dataset words's vector representations found in model.

> Generating a BoV:
```
python3 text2bov.py --model models/Google/GoogleVectors_300.txt --input input/dataset/ --output output/bov/
```
> Converting a Doc-Term 'cat-pol' to Doc-Term 'cat' and 'pol':
```
python3 bag2bag.py --input output/txt/ --output output/txt/
```
> Converting a Doc-Term matrix to ARFF (Weka file):
```
python3 bag2arff.py --weka weka.jar --input output/txt/ --output output/arff/
```

### Scripts
* [text2bov.py](https://github.com/joao4ntunes/text-mining/blob/master/representations/bov/text2bov.py)
* [bag2bag.py](https://github.com/joao4ntunes/text-mining/blob/master/tools/bag2bag.py) *(use only if the classes are combined - e.g.: category_X-polarity_Y)*
* [bag2arff.py](https://github.com/joao4ntunes/text-mining/blob/master/tools/bag2arff.py)


### Observation
All generated files use *TAB* character as a separator.
