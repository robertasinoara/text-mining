# Text Representations

## Bag of Vectors
Bag of Vectors (BoV) is a text representation based in Vector Space Model. More precisally, this representation use a pre-trained "word embeddings" model to generate an unique vector representation to each document, calculating the arithmetic mean of dataset words's vector representations found in model.

> Tokenizing raw texts:
```
python3 text2tok.py --input input/dataset/ --output input/dataset/tokenized/
```
> Generating a BoV:
```
python3 text2bov.py --n_gram 1 --model models/Google/GoogleVectors_300.txt --input input/dataset/tokenized/ --output output/bov/txt/
```
> Converting a Doc-Term 'cat-pol' to Doc-Term 'cat' and 'pol':
```
python3 bag2bag.py --input output/bov/txt/ --output output/bov/txt/
```
> Converting a Doc-Term matrix to ARFF (Weka file):
```
python3 bag2arff.py --weka weka.jar --input output/bov/txt/ --output output/bov/arff/
```

### Scripts
* [text2tok.py](https://github.com/joao4ntunes/text-mining/blob/master/tools/text2tok.py)
* [text2bov.py](https://github.com/joao4ntunes/text-mining/blob/master/representations/bov/text2bov.py)
* [bag2bag.py](https://github.com/joao4ntunes/text-mining/blob/master/tools/bag2bag.py) *(use only if the classes are combined - e.g.: category_X-polarity_Y)*
* [bag2arff.py](https://github.com/joao4ntunes/text-mining/blob/master/tools/bag2arff.py)


### Observation
All generated files use *TAB* character as a separator.
