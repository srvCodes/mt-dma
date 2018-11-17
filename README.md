# Multi-Task Deep Morphological Analyzer

This repo contains the code for our paper entitled **Multi Task Deep Morphological Analyzer : Context Aware Neural Joint Morphological Tagging and Lemma Prediction**. The Web API service is accessible [here](http://35.154.251.44/).

A sample analysis:

![sample](https://github.com/Saurav0074/mt-dma/blob/master/hindi/images/sample.png)

## Experiements

Both the directories follow the organization:

1. [preProcessing](https://github.com/Saurav0074/mt-dma/tree/master/hindi/preProcessing) contains the code for dataset parsing. Datasets can be downloaded from the website of [Universal Dependencies](http://universaldependencies.org/).

2. [dataInfo](https://github.com/Saurav0074/mt-dma/tree/master/hindi/dataInfo) contains details on data set statistics.

3. [Models](https://github.com/Saurav0074/mt-dma/tree/master/hindi/models) for all experiments:
  - [multiTask_with_context4.py](https://github.com/Saurav0074/mt-dma/blob/master/hindi/models/multiTask_with_context4.py) hosts the fully BiLSTM model for a CW of 4 words.
  - [multiTask_with_attention.py](https://github.com/Saurav0074/mt-dma/blob/master/hindi/models/multiTask_with_attention.py) hosts the character CNN-RNN based MT-DMA model, as reported in the paper.
  - [onlyFeatures.py](https://github.com/Saurav0074/mt-dma/blob/master/hindi/models/onlyFeatures.py) and [onlyRoots.py](https://github.com/Saurav0074/mt-dma/blob/master/hindi/models/onlyRoots.py) contain the codes for individual learning.
  
4. [Code](https://github.com/Saurav0074/mt-dma/tree/master/hindi/featureOptimization)
 for MOO based GA feature selection.
 
5. Code for post processing, visualization, BLEU, Levenshtein and word accuracy calculation can be found in [postProcessingAndVisualization](https://github.com/Saurav0074/mt-dma/tree/master/hindi/postProcessingAndVisualization).

5. [Outputs](https://github.com/Saurav0074/mt-dma/tree/master/hindi/outputs) on the HDTB and UDTB datasets.

6. Outputs for [t-SNE plots](https://github.com/Saurav0074/mt-dma/tree/master/hindi/tsnePlots), [GA graphs](https://github.com/Saurav0074/mt-dma/tree/master/hindi/gaGraphs), and [Precision-Recall curves](https://github.com/Saurav0074/mt-dma/tree/master/hindi/prCurves).


