## ArtEmis: Affective Language for Visual Art
A codebase created and maintained by <a href="https://ai.stanford.edu/~optas" target="_blank">Panos Achlioptas</a>.

![representative](https://github.com/optas/artemis/blob/master/doc/images/speaker_productions_teaser.png)


### Introduction
This work is based on the [arXiv tech report](https://arxiv.org/abs/2101.07396) which is __provisionally__ accepted in [CVPR-2021](http://cvpr2021.thecvf.com/), for an <b>Oral</b> presentation. 

### Citation
If you find this work useful in your research, please consider citing:
	
	@article{achlioptas2021artemis,
        title={ArtEmis: Affective Language for Visual Art},
        author={Achlioptas, Panos and Ovsjanikov, Maks and Haydarov, Kilichbek and
                Elhoseiny, Mohamed and Guibas, Leonidas},
        journal = {CoRR},
        volume = {abs/2101.07396},
        year={2021}
    }

### Dataset
To get the most out of this repo, please __download__ the data associated with ArtEmis, by filling this [form](https://forms.gle/7eqiRgb764uTuexd7).

### Installation
This code has been tested with Python 3.6.9, Pytorch 1.3.1, CUDA 10.0 on Ubuntu 16.04.

Assuming some (potentially) virtual environment and __python 3x__ 
```Console
git clone https://github.com/optas/artemis.git
cd artemis
pip install -e .
```
This will install the repo with all its dependencies (listed in setup.py) and will enable you to do things like:
``` 
from artemis.models import xx
```   
(provided you add this artemis repo in your PYTHON-PATH) 
  
### Playing with ArtEmis 

#### Step-1 (important &nbsp; :pushpin:) 

 __Preprocess the provided annotations__ (spell-check, patch, tokenize, make train/val/test splits, etc.).
 ```Console 
    artemis/scripts/preprocess_artemis_data.py
 ```
This script allows you to preprocess ArtEmis according to your needs. The __default__ arguments will do __minimal__ 
preprocessing so the resulting output can be used to _fairly_ compare ArtEmis with other datasets; and, derive most _faithful_ statistics 
about ArtEmis's nature. That is what we used in our __analysis__ and what you should use in "Step-2" below. With this in mind do: 
  ```Console 
    python artemis/scripts/preprocess_artemis_data.py -save-out-dir <ADD_YOURS> -raw-artemis-data-csv <ADD_YOURS>
 ```

If you wish to train __deep-nets__ (speakers, emotion-classifiers etc.) *exactly* as we did it in our paper, then you need to rerun this script
by providing only a single extra optional argument ("__--preprocess-for-deep-nets True__"). This will do more aggressive filtering and you should use its output for
"Steps-3" and "Steps-4" below (please use a new save-out-dir to avoid overwriting).  
  ```Console 
    python artemis/scripts/preprocess_artemis_data.py -save-out-dir <ADD_YOURS> -raw-artemis-data-csv <ADD_YOURS> --preprocess-for-deep-nets True
 ```   
(If you wish to understand the nature of the different hyper-parameters please read the details in the provided _help_ messages of the used argparse.)   

#### Step-2
__Analyze & explore the dataset__. :microscope:

Using the _minimally_ preprocessed version of ArtEmis which includes __all__ (454,684) collected annotation. 
   
   1. This is a great place to __start__ :checkered_flag:. Run this [notebook](artemis/notebooks/analysis/analyzing_artemis.ipynb) to do basic _linguistic_, _emotion_ & _art-oriented_ __analysis__ of the ArtEmis dataset.
   2. Please run this [notebook](artemis/notebooks/analysis/concreteness_subjectivity_sentiment_and_POS.ipynb) to analyze ArtEmis in terms of its: _concreteness_, _subjectivity_, _sentiment_ and _Parts-of-Speech_. Optionally, contrast these values with 
   with other common datasets like COCO.
   3. Please run this [notebook](artemis/notebooks/analysis/extract_emotion_histogram_per_image.ipynb) to extract the _emotion histograms_ (empirical distributions) of each artwork. This in __necessary__ for the Step-3 (1).
   4. Please run this [notebook](artemis/notebooks/analysis/emotion_entropy_per_genre_or_artstyle.ipynb) to analyze the extracted emotion histograms (previous step) per art genre and style.  
 
#### Step-3

__Train and evaluate emotion-centric image & text classifiers__. :hearts:

(Using the preprocessed version of ArtEmis for __deep-nets__ which includes 429,431 annotations. Training on a single GPU from scratch is a matter of _minutes_ for these classifiers!)

   1. Please run this [notebook](artemis/notebooks/deep_nets/emotions/image_to_emotion_classifier.ipynb) to train an __image-to-emotion__ classifier.
   2. Please run this [notebook](artemis/notebooks/deep_nets/emotions/utterance_to_emotion_classifier.ipynb) to train an LSTM-based __utterance-to-emotion__ classifier. Or, this [notebook](artemis/notebooks/deep_nets/emotions/utterance_to_emotion_with_transformer.ipynb) to train a BERT-based one.          
   
    
#### Step-4
__Train & evaluate neural-speakers.__ :bomb:
   
   - To __train__ our customized SAT model on ArtEmis  (__~2 hours__ to train in a single GPU!) do:
```Console 
    python artemis/scripts/train_speaker.py -log-dir <ADD_YOURS> -data-dir <ADD_YOURS> -img-dir <ADD_YOURS>

    log-dir: where to save the output of the training process, models etc.
    data-dir: directory you used as _input_  (termed -save-dir) when you run the preprocess_artemis_data.py
              the directory should contain the ouput of preprocess_artemis_data.csv: e.g., 
               the artemis_preprocessed.csv, the vocabulary.pkl
    img-dir: the top folder containing the WikiArt image dataset in its "standard" format:
                img-dir/art_style/painting-xx.jpg
```
    
   Note. The default optional arguments will create the same vanilla-speaker variant we used in our paper.           
   
  - To __train__ the __emotionally-grounded__ variant of SAT add one parameter in the above call:
```Console 
    python artemis/scripts/train_speaker.py -log-dir <ADD_YOURS> -data-dir <ADD_YOURS> -img-dir <ADD_YOURS>
                                            --use-emo-grounding True
```
   - To __sample__ utterances for a trained speaker:
   ```Console 
    python artemis/scripts/sample_speaker.py -arguments
   ```
   For an explanation of the arguments see the argparse help messages. It worth noting that if you 
   want to sample an emotionally-grounded variant you also need to provide a pretrained image2emotion 
   classifier that will be used to extract _the most likely_ emotion of each image as grounding input to 
   the speaker. See Step-3 (1) for how to train such a net.
   
   - To __evaluate__ the sampled utterances of a speaker (e.g., per BLEU, emotional alignment, methaphors etc.) use this 
   [notebook](artemis/notebooks/deep_nets/evaluate_sampled_captions.ipynb). As bonus you can see the _neural attention_ placed on 
   the different tokens/images.               
                         

### Pretrained Models
   * [Image-To-Emotion classifier (81MB)](https://www.dropbox.com/s/8dfj3b36q15iieo/best_model.pt?dl=0)
   * [LSTM-based Text-To-Emotion classifier (8MB)](https://www.dropbox.com/s/9jcenruxk05lx0k/best_model.pt?dl=0)
   * [SAT-Speaker (434MB)](https://www.dropbox.com/s/tnbfws0m3yi06ge/vanilla_sat_speaker_cvpr21.zip?dl=0) 
   * [SAT-Speaker-with-emotion-grounding (431MB)](https://www.dropbox.com/s/0erh464wag8ods1/emo_grounded_sat_speaker_cvpr21.zip?dl=0)
   + Note: the above speaker links include the sampled captions for the test split. You can use them to evaluate the model without re-sampling it. Please read the also included README.txt.   
   + __Caveats__: ArtEmis is a real-world dataset containing the opinion and sentiment of thousands of people. It is expected thus to contain text with biases, factual inaccuracies, and perhaps foul language. Please use responsibly.
   The provided models are likely to be biased and/or inaccurate in ways reflected in the training data.          
    
### News

- :champagne: &nbsp; ArtEmis has attracted already some noticeable media coverage. E.g., @ [New-Scientist](https://www.newscientist.com/article/2266240-ai-art-critic-can-predict-which-emotions-a-painting-will-evoke), 
[HAI](https://hai.stanford.edu/news/artists-intent-ai-recognizes-emotions-visual-art), 
[MarkTechPost](https://www.marktechpost.com/2021/01/30/stanford-researchers-introduces-artemis-a-dataset-containing-439k-emotion-attributions),
[KCBS-Radio](https://ai.stanford.edu/~optas/data/interviews/artemis/kcbs/SAT-AI-ART_2_2-6-21(disco_mix).mp3), 
[Communications of ACM](https://cacm.acm.org/news/250312-ai-art-critic-can-predict-which-emotions-a-painting-will-evoke/fulltext),
[Synced Review](https://medium.com/@Synced/ai-art-critic-new-dataset-and-models-make-emotional-sense-of-visual-artworks-2289c6c71299),
[École Polytechnique](https://www.polytechnique.edu/fr/content/des-algorithmes-emotifs-face-des-oeuvres-dart)
 
- :telephone_receiver: &nbsp; __important__ More code, will be added on the _1st week of April_. Namely, for the ANP-baseline, the comparisons of ArtEmis with other datasets, please do a git-pull at that time. The update will be _seamless_! During this first months, if you have _ANY_ question feel free to send me an email at __optas@stanford.edu__. 

- :trophy: &nbsp; If you are developing more models with ArtEmis and you want to incorporate them here please talk to me or simply do a pull-request. 
  
  
### MISC
- You can make a _pseudo_ "neural speaker" by copying training-sentences to the test according to __Nearest-Neighbors__ in a pretrained
network feature space by running this 5 min. [notebook](artemis/notebooks/deep_nets/speakers/nearest_neighbor_speaker.ipynb).     



#### License
This code is released under MIT License (see LICENSE file for details).
 _In simple words, if you copy/use parts of this code please __keep the copyright note__ in place._

   