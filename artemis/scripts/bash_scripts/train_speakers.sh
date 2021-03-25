##
## Bash script to train multiple Artemis-based speakers with different configurations.
##
## you can use this to sample several speaker models you might have trained. obviously you need to addapt to YOUR paths


train_script=/home/optas/Git_Repos/artemis/artemis/scripts/train_speaker.py
artemis_prepocessed_data_dir=/home/optas/DATA/OUT/artemis/preprocessed_data/for_neural_nets
wiki_img_dir=/home/optas/DATA/Images/Wiki-Art/rescaled_max_size_to_600px_same_aspect_ratio

#optional hyper-params
n_workers=10
gpu_id=0


##
## Without emo-grounding
##
top_log_dir=/home/optas/DATA/OUT/artemis/neural_nets/speakers/default_training

### Train ArtEmis without emo-grounding and default hyper-parameters.
python $train_script -log-dir $top_log_dir\
 -data-dir $artemis_prepocessed_data_dir\
 -img-dir $wiki_img_dir\
 --num-workers $n_workers --gpu $gpu_id\
 --save-each-epoch False

#### Train ArtEmis without grounding and default hyper-parameters + atn-spatial-img-size=none + more dropout rate.
python $train_script -log-dir $top_log_dir\
 -data-dir $artemis_prepocessed_data_dir\
 -img-dir $wiki_img_dir\
 --num-workers $n_workers --gpu $gpu_id\
 --dropout-rate 0.2\
 --save-each-epoch False


##
## With emo-grounding
##
top_log_dir=/home/optas/DATA/OUT/artemis/neural_nets/speakers/emo_grounding

## Train ArtEmis *with* emotion-grounding and default hyper-parameters.
python $train_script -log-dir $top_log_dir\
 -data-dir $artemis_prepocessed_data_dir\
 -img-dir $wiki_img_dir\
 --num-workers $n_workers --gpu $gpu_id\
 --save-each-epoch False\
 --use-emo-grounding True

## Train ArtEmis *with* emotion-grounding and more droupout
python $train_script -log-dir $top_log_dir\
 -data-dir $artemis_prepocessed_data_dir\
 -img-dir $wiki_img_dir\
 --num-workers $n_workers --gpu $gpu_id\
 --save-each-epoch False\
 --use-emo-grounding True\
 --dropout-rate 0.2
