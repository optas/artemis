##
## Bash script to test Artemis-based speakers trained under different configurations.
##
## you can use this to sample several speaker models you might have trained. obviously you need to addapt to YOUR paths

sampling_script=/home/optas/Git_Repos/artemis/artemis/scripts/sample_speaker.py
wiki_img_dir=/home/optas/DATA/Images/Wiki-Art/rescaled_max_size_to_600px_same_aspect_ratio

#sampling hyper-params:
sampling_config_file=/home/optas/Git_Repos/artemis/artemis/data/speaker_sampling_configs/\
mini_hyper_param_ablation.json.txt

# other micro hyper-params
gpu_id=2
n_workers=4

##
## Sampling Speaker trained without emo-grounding in the corresponding test split
##
do_sampling=true


if $do_sampling
then
  for datestamp in 03-16-2021-22-21-45 03-17-2021-02-36-26
  do

    speaker=/home/optas/DATA/OUT/artemis/neural_nets/speakers/default_training/$datestamp/checkpoints/best_model.pt
    speaker_config=/home/optas/DATA/OUT/artemis/neural_nets/speakers/default_training/$datestamp/config.json.txt
    out_file_prefix=/home/optas/DATA/OUT/artemis/neural_nets/speakers/default_training/$datestamp/samples_from_best_model

    for split in test
    do
      echo 'Sampling split:' $split
      echo $out_file_prefix"_""${split}_split.pkl"

      python $sampling_script\
       -speaker-checkpoint $speaker\
       -out-file $out_file_prefix"_""${split}_split.pkl"\
       -speaker-saved-args $speaker_config\
       -img-dir $wiki_img_dir\
       --sampling-config-file $sampling_config_file\
       --split $split\
       --gpu $gpu_id\
       --compute-nll False\
       --n-workers $n_workers
    done
  done
fi


###
### Sampling Speaker with-emo-grounding test split
###

do_sampling=true
img2emo_chekpoint=/home/optas/DATA/OUT/artemis/neural_nets/img_to_emotion/best_model.pt


if $do_sampling
then
  for datestamp in 03-18-2021-00-55-47 03-17-2021-20-32-19
  do

    speaker=/home/optas/DATA/OUT/artemis/neural_nets/speakers/emo_grounding/$datestamp/checkpoints/best_model.pt
    speaker_config=/home/optas/DATA/OUT/artemis/neural_nets/speakers/emo_grounding/$datestamp/config.json.txt
    out_file_prefix=/home/optas/DATA/OUT/artemis/neural_nets/speakers/emo_grounding/$datestamp/samples_from_best_model

    for split in test
    do
      echo 'Sampling split:' $split
      echo $out_file_prefix"_""${split}_split.pkl"

      python $sampling_script\
       -speaker-checkpoint $speaker\
       -out-file $out_file_prefix"_""${split}_split.pkl"\
       -speaker-saved-args $speaker_config\
       -img-dir $wiki_img_dir\
       --sampling-config-file $sampling_config_file\
       --split $split\
       --gpu $gpu_id\
       --compute-nll False\
       --n-workers $n_workers\
       --img2emo-checkpoint $img2emo_chekpoint
    done
  done
fi
