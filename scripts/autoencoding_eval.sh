#dataset model type(random/step)
if [[ "$1" == 'fonts_128' ]]; then
  data_dir='fonts_128'
elif [[ "$2" == 'ilab_128' ]]; then
  data_dir='ilab_128'
else
  echo "No such dataset"
fi

if [[ "$2" == 'gsl' ]]; then
  eval_dir="/home/lyp/Code/DiffuseGAE/Recon/${data_dir}/GSL"
elif [ "$2" == 'gae' ]; then
  eval_dir="/home/lyp/Code/DiffuseGAE/Recon/${data_dir}/gae_out/$3_100"
elif [ "$2" == 'encoder_out' ]; then
  eval_dir="/home/lyp/Code/DiffuseGAE/Recon/${data_dir}/encoder_out/$3_100"
else
  echo "No such model"
fi

#python main/evaluation/autoencoding_evaluation.py dataset=ilab_20M/test \
#                                        dataset.joint.data.autoencoding_imgs_root0="${eval_dir}/original" \
#                                        dataset.joint.data.autoencoding_imgs_root1="${eval_dir}/recon" \

python main/evaluation/autoencoding_evaluation.py dataset=fonts_128/test \
                                        dataset.joint.data.autoencoding_imgs_root0="${eval_dir}/original" \
                                        dataset.joint.data.autoencoding_imgs_root1="${eval_dir}/recon" \


