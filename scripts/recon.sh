eval_model=$1
eval_mode=$2
eval_step=$3
python main/evaluation/generate_recons.py dataset=ilab_20M/test \
                                dataset.joint.evaluation.recon_model=${eval_model} \
                                dataset.joint.evaluation.recon_mode=${eval_mode}  \
                                dataset.joint.evaluation.pred_steps=${eval_step}
