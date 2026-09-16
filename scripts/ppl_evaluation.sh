python main/evaluation/ppl_evaluation.py dataset=ilab_20M/test \
                                dataset.joint.evaluation.ppl_evaluation_model="$1" \
                                dataset.joint.evaluation.ppl_evaluation_mode="$2" \
                                dataset.joint.evaluation.T_encode_reversed=100 \
                                dataset.joint.evaluation.pred_steps=100 \
                                dataset.joint.evaluation.use_xT=False \
