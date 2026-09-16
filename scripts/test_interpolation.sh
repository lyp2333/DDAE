#python main/evaluation/interpolation.py dataset=ilab_20M/test \
#                                dataset.joint.data.imgs_root='/home/lyp/Code/DiffuseGAE/main/evaluation/imgs_interpolation/ilab_128' \
#                                dataset.joint.evaluation.num_intp=10 \
#                                dataset.joint.evaluation.T_encode_reversed=100 \
#                                dataset.joint.evaluation.pred_steps=100

python main/evaluation/interpolation.py dataset=fonts/test \
                                dataset.joint.data.imgs_root='/home/lyp/Code/DiffuseGAE/main/evaluation/imgs_interpolation/fonts_128' \
                                dataset.joint.evaluation.num_intp=10 \
                                dataset.joint.evaluation.T_encode_reversed=100 \
                                dataset.joint.evaluation.pred_steps=20