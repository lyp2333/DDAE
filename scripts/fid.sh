eval_model=encoder_out
eval_mode=step
eval_step=100

readonly dir="$eval_mode"_"$eval_step"
original_path="/home/lyp/Code/DiffuseGAE/Recon/${eval_model}/${dir}/original"
recon_path="/home/lyp/Code/DiffuseGAE/Recon/${eval_model}/${dir}/recon"

for i in $(ls $original_path | grep -E '^5.*.png' | grep -v 50000.png); do
  rm -f ${original_path}/${i}
  rm -f ${recon_path}/${i}
done
#rm -f "$original_path/00006.png"
#rm -f "$recon_path/00006.png"

readonly fid_txt="/home/lyp/Code/DiffuseGAE/Recon/fid.txt"

if [ ! -d $fid_txt ]; then
  touch $fid_txt
fi

fid_score=$(
  fidelity --gpu 0 \
    --fid --input1 $original_path \
    --input2 $recon_path \
    -b 72 \
    --feature-layer-fid 2048 |
    grep frechet_inception_distance | awk '{print $2}'
)
echo "${i}_fid: ${fid_score}" >> $fid_txt
echo  "$i is completed"

