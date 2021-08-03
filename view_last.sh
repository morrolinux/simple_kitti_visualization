num=$(ls ~/Carla0.9.10-kitti-data-export/_out/training/image_2/|sort|tail -n1|cut -d. -f1)
if [[ -z $1 ]]; then nth=0; else nth=$1;fi
num=$(printf "%06d" $(echo $num + $nth| bc))
echo $1
echo SHOWING $num
python 3dbox_to_img.py --file_id $num
