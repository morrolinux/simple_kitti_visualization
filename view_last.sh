num=$(ls ~/Carla0.9.10-kitti-data-export/_out/training/image_2/|sort|tail -n1|cut -d. -f1)
num=$(printf "%06d" $(($num+$1)))
python 3dbox_to_img.py --file_id $num
