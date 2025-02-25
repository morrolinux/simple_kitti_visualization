import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from math import degrees
from skimage import io
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--file_id", default='000010')
parser.add_argument("--batch", action="store_true", default=False)
args = parser.parse_args()

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    yaw=math.atan2(R[1,0], R[0,0])
    pitch=math.atan2(-R[2,0], math.sqrt(R[2,1]**2+R[2,2]**2))
    roll=math.atan2(R[2,1], R[2,2])
    return pitch, yaw, roll

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

# Calculates projection matrix given intrinsics and extrinsic mat
def calc_P(intrinsic_mat, extrinsic_mat, offset=0):
    # print("\ncalc_P offset: ", offset, "\n")
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))

    R = extrinsic_mat[:3, :3]
    T = extrinsic_mat[:3, 3]    # X, Y, Z (CARLA coords, meters)
    # print("T[0] = CARLA X = ", T[0])
    # print("T[1] = CARLA Y = ", T[1])
    # print("T[2] = CARLA Z = ", T[2])

    # T1 = np.array([T[1], T[2], 0])   # x (carla -y), y (carla -z), z (carla x)
    # T1 = np.array([0, T[2], 0])   # x (carla -y), y (carla -z), z (carla x)
    T1 = np.array([0, 0, 0])   # x (carla -y), y (carla -z), z (carla x)

    pitch, yaw, roll = rotationMatrixToEulerAngles(R)

    # degree representation of carla pitch, yaw, roll
    # Pitch and roll have inverted sign 
    pitch_deg, yaw_deg, roll_deg = map(degrees, (-pitch, yaw, -roll))
    # print("\nCARLA pitch = ", pitch_deg, "\nCARLA yaw = ", yaw_deg, "\nCARLA roll = ", roll_deg)

    # R1 = np.array(eulerAnglesToRotationMatrix((pitch, 0, roll)))  # roll (carla pitch), pitch (carla yaw), img_yaw (carla roll)
    # R1 = np.array(eulerAnglesToRotationMatrix((pitch, 0, roll)))  # roll (carla pitch), pitch (carla yaw), img_yaw (carla roll)
    R1 = np.array(eulerAnglesToRotationMatrix((0, 0, 0)))  # roll (carla pitch), pitch (carla yaw), img_yaw (carla roll)
    
    RT1 = np.column_stack((R1, T1))
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    # print("RT1:", RT1)

    P0 = np.matmul(P0, RT1)

    return P0


def write_flat(f, name, arr):
    f.write("{}: {}\n".format(name, ' '.join(
        map(str, arr.flatten('C').squeeze()))))


def draw_labels(ax, labels, P2, R1, pause=0.001, vtc="", itv=""):
  # draw image
  # plt.imshow(img)

  for line in labels:
    line = line.split()
    lab, trunc, occ, alpha, bb_l, bb_t, bb_r, bb_b, h, w, l, x, y, z, rot_y = line
    h, w, l, x, y, z, rot_y = map(float, [h, w, l, x, y, z, rot_y])
    if lab != 'DontCare':
      x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
      y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
      z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
      corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

      # Apply object label rotation 
      R = eulerAnglesToRotationMatrix((0, rot_y, 0))
      R = np.dot(R1, R)

      if pause == 0:
        return
    
      corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

      # transform the 3d bbox from camera_0 coordinate to camera_x image
      corners_3d_hom = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
      corners_img = np.matmul(corners_3d_hom, P2.T)
      corners_img = corners_img[:, :2] / corners_img[:, 2][:, None]


      def line(p1, p2, front=1):
        ax.add_line(Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=colors[names.index(lab) * 2 + front]))


      # draw the upper 4 horizontal lines
      line(corners_img[0], corners_img[1], 0)  # front = 0 for the front lines
      line(corners_img[1], corners_img[2])
      line(corners_img[2], corners_img[3])
      line(corners_img[3], corners_img[0])

      # draw the lower 4 horizontal lines
      line(corners_img[4], corners_img[5], 0)
      line(corners_img[5], corners_img[6])
      line(corners_img[6], corners_img[7])
      line(corners_img[7], corners_img[4])

      # draw the 4 vertical lines
      line(corners_img[4], corners_img[0], 0)
      line(corners_img[5], corners_img[1], 0)
      line(corners_img[6], corners_img[2])
      line(corners_img[7], corners_img[3])

      props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
      ax.text(int(bb_r)/2000, 1-int(bb_b)/1000, lab, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

  # fig.patch.set_visible(False)
  # plt.savefig('examples/kitti_3dbox_to_img.png', bbox_inches='tight')
  fig.canvas.draw()
  plt.pause(pause)
  ax.lines.clear()
  # plt.close(fig)
  # plt.show()


if __name__ == '__main__':

  file_ids = []

  if args.batch:
    with open("examples/kitti/trainval.txt") as f:
      for line in f.readlines():
        file_ids.append(line.strip())
  else:
    file_ids.append(args.file_id)

  for file_id in file_ids:

    print(file_id)

    # load image
    img = np.array(io.imread(f'examples/kitti/image/{file_id}.png'), dtype=np.int32)

    # load labels
    with open(f'examples/kitti/label/{file_id}.txt', 'r') as f:
      labels = f.readlines()

    # load intrinsics and extrinsics
    with open(f'examples/kitti/calib/{file_id}.i', 'rb') as f:
      intrinsic_mat = np.load(f)
    with open(f'examples/kitti/calib/{file_id}.e', 'rb') as f:
      extrinsic_mat = np.load(f)

    # load calibration file
    with open(f'examples/kitti/calib/{file_id}.txt', 'r') as f:
      lines = f.readlines()
      velo_to_cam = lines[5]
      imu_to_velo = lines[6]
      
    P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
    R = np.array(lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)

    # FIX CARLA PROJECTIONS USING R (NO LONGER NEEDED)
    # P2 = calc_P(intrinsic_mat, extrinsic_mat)
    # # transform the 3d bbox from object coordiante to camera_0 coordinate
    # R = eulerAnglesToRotationMatrix((0, 0, 0))
    # R1 = extrinsic_mat[:3, :3]
    # pitch, yaw, roll = rotationMatrixToEulerAngles(R1)
    # R1 = np.array(eulerAnglesToRotationMatrix((-pitch, 0, roll)))  # roll (carla pitch), pitch (carla yaw), img_yaw (carla roll)
    # R = np.dot(R1.T, R)

    if args.batch:
      # All matrices are written on a line with spacing
      with open(f'examples/kitti/calib/{file_id}.txt', 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), np.ravel(P2, order='C'))
        write_flat(f, "R0_rect", R)
        f.write(velo_to_cam)
        f.write(imu_to_velo)
    else:
      fig = plt.figure()
      ax = fig.gca()
      fig.show()
      ax.imshow(img)
      plt.axis('off')
      plt.tight_layout()
      figManager = plt.get_current_fig_manager()
      figManager.window.showMaximized()   
      draw_labels(ax, labels, P2, R, pause=10, vtc=velo_to_cam, itv=imu_to_velo)

    # for i in range(200):
    #   draw_labels(ax, labels, calc_P(intrinsic_mat, extrinsic_mat, offset=i/100), extrinsic_mat)
