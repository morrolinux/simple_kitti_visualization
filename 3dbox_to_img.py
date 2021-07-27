import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from skimage import io
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--file_id", default='000010')
args = parser.parse_args()

colors = sns.color_palette('Paired', 9 * 2)
names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

file_id = args.file_id


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
    ravel_mode = 'C'
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))

    R = extrinsic_mat[:3, :3]
    T = extrinsic_mat[:3, 3]
    T1 = np.array([0, T[1], 10])   # x (carla -y), y (carla -z), z (carla x),  

    pitch, yaw, roll = rotationMatrixToEulerAngles(R)

    R1 = np.array(eulerAnglesToRotationMatrix((offset, 0, 0)))  # roll (carla pitch), pitch (carla yaw), img_yaw (carla roll)
    
    RT1 = np.column_stack((R1, T1))
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    print("RT1:", RT1)

    P0 = np.matmul(P0, RT1)

    return P0


def draw_labels(ax, labels, P2, pause=0.001):
  # draw image
  # plt.imshow(img)

  for line in labels:
    line = line.split()
    lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line
    h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
    if lab != 'DontCare':
      x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
      y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
      z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
      corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

      # transform the 3d bbox from object coordiante to camera_0 coordinate
      R = np.array([[np.cos(rot), 0, np.sin(rot)],
                    [0, 1, 0],
                    [-np.sin(rot), 0, np.cos(rot)]])
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

  # fig.patch.set_visible(False)
  # plt.savefig('examples/kitti_3dbox_to_img.png', bbox_inches='tight')
  fig.canvas.draw()
  plt.pause(pause)
  ax.lines.clear()
  # plt.close(fig)
  # plt.show()


if __name__ == '__main__':

  # load image
  img = np.array(io.imread(f'examples/kitti/image_2/{file_id}.png'), dtype=np.int32)

  # load labels
  with open(f'examples/kitti/label_2/{file_id}.txt', 'r') as f:
    labels = f.readlines()

  # load intrinsics and extrinsics
  with open(f'examples/kitti/calib/{file_id}.i', 'rb') as f:
    intrinsic_mat = np.load(f)
  with open(f'examples/kitti/calib/{file_id}.e', 'rb') as f:
    extrinsic_mat = np.load(f)

  # load calibration file
  # with open(f'examples/kitti/calib/{file_id}.txt', 'r') as f:
  #   lines = f.readlines()
  #   P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

  fig = plt.figure()
  ax = fig.gca()
  fig.show()
  ax.imshow(img)
  plt.axis('off')
  plt.tight_layout()
  figManager = plt.get_current_fig_manager()
  figManager.window.showMaximized()

  # P2 = calc_P(intrinsic_mat, extrinsic_mat)
  # draw_labels(labels, P2)

  for i in range(100):
    draw_labels(ax, labels, calc_P(intrinsic_mat, extrinsic_mat, offset=i/100))
