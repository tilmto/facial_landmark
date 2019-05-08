import os
import sys
import numpy as np
import cv2 as cv
import shutil
from scipy.io import loadmat

if len(sys.argv)!=1:
	limit = float(sys.argv[1])
else:
	limit = 0.7

debug = True

def draw_points(img,points):
	for i in range(points.shape[0]):
		cv.circle(img,(int(points[i][0]),int(points[i][1])),1,(0,0,255))
	return img

def distance(a,b):
	return np.sqrt(np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2))

def get_face_range(ground_truth):
	return np.sqrt((np.max(ground_truth[:,0])-np.min(ground_truth[:,0]))*(np.max(ground_truth[:,1])-np.min(ground_truth[:,1])))

def get_yaw(ground_truth):
	return np.fabs(distance(ground_truth[2],ground_truth[33])-distance(ground_truth[33],ground_truth[14]))/get_face_range(ground_truth)

def get_side_face(limit):
	side_face = []
	for root,dirs,files in os.walk('../300W_LP'):
		for fname in files:
			if fname.endswith('.mat'):
				gt = loadmat(os.path.join(root,fname))['pts_2d']
				yaw_metric = get_yaw(gt)
				if yaw_metric > limit:
					side_face.append(os.path.join(root,fname))
					'''
					if debug:
						img = cv.imread(root+'/'+fname[:-8]+'.jpg')
						img = draw_points(img,gt)
					'''
	print('limit:',limit,' number:',len(side_face))
	return side_face


if __name__ == '__main__':	
	side_face = get_side_face(limit)