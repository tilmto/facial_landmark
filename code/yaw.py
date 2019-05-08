import os
import sys
import numpy as np
import cv2 as cv
import shutil

if len(sys.argv)!=1:
	limit = float(sys.argv[1])
else:
	limit = 0.3

debug = True

def draw_points(img,points):
    for i in range(106):
        cv.circle(img,(int(points[i][0]),int(points[i][1])),1,(0,0,255))
        #cv.putText(img, str(i), (int(points[i][0]),int(points[i][1])),cv.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),1) 
    cv.imshow('facial_landmark',img)
    cv.waitKey(500)

def get_ground_truth_frame(fname):
	ground_truth_file=open(fname,'r')
	ground_truth_file.readline()
	ground_truth_frame=[]
	for i in range(106):
		line=ground_truth_file.readline()
		if not line:
			break
		ground_truth_frame.append([float(x) for x in line.strip().split()])
	ground_truth_file.close()
	return np.array(ground_truth_frame)

def get_face_range(ground_truth):
	return np.sqrt((np.max(ground_truth[:,0])-np.min(ground_truth[:,0]))*(np.max(ground_truth[:,1])-np.min(ground_truth[:,1])))

def distance(a,b):
	return np.sqrt(np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2))

def get_yaw(ground_truth):
	return np.fabs(distance(ground_truth[6],ground_truth[54])-distance(ground_truth[54],ground_truth[26]))/get_face_range(ground_truth)

def cat_face(limit):
	side_face = []
	front_face = []
	valid_face = []
	for root,dirs,files in os.walk('../training_set'):
		for fname in files:
			if fname.endswith('.txt'):
				if fname.find('IBUG')==-1 and fname.find('ibug')==-1:
					gt = get_ground_truth_frame(root+'/'+fname)
					yaw_metric = get_yaw(gt)
					if yaw_metric > limit:
						side_face.append(fname.strip('.txt'))
						'''
						if debug:
							img = cv.imread(root+'/'+fname.strip('.txt'))
							draw_points(img,gt)
						'''
					else:
						front_face.append(fname.strip('.txt'))
				else:
					valid_face.append(fname.strip('.txt'))

	print('limit:',limit,' number:',len(side_face))

	return side_face,front_face,valid_face

if __name__ == '__main__':
	slide_face,front_face,valid_face = cat_face(limit)
	
	if not debug:
		root = '../training_set/'
		side_face_dir = '../side_face/'
		front_face_dir = '../front_face/'

		for i in range(len(side_face)):
			shutil.copy(root+side_face[i],side_face_dir+side_face[i])
			shutil.copy(root+side_face[i]+'.txt',side_face_dir+side_face[i]+'.txt')
			shutil.copy(root+side_face[i]+'.rect',side_face_dir+side_face[i]+'.rect')

		for i in range(len(front_face)):
			shutil.copy(root+front_face[i],front_face_dir+front_face[i])
			shutil.copy(root+front_face[i]+'.txt',front_face_dir+front_face[i]+'.txt')
			shutil.copy(root+front_face[i]+'.rect',front_face_dir+front_face[i]+'.rect')

		for i in range(len(valid_face)):
			shutil.copy(root+valid_face[i],side_face_dir+valid_face[i])
			shutil.copy(root+valid_face[i]+'.txt',side_face_dir+valid_face[i]+'.txt')
			shutil.copy(root+valid_face[i]+'.rect',side_face_dir+valid_face[i]+'.rect')

			shutil.copy(root+valid_face[i],front_face_dir+valid_face[i])
			shutil.copy(root+valid_face[i]+'.txt',front_face_dir+valid_face[i]+'.txt')
			shutil.copy(root+valid_face[i]+'.rect',front_face_dir+valid_face[i]+'.rect')



