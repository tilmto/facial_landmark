import numpy as np
import cv2 as cv
import slice_image
import os

debug = False

def get_ground_truth_frame(fname):
	ground_truth_file = open(fname,'r')
	ground_truth_file.readline()
	ground_truth_frame = []
	for i in range(106):
		line = ground_truth_file.readline()
		if not line:
			break
		ground_truth_frame.append([float(x) for x in line.strip().split()])
	ground_truth_file.close()
	return np.array(ground_truth_frame,np.float32)

def draw_points(img,ground_truth):
	for i in range(106):
		cv.circle(img,(int(ground_truth[i][0]),int(ground_truth[i][1])),1,(0,0,255))
	return img

def get_face_bound(ground_truth_frame):
	return np.min(ground_truth_frame[:,0]),np.max(ground_truth_frame[:,0]),np.min(ground_truth_frame[:,1]),np.max(ground_truth_frame[:,1])

fname_list = []
for root,dirs,files in os.walk('../side_face_extern'):
	for fname in files:
		if fname.endswith('.txt'):
			fname_list.append(os.path.join(root,fname))
fname_list = np.array(fname_list)

for i in range(fname_list.shape[0]):
	ground_truth_frame = get_ground_truth_frame(fname_list[i])
	x_mean = np.mean(ground_truth_frame[:,0])
	y_mean = np.mean(ground_truth_frame[:,1])
	left,right,top,bottom = get_face_bound(ground_truth_frame)

	#print('left:',left,' right:',right,' top:',top,' bottom:',bottom)

	if x_mean-left>right-x_mean:
		left = max(0,left-10)
		right = min(449,2*x_mean-left)
	else:
		right = min(449,right+10)
		left = max(0,2*x_mean-right)

	if y_mean-top>bottom-y_mean:
		top = max(0,top-10)
		bottom = min(449,2*y_mean-top)
	else:
		bottom = min(449,bottom+10)
		top = max(0,2*y_mean-bottom)

	#print('left:',left,' right:',right,' top:',top,' bottom:',bottom)

	with open(fname_list[i][:-4]+'.rect','w') as f:
		if debug:
			img = cv.imread(fname_list[i][:-4])
			img,_,s_f= slice_image.slice_frame(img,int(left),int(top),int(right),int(bottom),(450,450),slicewh=(256,256))
			ground_truth_frame=(ground_truth_frame-np.array([left,top]))*s_f
			img = draw_points(img,ground_truth_frame)
			cv.imwrite('./'+fname_list[i][:-4].split('/')[-1],img)
			input()

		f.write(str(left)+' '+str(top)+' '+str(right)+' '+str(bottom)+'\n')
