import os
import shutil
import numpy as np
from scipy.io import loadmat
import yaw

limit = 0.8

debug = False

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
    return np.array(ground_truth_frame,np.float32)

def get_face_range(ground_truth):
	return np.sqrt((np.max(ground_truth[:,0])-np.min(ground_truth[:,0]))*(np.max(ground_truth[:,1])-np.min(ground_truth[:,1])))

def distance(a,b):
	return np.sqrt(np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2))

def get_yaw_68(ground_truth):
	return np.fabs(distance(ground_truth[2],ground_truth[33])-distance(ground_truth[33],ground_truth[14]))/get_face_range(ground_truth)

def get_yaw_106(ground_truth):
	return np.fabs(distance(ground_truth[6],ground_truth[54])-distance(ground_truth[54],ground_truth[26]))/get_face_range(ground_truth)

data_list = []
trash_list = []
for root,dirs,files in os.walk('../training_set'):
	for fname in files:
		if fname.find('IBUG') == -1 and fname.find('ibug') == -1:
			if fname.endswith('.jpg'):
				img_fname = '../300W_LP/'+fname.split('_')[0]+'/'+fname
				cd_fname = img_fname[:-4]+'_pts.mat'
				gt = loadmat(cd_fname)['pts_2d']
				gt2 = get_ground_truth_frame(root+'/'+fname+'.txt')

				if get_yaw_68(gt)>limit:
					data_list.append([img_fname,cd_fname])
				else:
					trash_list.append([img_fname,cd_fname])

data_list = np.array(data_list)
np.random.shuffle(data_list)
valid_size = int(data_list.shape[0]/10)
valid_data_list = data_list[:valid_size]
train_data_list = data_list[valid_size:]

if not debug:
	for train_data in train_data_list:
		shutil.copy(train_data[0],'../conv_train/'+os.path.basename(train_data[0]))
		shutil.copy(train_data[1],'../conv_train/'+os.path.basename(train_data[1]))

	for valid_data in valid_data_list:
		shutil.copy(valid_data[0],'../conv_valid/'+os.path.basename(valid_data[0]))
		shutil.copy(valid_data[1],'../conv_valid/'+os.path.basename(valid_data[1]))

	for trash_data in trash_list:
		shutil.copy(trash_data[0],'../conv_trash/'+os.path.basename(trash_data[0]))
		shutil.copy(trash_data[1],'../conv_trash/'+os.path.basename(trash_data[1]))

	side_face = yaw.get_side_face(limit)
	for i in range(len(side_face)):
		shutil.copy(side_face[i],'../side_face_extern/'+side_face[i].split('/')[-1])
		shutil.copy(side_face[i][:-8]+'.jpg','../side_face_extern/'+side_face[i].split('/')[-1][:-8]+'.jpg')
