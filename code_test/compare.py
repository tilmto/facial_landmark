import os
import cv2
import json
import compute_error
import numpy as np
import copy


def compute_error_avg(landmark,ground_truth):
	error_sum = 0
	for i in range(len(landmark)):
		_,under_threshold,_,error,error_array=compute_error.compute_error_all_point(landmark[i],ground_truth[i],0.08,256)
		error_sum += error
	return error_sum/len(landmark)
'''

def get_face_range(ground_truth):
	return np.sqrt((np.max(ground_truth[:,0])-np.min(ground_truth[:,0]))*(np.max(ground_truth[:,1])-np.min(ground_truth[:,1])))

def compute_error_avg(landmark,ground_truth):
	error_sum = 0
	for i in range(len(landmark)):
		_,under_threshold,_,error,error_array=compute_error.compute_error_all_point(landmark[i],ground_truth[i],0.08,256)
		error_sum += error*256/get_face_range(np.array(ground_truth[i]))
	return error_sum/len(landmark)
'''

with open('res152.json','r') as f:
	dict_res152 = json.load(f)

with open('HourGlass_Conv.json','r') as f:
	dict_hg = json.load(f)

with open('HourGlass_Locally_Connected.json','r') as f:
	dict_hglc = json.load(f)

with open('dense.json','r') as f:
	dict_dense = json.load(f)

with open('resplus.json','r') as f:
	dict_resplus = json.load(f)

with open('resext.json','r') as f:
	dict_resext = json.load(f)


landmark_res152 = []
landmark_hg = []
landmark_hglc = []
landmark_dense = []
landmark_resplus = []
landmark_resext = []

landmark_avg = []

ground_truth = []

for fname in dict_res152.keys():
	'''
	if fname == 'IBUG_image_048_5.jpg' or fname == 'IBUG_image_048_4.jpg':
		continue
	'''
	landmark_res152.append(dict_res152[fname][0])
	landmark_hg.append(dict_hg[fname][0])
	landmark_hglc.append(dict_hglc[fname][0])
	landmark_dense.append(dict_dense[fname][0])
	landmark_resplus.append(dict_resplus[fname][0])
	landmark_resext.append(dict_resext[fname][0])
	ground_truth.append(dict_res152[fname][1])


landmark_res152 = np.array(landmark_res152)
landmark_hg = np.array(landmark_hg)
landmark_hglc = np.array(landmark_hglc)
landmark_dense = np.array(landmark_dense)
landmark_resplus = np.array(landmark_resplus)
landmark_resext = np.array(landmark_resext)

landmark_avg = landmark_hg*0.3 + landmark_dense*0.175 + landmark_res152*0.175 + landmark_hglc*0.175 + landmark_resplus*0.175

'''
best_weight = [0.33,0.20,0.19,0.16,0.12]
best_error = compute_error_avg(landmark_avg,ground_truth)
weight = copy.deepcopy(best_weight)

no_update = 0
for i in range(1000):
	a = np.random.randint(5)
	b = np.random.randint(5)
	weight[a] += 0.01
	weight[b] -= 0.01
	landmark_avg = landmark_hg*weight[0] + landmark_dense*weight[1] + landmark_res152*weight[2] + landmark_hglc*weight[3] + landmark_resplus*weight[4]
	error = compute_error_avg(landmark_avg,ground_truth)
	if error<best_error:
		best_weight = copy.deepcopy(weight)
		best_error = error
		no_update = 0
		print('Find new weight with error ',best_error)
	else:
		no_update += 1
	if no_update>=50:
		weight = copy.deepcopy(best_weight)
		print('Reset')
print(best_weight)
'''

print('ResNet error_avg:',compute_error_avg(landmark_res152,ground_truth))
print('Hourglass error_avg:',compute_error_avg(landmark_hg,ground_truth))
print('HourGlass_Locally_Connected error_avg:',compute_error_avg(landmark_hglc,ground_truth))
print('DenseNet error_avg:',compute_error_avg(landmark_dense,ground_truth))
print('Resnet_Plus error_avg:',compute_error_avg(landmark_resplus,ground_truth))
print('Resnet_Extern error_avg:',compute_error_avg(landmark_resext,ground_truth))

print('Integrated model error_avg:',compute_error_avg(landmark_avg,ground_truth))


'''
min_arg = {}
for i in range(106):
	min_arg[i] = [0 for i in range(4)]

for i in range(150):
	_,under_threshold,_,error,error_array1=compute_error.compute_error_all_point(landmark_res152[i],ground_truth[i],0.08,256)
	_,under_threshold,_,error,error_array2=compute_error.compute_error_all_point(landmark_hg[i],ground_truth[i],0.08,256)
	_,under_threshold,_,error,error_array3=compute_error.compute_error_all_point(landmark_hglc[i],ground_truth[i],0.08,256)
	_,under_threshold,_,error,error_array4=compute_error.compute_error_all_point(landmark_dense[i],ground_truth[i],0.08,256)

	for j in range(106):
		arg = np.argmin([error_array1[j],error_array2[j],error_array3[j],error_array4[j]])
		min_arg[j][arg] += 1


error_array_min = np.array([0.0 for i in range(106)])
for i in range(150,337):
	_,under_threshold,_,error,error_array1=compute_error.compute_error_all_point(landmark_res152[i],ground_truth[i],0.08,256)
	_,under_threshold,_,error,error_array2=compute_error.compute_error_all_point(landmark_hg[i],ground_truth[i],0.08,256)
	_,under_threshold,_,error,error_array3=compute_error.compute_error_all_point(landmark_hglc[i],ground_truth[i],0.08,256)
	_,under_threshold,_,error,error_array4=compute_error.compute_error_all_point(landmark_dense[i],ground_truth[i],0.08,256)

	for j in range(106):
		candidate = [error_array1[j],error_array2[j],error_array3[j],error_array4[j]]
		error_array_min[j] += candidate[np.argmax(min_arg[j])]

error_array_min = sum(error_array_min)/(landmark_res152.shape[0]-150)/106

print('Min error_avg:',error_array_min)
'''




'''
def compute_error_avg(valid):
	error_sum = 0
	for error in valid.values():
		error_sum += error
	return error_sum/len(valid)

error = 0
i = 0

valid_fyg = {}
valid_ys = {}

for fname in os.listdir('./valid_img'):
	valid_fyg[fname.split('@')[1]] = float(fname.split('@')[0])

for fname in os.listdir('./valid_img_ys'):
	valid_ys[fname.split('@')[1]] = float(fname.split('@')[0])

print('error_fyg:',compute_error_avg(valid_fyg))

print('error_ys:',compute_error_avg(valid_ys))

valid_min = {}
worse_pic = {}

for fname in valid_fyg.keys():
	if valid_fyg[fname]<valid_ys[fname]:
		valid_min[fname]=valid_fyg[fname]
	else:
		valid_min[fname]=valid_ys[fname]
		worse_pic[fname]=[valid_fyg[fname],valid_ys[fname]]

print('error_min:',compute_error_avg(valid_min))
print('Among ',len(valid_fyg),' pictures, model_ResNet is better on ',len(valid_fyg)-len(worse_pic),' pictures.')

show = False
if show:
	for fname in worse_pic.keys():
		img_fyg = cv2.imread('./valid_img/'+str(worse_pic[fname][0])+'@'+fname)
		img_ys = cv2.imread('./valid_img_ys/'+str(worse_pic[fname][1])+'@'+fname)
		cv2.imshow('ResNet:'+str(worse_pic[fname][0]),img_fyg)
		cv2.imshow('S-Hourglass:'+str(worse_pic[fname][1]),img_ys)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
'''