import os
import numpy as np
import cv2 as cv
import slice_image
from scipy.io import loadmat
import copy
import random
from scipy import misc
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

BATCH_SIZE=20
root_fold=''
ICME_Root =  '../training_set/'
debug = False

debug_flip = False
debug_big_small_face = False

def walk_dir(dirname):
    for root,dirs,files in os.walk(dirname):
        for f in files:
            yield [root, f]

walk_obj_ICME=walk_dir(ICME_Root)
already_load_img_dict={}
total_load_nums=0
fname_list=[]
fname_list_temp=[]
test_id={}
test_img=[]
test_tensor=[]
test_labels=[]
END_OF_ONE_EPOCH=False

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

def rot(image, coord_array, angle):
    im_rot = rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    for i in range(coord_array.shape[0]):
        xy=coord_array[i]
        org = xy-org_center
        a = np.deg2rad(angle)
        new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),-org[0]*np.sin(a) + org[1]*np.cos(a)])
        coord_array[i]=new+rot_center
    return im_rot, coord_array

########sketch stylize########
def dodgeV2(image, mask,scale=150):
    return cv.divide(image, 255-mask,scale=scale)


def burnV2(image, mask):
    return 255 - cv.divide(255-image, 255-mask, scale=256)

def sketch(img_rgb,ksize=(31,31),scale=150):
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    img = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

    if debug:
        cv.imshow("pencil sketch", img)
        cv.waitKey(0)

    img_gray_inv = 255 - img_gray
    img_blur = cv.GaussianBlur(img_gray_inv, ksize, sigmaX=60, sigmaY=60)

    img_blend = dodgeV2(img_gray, img_blur, scale)
    #final_img = cv.cvtColor(img_blend,cv.COLOR_GRAY2RGB)
    #if debug:
    # cv.imshow("pencil sketch", img_blend)
    # cv.waitKey(1000)
    return np.expand_dims(img_blend,axis=-1)
########sketch stylize########

########light stylize########
def light(img_rgb,ksize=(31,31),scale=150):
  img_ske = sketch(img_rgb,ksize,scale)
  img_light = img_ske*0.6 + 0.6*np.array([39,192,249])
  return img_light
########light stylize########

########random cover########
def draw_points(img,points):
    for i in range(106):
        cv.circle(img,(int(points[i][0]),int(points[i][1])),1,(0,0,255))
        #cv.putText(img, str(i), (int(points[i][0]),int(points[i][1])),cv.FONT_HERSHEY_SIMPLEX ,1,(0,0,255),1) 
    cv.imshow('facial_landmark',img)
    cv.waitKey(0)

def random_cover(image,coords,cwidth,cheight):
    outliers = coords#[0:33]
    center = np.mean(outliers,axis=0)
    center = center.astype(np.int)
    coord1 = np.array(coords[np.random.randint(len(outliers))],int)
    coord2 = np.array(coords[np.random.randint(len(coords))],int)
    tcoord = np.concatenate(([coord1],[coord2]))
    orent = center - coord1
    #if debug:
    # tcoord = np.concatenate(([coord1],[coord2]))
    # print(tcoord.shape)
    # print('source point:',coord1)
    # print('dest point:',coord2)
    im_width = image.shape[1]
    im_height = image.shape[0]
    width = max(min(cwidth,int(coord1[0]/2),int((im_width-coord1[0])/2),coord2[0],im_width-coord2[0]),0)
    height = max(min(cheight,int(coord1[1]/2),int((im_height-coord1[1])/2),coord2[1],im_width-coord2[1]),0)
    #if debug:
    # print('cover width:',width,',  height:',height)
    # print('cover region:',coord2[1]-height,':',coord2[1]+height,coord2[0]-width,':',coord2[0]+width)
    image1 = image
    if orent[0]>0 and orent[1]>0:
        cover = image[coord1[1]-2*height:coord1[1],coord1[0]-2*width:coord1[0]]
    elif orent[0]<0 and orent[1]>0:
        cover = image[coord1[1]:coord1[1]+2*height,coord1[0]-2*width:coord1[0]]
    elif orent[0]>0 and orent[1]<0:
        cover = image[coord1[1]-2*height:coord1[1],coord1[0]:coord1[0]+2*width]
    elif orent[0]<0 and orent[1]<0:
        cover = image[coord1[1]:coord1[1]+2*height,coord1[0]:coord1[0]+2*width]
    else:
        return image
    #print(image1[coord2[1]-height:coord2[1]+height,coord2[0]-width:coord2[0]+width].shape)
    #print(cover.shape)
    image1[coord2[1]-height:coord2[1]+height,coord2[0]-width:coord2[0]+width] = cover
    #if debug:
    # draw_points(image1,tcoord)
    # cv.imshow('covered_image',image1)
    # cv.waitKey(1000)
    return image1
#########random cover###########

#########random illumination##########
def rotate_bound_bg(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    color = (37,37,37)
    return cv.warpAffine(image, M, (nW, nH), borderValue=color)

#dst = light * light_weight + face * face_wegiht + gamma=0;
def random_lighten(face,lpath='./'):
    type = np.random.randint(2)
    #type = 0
    if type==0:
        seed = np.random.randint(1,4)
        light = cv.imread(lpath+'light_point_'+str(seed)+'.jpg')
        light_gray = cv.cvtColor(light,cv.COLOR_RGB2GRAY)
        light = cv.cvtColor(light_gray,cv.COLOR_GRAY2RGB)
        border =np.random.randint(1,500)
        top = np.random.randint(0,border)
        bottom = np.random.randint(0,border)
        left = np.random.randint(0,border)
        right = np.random.randint(0,border)
        light = cv.copyMakeBorder(light, top , bottom, left, right, cv.BORDER_REPLICATE)
        shape = np.array(light.shape)
        center = shape/2
        x = np.random.randint(shape[1])
        y = np.random.randint(shape[0])
        if x<center[1] and y<center[0]:
            src1 = cv.resize(light[y:shape[0],x:shape[1]],(face.shape[1],face.shape[0]))
        elif x<center[1] and y>center[0]:
            src1 = cv.resize(light[0:y,x:shape[1]],(face.shape[1],face.shape[0]))
        elif x>center[1] and y<center[0]:
            src1 = cv.resize(light[y:shape[0],0:x],(face.shape[1],face.shape[0]))
        elif x>center[1] and y>center[0]:
            src1 = cv.resize(light[0:y,0:x],(face.shape[1],face.shape[0]))
            #print(center[1],x,center[0],y)
        else:
            return face
        src2 = face
        if debug:
            cv.imshow('light image',src1)
            cv.waitKey(10)
            print(src1.shape)
            print(src2.shape)
        dst = cv.addWeighted(src1,0.3*(random.random()+1.0),src2,0.5*(random.random()+1.0),0)
        if debug:
            cv.imshow('lighten image',dst)
            cv.waitKey(10)
        return dst
    if type==1:
        seed = np.random.randint(1,4)
        light = cv.imread(lpath+'light_straight_'+str(seed)+'.jpg')
        #print(seed)
        #print(light.shape)
        light_gray = cv.cvtColor(light,cv.COLOR_RGB2GRAY)
        light = cv.cvtColor(light_gray,cv.COLOR_GRAY2RGB)
        angle = np.random.randint(360)
        light = rotate_bound_bg(light, angle)
        src1 = cv.resize(light,(face.shape[1],face.shape[0]))
        src2 = face
        if debug:
            cv.imshow('light image',src1)
            cv.waitKey(10)
            print(src1.shape)
            print(src2.shape)
        dst = cv.addWeighted(src1,0.5,src2,0.8,0)
        if debug:
            cv.imshow('lighten image',dst)
            cv.waitKey(10)
        return dst

##########histogram_norm###########
def histogram_correction(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    return img_output

def clahe_correction(img):
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0]  = clahe.apply(img_yuv[:,:,0])
    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    return img_output

def gamma_correction(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv.LUT(image, table)

def flip_img(img,ground_truth_frame):
    img_flip = cv.flip(img,1)
    gt = copy.deepcopy(ground_truth_frame)
    gt[:,0] = img.shape[1]-gt[:,0]

    gt = gt.tolist()
    gt_flip = []
    for i in range(106):
        gt_flip.append(gt[flip_map(i)])

    if debug_flip:    
        draw_points(img,ground_truth_frame)
        cv.imshow("origin_img",img)
        cv.waitKey(0)
 
        draw_points(img_flip,np.array(gt_flip))
        cv.imshow("flipped_img",img_flip)
        cv.waitKey(0)
        cv.destroyAllWindows()  

    return img_flip,np.array(gt_flip,np.float32)


def flip_map(x):
    if x<33:
        return 32-x
    elif 33<=x<38 or 42<=x<47:
        return 79-x
    elif 38<=x<42 or 47<=x<51:
        return 88-x
    elif 51<=x<55:
        return x
    elif 55<=x<66:
        return 120-x
    elif 66<=x<71 or 75<=x<80:
        return 145-x
    elif 71<=x<74 or 80<=x<83:
        return 153-x
    elif x==74 or x==83:
        return 157-x
    elif 84<=x<91:
        return 174-x
    elif 91<=x<96:
        return 186-x
    elif 96<=x<101:
        return 196-x
    elif 101<=x<104:
        return 204-x
    elif x==104 or x==105:
        return 209-x

def compress_x(img,ground_truth_frame):
    fx = (np.random.randint(6,10)+np.random.rand())/10
    img_compress = cv.resize(img,(0,0),fx=fx,fy=1,interpolation=cv.INTER_LINEAR)
    ground_truth_frame = ground_truth_frame*np.array([fx,1])
    
    shape_pad1 = int((img.shape[1]-img_compress.shape[1])/2)
    shape_pad2 = 256-img_compress.shape[1]-shape_pad_x1

    img_padding = cv.copyMakeBorder(img_compress,0,0,int(shape_pad1),int(shape_pad2),cv.BORDER_CONSTANT,0)
    ground_truth_frame = ground_truth_frame+np.array([shape_pad1,0])

    return img_padding,ground_truth_frame

def compress_y(img,ground_truth_frame):
    fy = (np.random.randint(6,10)+np.random.rand())/10
    img_compress = cv.resize(img,(0,0),fx=1,fy=fy,interpolation=cv.INTER_LINEAR)
    ground_truth_frame = ground_truth_frame*np.array([1,fy])
    
    shape_pad1 = int((img.shape[0]-img_compress.shape[0])/2)
    shape_pad2 = 256-img_compress.shape[0]-shape_pad_y1

    img_padding = cv.copyMakeBorder(img_compress,int(shape_pad1),int(shape_pad2),0,0,cv.BORDER_CONSTANT,0)
    ground_truth_frame = ground_truth_frame+np.array([0,shape_pad1])

    return img_padding,ground_truth_frame

def compress(img,ground_truth_frame):
    fx = (np.random.randint(6,10)+np.random.rand())/10
    fy = (np.random.randint(6,10)+np.random.rand())/10
    img_compress = cv.resize(img,(0,0),fx=fx,fy=fy,interpolation=cv.INTER_LINEAR)
    ground_truth_frame = ground_truth_frame*np.array([fx,fy])
    
    shape_pad_x1 = int((img.shape[1]-img_compress.shape[1])/2)
    shape_pad_x2 = 256-img_compress.shape[1]-shape_pad_x1

    shape_pad_y1 = int((img.shape[0]-img_compress.shape[0])/2)
    shape_pad_y2 = 256-img_compress.shape[0]-shape_pad_y1

    img_padding = cv.copyMakeBorder(img_compress,int(shape_pad_y1),int(shape_pad_y2),int(shape_pad_x1),int(shape_pad_x2),cv.BORDER_CONSTANT,0)
    ground_truth_frame = ground_truth_frame+np.array([shape_pad_x1,shape_pad_y1])

    return img_padding,ground_truth_frame


def compress_trans(img,ground_truth_frame):
    fx = (np.random.randint(6,10)+np.random.rand())/10
    fy = (np.random.randint(6,10)+np.random.rand())/10
    img_compress = cv.resize(img,(0,0),fx=fx,fy=fy,interpolation=cv.INTER_LINEAR)
    ground_truth_frame = ground_truth_frame*np.array([fx,fy])
    
    shape_pad_x1 = int((img.shape[1]-img_compress.shape[1])*np.random.rand())
    shape_pad_x2 = 256-img_compress.shape[1]-shape_pad_x1

    shape_pad_y1 = int((img.shape[0]-img_compress.shape[0])*np.random.rand())
    shape_pad_y2 = 256-img_compress.shape[0]-shape_pad_y1

    img_padding = cv.copyMakeBorder(img_compress,int(shape_pad_y1),int(shape_pad_y2),int(shape_pad_x1),int(shape_pad_x2),cv.BORDER_CONSTANT,0)
    ground_truth_frame = ground_truth_frame+np.array([shape_pad_x1,shape_pad_y1])

    return img_padding,ground_truth_frame 


def ladder(img,ground_truth_frame):
    img_ladder = []
    img_ladder.append(img[0])
    rate = (np.random.randint(5,10)+np.random.rand())/10

    for i in range(1,256):
        size = int(256-i*(1-rate))
        img_temp = cv.resize(img,(size,256),interpolation=cv.INTER_LINEAR)
        pad_left = int((256-size)/2)
        pad_right = 256-size-pad_left
        img_temp = cv.copyMakeBorder(img_temp,0,0,pad_left,pad_right,cv.BORDER_CONSTANT,0)
        img_ladder.append(img_temp[i])

    for i in range(106):
        sx = ground_truth_frame[i][1]*(rate-1)/256+1
        ground_truth_frame[i][0] = sx*ground_truth_frame[i][0]+ground_truth_frame[i][1]*(1-rate)/2
    
    img_ladder = np.array(img_ladder)
    #draw_points(img_ladder,ground_truth_frame)
    
    return img_ladder,ground_truth_frame


def big_small_face(img,ground_truth_frame):
	if debug_big_small_face:
		draw_points(copy.deepcopy(img),ground_truth_frame)

	split_x = int(ground_truth_frame[54][0])
	img = copy.deepcopy(img)
	left_face = img[:,:split_x]
	right_face = img[:,split_x:]

	min_sx = 5

	rand_seed = np.random.rand()
	
	if rand_seed<0.33:
		sx = (np.random.randint(min_sx,10)+np.random.rand())/10
		left_face = cv.resize(left_face,(0,0),fx=sx,fy=1,interpolation=cv.INTER_LINEAR)
		new_width = left_face.shape[1]
		left_face = cv.copyMakeBorder(left_face,0,0,split_x-new_width,0,cv.BORDER_CONSTANT,0)
		img[:,:split_x] = left_face

		for i in range(106):
			if ground_truth_frame[i][0]<split_x:
				ground_truth_frame[i][0] = ground_truth_frame[i][0]*sx+split_x-new_width

	elif rand_seed<0.66:
		sx = (np.random.randint(min_sx,10)+np.random.rand())/10
		right_face = cv.resize(right_face,(0,0),fx=sx,fy=1,interpolation=cv.INTER_LINEAR)
		right_face = cv.copyMakeBorder(right_face,0,0,0,img.shape[1]-split_x-right_face.shape[1],cv.BORDER_CONSTANT,0)
		img[:,split_x:] = right_face

		for i in range(106):
			if ground_truth_frame[i][0]>split_x:
				ground_truth_frame[i][0] = split_x+(ground_truth_frame[i][0]-split_x)*sx

	else:
		sx1 = (np.random.randint(min_sx,10)+np.random.rand())/10
		sx2 = (np.random.randint(min_sx,10)+np.random.rand())/10

		left_face = cv.resize(left_face,(0,0),fx=sx1,fy=1,interpolation=cv.INTER_LINEAR)
		new_width = left_face.shape[1]
		left_face = cv.copyMakeBorder(left_face,0,0,split_x-new_width,0,cv.BORDER_CONSTANT,0)
		img[:,:split_x] = left_face

		right_face = cv.resize(right_face,(0,0),fx=sx2,fy=1,interpolation=cv.INTER_LINEAR)
		right_face = cv.copyMakeBorder(right_face,0,0,0,img.shape[1]-split_x-right_face.shape[1],cv.BORDER_CONSTANT,0)
		img[:,split_x:] = right_face

		for i in range(106):
			if ground_truth_frame[i][0]<split_x:
				ground_truth_frame[i][0] = ground_truth_frame[i][0]*sx1+split_x-new_width
			else:
				ground_truth_frame[i][0] = split_x+(ground_truth_frame[i][0]-split_x)*sx2

	if debug_big_small_face:
		draw_points(img,ground_truth_frame)

	return img,ground_truth_frame


def thin_face(img,ground_truth_frame):
	split_y = int(ground_truth_frame[60][1])

	down_face = img[split_y:,:]

	sx = (np.random.randint(8,10)+np.random.rand())/10

	down_face = cv.resize(down_face,(0,0),fx=sx,fy=1,interpolation=cv.INTER_LINEAR)
	pad_left = int((256-down_face.shape[1])/2)
	pad_right = 256-down_face.shape[1]-pad_left
	down_face = cv.copyMakeBorder(down_face,0,0,pad_left,pad_right,cv.BORDER_CONSTANT,0)

	img[split_y:,:] = down_face
	#img = cv.GaussianBlur(img,(5,5),sigmaX=60,sigmaY=60)

	for i in range(106):
			if ground_truth_frame[i][1]>split_y:
				ground_truth_frame[i][0] = pad_left+ground_truth_frame[i][0]*sx

	draw_points(img,ground_truth_frame)

	return img,ground_truth_frame

def get_face_bound(ground_truth_frame):
    return np.min(ground_truth_frame[:,0]),np.max(ground_truth_frame[:,0]),np.min(ground_truth_frame[:,1]),np.max(ground_truth_frame[:,1])

def rand_bound(left,top,right,bottom,ground_truth_frame):
    face_left,face_right,face_top,face_bottom = get_face_bound(ground_truth_frame)

    if face_left>left:
        left = left + (face_left-left)*np.random.rand()
    if face_right<right:
        right = right - (right-face_right)*np.random.rand()
    if face_top>top:
        top = top + (face_top-top)*np.random.rand()
    if face_bottom<bottom:
        bottom = bottom - (bottom-face_bottom)*np.random.rand()

    return int(left),int(top),int(right),int(bottom)


def elastic_transform(img,alpha,sigma):
	shape = img.shape

	dx = gaussian_filter(np.random.rand(shape[0],shape[1])*2-1,sigma)*alpha
	dx = np.expand_dims(dx,axis=2)
	dx = np.concatenate((dx,dx,dx),axis=2)

	dy = gaussian_filter(np.random.rand(shape[0],shape[1])*2-1,sigma)*alpha
	dy = np.expand_dims(dy,axis=2)
	dy = np.concatenate((dy,dy,dy),axis=2)

	x,y,z = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]),np.arange(shape[2]))
	indices = np.reshape(y+dy,(-1,1)),np.reshape(x+dx,(-1,1)),np.reshape(z,(-1,1))

	return map_coordinates(img,indices,order=1,mode='reflect').reshape(shape)


def get_ICME_img(H=256,W=256,use_random_sample=False):
    global BATCH_SIZE,walk_obj_ICME,ICME_Root,total_load_nums,already_load_img_dict,fname_list,fname_list_temp,test_id,test_img,test_tensor,test_labels,END_OF_ONE_EPOCH,root_fold
    coord_labels=[]
    state_tensor=[]
    i=0
    while i<BATCH_SIZE:
        if not use_random_sample:
            try:
                root_fold,fname=walk_obj_ICME.__next__()
            except StopIteration:
                walk_obj_ICME=walk_dir(ICME_Root)
                root_fold,fname=walk_obj_ICME.__next__()
                total_load_nums=0
                END_OF_ONE_EPOCH=True
        else:
            sample_id= np.random.randint(len(fname_list))
            fname=fname_list.pop(sample_id)
            fname_list_temp.append(fname)
            if len(fname_list)==0:
                fname_list=copy.deepcopy(fname_list_temp)
                fname_list_temp=[]
                total_load_nums=0
                END_OF_ONE_EPOCH=True
        if fname.endswith('.png') or fname.endswith('.jpg'):
            frame_img= cv.imread(root_fold+'/'+fname)
            ground_truth_frame=get_ground_truth_frame(root_fold+'/'+fname+'.txt')
            file_rect=open(root_fold+'/'+fname+'.rect','r')
            left,top,right,bottom=map(int,file_rect.readline().strip().split())
            file_rect.close()

            if not (fname.find('IBUG')!=-1 or fname.find('ibug')!=-1):
                '''
                left+=int((right-left)*(random.random()-0.5)/10)
                top+=int((bottom-top)*(random.random()-0.5)/10)
                right+=int((right-left)*(random.random()-0.5)/10)
                bottom+=int((bottom-top)*(random.random()-0.5)/10)
                '''
                left,top,right,bottom = rand_bound(left,top,right,bottom,ground_truth_frame)

            cut_img,_,s_f=slice_image.slice_frame(frame_img,left,top,right,bottom,(frame_img.shape[1],frame_img.shape[0]),slicewh=(W,H))

            #cut_img=adjust_color(cut_img)
            # print(cut_img.shape)
            # clahe = cv.createCLAHE(clipLimit=600.0, tileGridSize=(1,1))
            # cut_img = clahe.apply(cut_img)            

            if not use_random_sample:
                fname_list.append(fname)
            if fname.find('IBUG')!=-1 or fname.find('ibug')!=-1:
                test_id[fname]=True
                ground_truth_frame=(ground_truth_frame-np.array([left,top]))*s_f
                test_labels.append(ground_truth_frame)
                # show_img=copy.deepcopy(cut_img)
                # for test_point in range(ground_truth_frame.shape[0]):
                #     cv.circle(show_img,(int(test_labels[-1][test_point][0]),int(test_labels[-1][test_point][1])),2,(0,255,0),thickness=-1)
                # cv.imshow('show_img',show_img)
                # cv.waitKey(1000)                
                #print('hit test: '+fname)
                
                test_img.append(cut_img)
                s_input=(cut_img-127.5)/128.0
                test_tensor.append(s_input)
                # show_img=copy.deepcopy(cut_img)
                # for test_point in range(ground_truth_frame.shape[0]):
                #     cv.circle(show_img,(int(test_labels[-1][test_point][0]),int(test_labels[-1][test_point][1])),2,(0,255,0),thickness=-1)
                # cv.imshow('show_img',show_img)
                # cv.waitKey(1000)                
            else:
                ground_truth_frame=(ground_truth_frame-np.array([left,top]))*s_f
                
                cut_img=random_cover(cut_img,ground_truth_frame,int(20*random.random()),int(20*random.random()))
                
                rand_seed=random.random()
                rand_seed1=random.random()
                rand_seed2=random.random()
                rand_seed3=random.random()
                
                '''
                if rand_seed1<0.5:
                    cut_img,ground_truth_frame=compress_trans(cut_img,ground_truth_frame)
                
                if rand_seed2<0.5:
                    cut_img,ground_truth_frame=big_small_face(cut_img,ground_truth_frame)
                '''

                if rand_seed3<0.5:
                	cut_img = elastic_transform(cut_img,30,5)

                if rand_seed<0.1:
                    cut_img=cv.cvtColor(cut_img,cv.COLOR_RGB2GRAY)
                    cut_img=cv.cvtColor(cut_img,cv.COLOR_GRAY2RGB)                
                elif rand_seed<0.3:
                    cut_img=random_lighten(cut_img)
                elif rand_seed<0.8:
                    cut_img, ground_truth_frame=rot(cut_img, ground_truth_frame, 50*(random.random()-0.5))
                else:
                    cut_img=gamma_correction(cut_img,random.random()+0.5)
                # elif random.random()<0.6:
                #     print(cut_img.shape)
                #cut_img, ground_truth_frame=rot(cut_img, ground_truth_frame, 45*(random.random()-0.5))
                ground_truth_frame=ground_truth_frame*np.array([W,H])/np.array([cut_img.shape[1],cut_img.shape[0]])
                cut_img=cv.resize(cut_img, (W,H), interpolation = cv.INTER_LINEAR)
                
                s_input=(cut_img-127.5)/128.0

                
                img_flip,gt_flip = flip_img(s_input,ground_truth_frame)

                state_tensor.append(s_input)
                coord_labels.append(ground_truth_frame)

                state_tensor.append(img_flip)
                coord_labels.append(gt_flip)
                
                '''
                if np.random.rand()<0.5:
                    state_tensor.append(s_input)
                    coord_labels.append(ground_truth_frame)
                else:
                    img_flip,gt_flip = flip_img(s_input,ground_truth_frame)
                    state_tensor.append(img_flip)
                    coord_labels.append(gt_flip)
                '''

                i+=1
                ##############visualization##############
                # show_img=copy.deepcopy(cut_img)
                # for test_point in range(ground_truth_frame.shape[0]):
                #     cv.circle(show_img,(int(coord_labels[-1][test_point][0]),int(coord_labels[-1][test_point][1])),2,(0,255,0),thickness=-1)
                # cv.imshow('show_img',show_img)
                # cv.waitKey(1000)
            total_load_nums+=1
            if total_load_nums%3000==0:
                print(root_fold+'/'+fname)
    #print(state_tensor[0].shape)
    return np.array(state_tensor),np.array(coord_labels)

def get_validate_img():
    global test_img,test_tensor,test_labels,BATCH_SIZE
    img_list=[]
    state_tensor=[]
    coord_labels=[]
    for _ in range(BATCH_SIZE):
        if len(test_tensor)==0:
            break
        img_list.append(test_img.pop())
        state_tensor.append(test_tensor.pop())
        coord_labels.append(test_labels.pop())
    return img_list, np.array(state_tensor),np.array(coord_labels) 

if __name__ == '__main__':
    # gen_img()
    # print('-------------------------')
    # gen_img()
    get_ICME_img()
    # print('-------------------------')
    # gen_AFLW_img()

