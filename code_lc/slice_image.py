# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import copy
import random

def in_image(imgwh,coordinate):
    if not 0<=coordinate[0]<imgwh[0]:
        return False
    if not 0<=coordinate[1]<imgwh[1]:
        return False
    return True

def slice_frame(frame_img,left,top,right,bottom,imgwh,slicewh=(256,256),true_neg=True,show=False):
    loww=int(max(left,0))
    upw=int(min(right,imgwh[0]))
    lowh=int(max(top,0))
    uph=int(min(bottom,imgwh[1]))
    slice_img1=np.zeros((bottom-top,right-left)+(3,),dtype=np.uint8)
    if lowh<uph and loww<upw:
        slice_img1[(lowh-top):(uph-top),(loww-left):(upw-left),:]=frame_img[lowh:uph,loww:upw,:]
    slice_img=cv.resize(slice_img1, slicewh, interpolation = cv.INTER_LINEAR)
    scale_factor=np.array(slice_img.shape[0:-1],np.float32)/np.array(slice_img1.shape[0:-1],np.float32)
    scale_factor=np.array(list(scale_factor)[::-1])
    if show:
        print(interocular_distance)
        cv.imshow('frame_img',frame_img)
        cv.waitKey(1)    
        cv.imshow('slice_img',slice_img)
        cv.waitKey(1)    
        cv.imshow('slice_img1',frame_img[lowh:uph,loww:upw,:])    
        cv.waitKey(3000)
    return (slice_img,slice_img1,scale_factor)
