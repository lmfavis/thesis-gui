import streamlit as st
import mmcv
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
import torch
import glob
import pandas as pd
from os import path
from skimage import io, color
from skimage.feature import local_binary_pattern
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
import cv2
from pathlib import Path
import PIL

catelog={'ad':'Atopic Dermatitis Eczema','ak':'Actinic Keratosis','av':'Acne Vulgaris','bcc':'Basal Cell Carcinoma','mn':'Melanocytic Nevi','on':'Onychomycosis','rs':'Rosacea','sk':'Seborrheic Keratosis','su':'Stasis Ulcer','tc':'Tinea Corporis'}

st.set_page_config(layout="wide")

STYLE = """
    <style>
    .reportview-container .main footer {visibility: hidden;} 
    </style>   
    """

RETINANET = "/home/favis/Downloads/Thesis Models/alpha_025_gamma_1_ckpt/latest.pth"
OCLBP = "/home/favis/Downloads/Thesis Models/alpha_025_gamma_1_colorlbp_2_12_ckpt/latest.pth"
LBPRETINA = "/home/favis/Downloads/Thesis Models/alpha_025_gamma_1_normallbp_3_12_ckpt/latest.pth"

def normalizeTo256(img):
	return (img * 255 / np.max(img)).astype('uint8')

def compute_lbp_feature(img, radius, center):
	patterns = local_binary_pattern(img, radius, center)
	return patterns

def getRGB(image):
	rgbImg = image.convert("RGB")
	width, height = image.size[:2]
	
	red = np.zeros((height,width))
	green = np.zeros((height,width))
	blue = np.zeros((height,width))
	for i in range(height):
		for j in range(width):
			pixel = rgbImg.getpixel((j,i))
			red[i, j] = pixel[0]/255
			green[i, j] = pixel[1]/255
			blue[i, j] = pixel[2]/255
	return [red, green, blue]

def LBP(img):
	return compute_lbp_feature(img, 12, 2)

def rgbLBP(image_file):
	#img = Image.open(image_file)
	rgb = getRGB(image_file)

	redLBP = normalizeTo256(LBP(rgb[0]))
	greenLBP = normalizeTo256(LBP(rgb[1]))
	blueLBP = normalizeTo256(LBP(rgb[2]))

	width, height = image_file.size[:2]
	rgbArray = np.zeros((height,width, 3), 'uint8')

	rgbArray[..., 0] = redLBP
	rgbArray[..., 1] = greenLBP
	rgbArray[..., 2] = blueLBP

	return rgbArray

def runLBPColor(image_file):
	rgb = rgbLBP(image_file)
	arr = np.array(rgb)
	img = PIL.Image.fromarray(arr)
	return img

def compute_lbp_feature(img, radius, center):
	patterns = local_binary_pattern(img, radius, center)
	return patterns

def generate_lbp_image(image_file):
	slice56 = compute_lbp_feature(image_file, 12, 3)
	formatted = (slice56 * 255 / np.max(slice56)).astype('uint8')
	img = PIL.Image.fromarray(formatted)
	return img
	
	
def getBbox(result):
	classes=['ad','ak','av','bcc','mn','on','rs','sk','su','tc']
	prediction_per_class=list()
	bbox_per_class=list()
	
	for i in range(len(result)):
		confidence = [r[-1:] for r in result[i]]
		temp=[r[:4] for r in result[i]]
		if len(confidence)!=0:
			max=np.max(confidence)
			bbox_max=np.argmax(confidence)
			prediction_per_class.append(max)
			bbox_per_class.append(temp[bbox_max])
		else:
			bbox_per_class.append([0,0,0,0])
			prediction_per_class.append(0)
	print(bbox_per_class)

	pred=np.argmax(prediction_per_class)
	return pred,prediction_per_class[pred],bbox_per_class[pred]

def getClass(num):
	classes=['ad','ak','av','bcc','mn','on','rs','sk','su','tc']
	return catelog[classes[num]]

def getTrueLabel(imgPath):
	filename=os.path.basename(imgPath)
	true_label=filename[:filename.rindex("_")]
	return catelog[true_label]

def get_optimal_font_scale(text, width):
    for scale in reversed(range(0, 60, 1)):
    	textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=scale/10, thickness=1)
    	new_width = textSize[0][0]
    	print(new_width)
    	if (new_width <= width):
        	return scale/10
    return 1

def showPreprocessOff(img):
	st.markdown("***")
	st.subheader("Pre-processing for RetinaNet")
	st.write("")
	
	image = Image.open(img)
	st.image(image, width=224, caption="Step 1: Resize image to 224x224")

def showPreprocessOC(img, option):
	st.markdown("***")
	st.subheader("Pre-processing for " + option)
	st.write("")

	image = Image.open(img)
	col1, col2, col3 = st.columns(3)
	
	# APPLY GRAYSCALE
	# grayscale_img = image.convert('L')
	# APPLY LBP
	lbp_image = runLBPColor(image)

	# APPLY PRE-PROCESSING HERE
	col1.image(image, use_column_width=True, caption="Step 1: Open image")
	col2.image(lbp_image, use_column_width=True, caption="Step 2: Apply OC-LBP filter")
	col3.image(lbp_image, width=224, caption="Step 3: Resize image to 224x224")

def showPreprocessStandard(img, option):
	st.markdown("***")
	st.subheader("Pre-processing for " + option)
	st.write("")

	image = Image.open(img)
	col1, col2, col3 = st.columns(3)
	
	# APPLY GRAYSCALE
	grayscale_img = image.convert('L')
	# APPLY LBP
	lbp_image = generate_lbp_image(grayscale_img)

	# APPLY PRE-PROCESSING HERE
	col1.image(grayscale_img, use_column_width=True, caption="Step 1: Convert to grayscale")
	col2.image(lbp_image, use_column_width=True, caption="Step 2: Apply LBP filter")
	col3.image(lbp_image, width=224, caption="Step 3: Resize image to 224x224")

def checkImage(img, option):
	image = PIL.Image.open(img)
	if option == "LBP-Retina":
		grayscale_img = image.convert('L')
		lbp_image = generate_lbp_image(grayscale_img)
		lbp_image.save(os.path.dirname(os.path.abspath(__file__)) + "/standardLBP.jpg")
		return os.path.dirname(os.path.abspath(__file__)) + "/standardLBP.jpg"
	elif option == "OCLBP-Retina":
		lbp_image = runLBPColor(image)
		lbp_image.save(os.path.dirname(os.path.abspath(__file__)) + "/OC-LBP.jpg")
		return os.path.dirname(os.path.abspath(__file__)) + "/OC-LBP.jpg"
	return img

def main():
	# HIDE FOOTER
	st.markdown(STYLE, unsafe_allow_html=True)

	st.header("Classification of Skin Diseases using Local Binary Pattern and RetinaNet")
	st.write("This system utilizes RetinaNet and Local Binary Pattern in order to classify ten different skin diseases, namely: Acne Vulgaris, Actinic Keratosis, Atopic Dermatitis Eczema, Basal Cell Carcinoma, Melanocytic Nevi, Onychomycosis, Rosacea, Seborrheic, Keratosis, Stasis Ulcer, and Tinea Corporis. \nThis system contains three models: RetinaNet, LBP-Retina, and OCLBP-Retina.")

	st.markdown("***")

	st.subheader("Upload files")
	st.write("")

	# MODEL AND IMAGE UPLOAD
	cfgPath = "/home/favis/mmdetection/configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py"
	modelPath = ""
	modelOption = st.selectbox('Select Model:', ('RetinaNet', 'LBP-Retina', 'OCLBP-Retina'))
	uploaded_file = st.file_uploader("Choose an image...", type="jpg")
	st.write("")

	cfg = Config.fromfile(cfgPath)
	cfg.model.bbox_head.num_classes=10
	if st.button("Classify"):
		image = Image.open(uploaded_file)
		directory = "tempDir"
		path = os.path.join(os.getcwd(), directory)
		p = Path(path)
		if not p.exists():
			os.mkdir(p)
		with open(os.path.join(path, uploaded_file.name),"wb") as f:
			f.write(uploaded_file.getbuffer()) 
		file_loc = os.path.join(path, uploaded_file.name)
	
		if modelOption == "RetinaNet":
			showPreprocessOff(file_loc)
			modelPath = RETINANET
		elif modelOption == "LBP-Retina":
			showPreprocessStandard(file_loc, modelOption)
			modelPath = LBPRETINA
		else:
			showPreprocessOC(file_loc, modelOption)
			modelPath = OCLBP
		
		model = init_detector(cfg, modelPath, device='cuda:0')
		result = inference_detector(model, checkImage(file_loc, modelOption))
		
		p,conf,bbox=getBbox(result)
		print(p,conf,bbox)
		
		matplotlib.use('TkAgg')
		# show_result_pyplot(model, checkImage(file_loc, modelOption), result, score_thr=0.10)

		img = cv2.imread(checkImage(file_loc, modelOption))
		start=int(bbox[0]), int(bbox[1])
		end=int(bbox[2]),int(bbox[3])
		cv2.rectangle(img, start, end, (16, 89, 234),5)

		start=int(bbox[0])+10, int(bbox[1])+35
		text=getClass(p)+" | confidence:"+"{: .2f}".format(conf)
		# img = cv2.putText(img, text, start, cv2.FONT_HERSHEY_COMPLEX, get_optimal_font_scale(text,int(bbox[2])-int(bbox[0])), (16, 89, 234), 1, cv2.LINE_AA)
		
		st.markdown("***")
		st.subheader("Results")
		st.write("")
		
		st.image(img, caption=None, channels="BGR", output_format="auto")
		st.markdown("<h4 style='text-align: center; '>True Label: "+getTrueLabel(file_loc)+"</h4>", unsafe_allow_html=True)
		st.markdown("<h4 style='text-align: center; '>Predicted: "+text+"</h4>", unsafe_allow_html=True)

main()
