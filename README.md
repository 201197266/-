# -
通过cv2和wordcloud简单实现了轮廓词云图
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2021/5/11 18:46
# @Author : YM
# @File : 词云.py
# @Software: PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import wordcloud
from PIL import Image
word_dict = './自定义词库/词库.txt'
jieba.load_userdict(word_dict)
stop_word = [w.replace('\n', '') for w in open('./自定义词库/停用词表.txt', encoding='utf8')]

# 1. 读取原图
img = cv2.imread('D:/Resource/nezha.jpg')  # 路径纯英文
if img is None:
    raise FileNotFoundError("图像加载失败，请检查路径。")

# 2. 获取灰度图用于边缘提取
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# image_colors = wordcloud.ImageColorGenerator(mask_array)
# 3. 边缘检测
# w = wordcloud.WordCloud(mask=mask_array)
image_colors = wordcloud.ImageColorGenerator(img_rgb)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, threshold1=100, threshold2=150)
edges = cv2.dilate(edges, np.ones((10, 10), np.uint8), iterations=1)
# 4. 提取“非白背景区域”：
# 将图像转为 HSV 模式，更易识别“接近白色”的区域
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 定义白色范围（你可以根据图像情况微调）
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
# 得到白色区域 mask
white_mask = cv2.inRange(hsv, lower_white, upper_white)
# 取反，得到“非白区域”
non_white_mask = cv2.bitwise_not(white_mask)
# 5. 与边缘图组合，得到最终词云 mask
combined_mask = cv2.bitwise_and(edges, non_white_mask)
# 6. 为 WordCloud 生成适配格式（255/0）
final_mask = np.where(combined_mask > 0, 255, 0).astype(np.uint8)
# 7. 生成词云（使用 mask 控制形状）
txt = open('./原始数据的txt/哪吒utf8.txt', encoding='utf8').read()
txtlist = [w for w in jieba.lcut(txt) if w not in stop_word]
string = " ".join(txtlist)

 # 示例词汇
wc = WordCloud(
	font_path='./词云/HuaWenYuanTi-REGULAR-2.ttf',
	background_color="white",
	scale=10,
    max_words=800, # 词过大时会报错
	repeat = True,
    mask=final_mask,
	mode="RGBA",
	color_func = image_colors
)
wc.generate(string)
wc.to_file('./词云/wk.png')

plt.imshow(wc, cmap='gray')
plt.show()
plt.imshow(final_mask, cmap='gray')
plt.show()

