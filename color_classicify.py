# coding=utf-8
# 导入一些python包
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import imutils


# 创建一个颜色标签类
class ColorLabeler:
	def __init__(self):
		# 初始化一个颜色词典
		colors = OrderedDict({
			"red":    (255, 0, 0),
			"green":  (0, 255, 0),
			"blue":   (0, 0, 255),
			"yellow": (255, 255, 0)})

		# 为LAB图像分配空间
		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []

		# 循环 遍历颜色词典
		for (i, (name, rgb)) in enumerate(colors.items()):
			# 进行参数更新
			self.lab[i] = rgb
			self.colorNames.append(name)

		# 进行颜色空间的变换
		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

	def label(self, image, c):
		# 根据轮廓构造一个mask，然后计算mask区域的平均值
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.erode(mask, None, iterations=2)
		mean = cv2.mean(image, mask=mask)[:3]

		# 初始化最小距离
		minDist = (np.inf, None)

		# 遍历已知的LAB颜色值
		for (i, row) in enumerate(self.lab):
			# 计算当前l*a*b*颜色值与图像平均值之间的距离
			d = dist.euclidean(row[0], mean)

			# 如果当前的距离小于最小的距离，则进行变量更新
			if d < minDist[0]:
				minDist = (d, i)

		# 返回最小距离对应的颜色值
		return self.colorNames[minDist[1]]


def img_process(img_path):
	# 读取图片
	image = cv2.imread(img_path, 1)

	# 进行图片灰度化
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# 进行颜色空间的变换
	lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	# 进行阈值分割
	thresh = cv2.threshold(gray, 219, 255, cv2.THRESH_BINARY)[1]

	# 在二值图片中寻找轮廓
	cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cnt = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)

	# 初始颜色标签
	cl = ColorLabeler()

	# 遍历每一个轮廓
	for c, h in zip(cnts, hierarchy[0]):  # TODO 找到包含关系
		if (h == [-1, -1, 1, -1]).all():
			# 最外层轮廓(整幅图)
			continue

		# 计算每一个轮廓的中心点
		M = cv2.moments(c)
		cX = int((M["m10"] / M["m00"]))
		cY = int((M["m01"] / M["m00"]))

		# 进行颜色检测
		color = cl.label(lab, c)

		# 进行坐标变换
		c = c.astype("float")
		c = c.astype("int")
		text = "{}".format(color)
		# 绘制轮廓并显示结果
		cv2.drawContours(image, [c], -1, (0, 0, 0), 2)
		cv2.putText(image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		cv2.imshow("Image", image)
		cv2.waitKey(0)


if __name__ == '__main__':
	img_process(r'C:\Work\Code\Opencv\2.png')