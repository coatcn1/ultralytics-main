from ultralytics import YOLO  # 导入YOLO模型类
from matplotlib import pyplot as plt
import numpy as np
import cv2  # 导入OpenCV库

# 加载预训练的模型
model = YOLO('runs/segment/train4/weights/best.pt')

# 读取图片并调整其大小以匹配模型训练时的输入尺寸
orig_img = cv2.imread('E:\huanggua\D10_20240414113132_32.jpg')  # 使用cv2读取图片
# 这里特别注意，因为使用yolov8训练的时候默认会把图片resize成448*640的尺寸，所以这里也得改成你训练的尺寸
orig_img_resized = cv2.resize(orig_img, (640, 448))  # 调整图片大小

# 使用模型对调整后的图片进行推理
results = model(orig_img_resized, save=True)

# 初始化一个布尔数组掩码，用于合并所有检测到的物体掩码
combined_mask = np.zeros(orig_img_resized.shape[:2], dtype=np.bool_)

# 遍历检测到的所有掩码
for mask in results[0].masks.data:
    mask_bool = mask.cpu().numpy().astype(np.bool_)  # 将掩码转换为布尔数组
    combined_mask |= mask_bool  # 使用逻辑或操作合并掩码

# 使用合并后的掩码创建抠图
masked_image = np.zeros_like(orig_img_resized)  # 初始化一个全黑的图片数组
masked_image[combined_mask] = orig_img_resized[combined_mask]  # 应用掩码

# 创建一个带有透明背景的RGBA图像
alpha_channel = np.ones(combined_mask.shape, dtype=orig_img.dtype) * 255  # 创建全白的alpha通道
masked_image_rgba = np.dstack((masked_image, alpha_channel))  # 将RGB图像和alpha通道合并
masked_image_rgba[~combined_mask] = (0, 0, 0, 0)  # 设置背景为透明

# 保存两种处理后的图像
cv2.imwrite('masked_image_all_objects.jpg', masked_image)  # 保存带黑色背景的图像
cv2.imwrite('masked_image_all_objects.png', masked_image_rgba)  # 保存带透明背景的图像

# 显示第一张处理后的图像
masked_image = cv2.resize(masked_image, (1200, 950))  # 调整图像大小
cv2.imshow("YOLOv8 Inference", masked_image)  # 显示图像

cv2.waitKey(0)  # 等待用户按键
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口