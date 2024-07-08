import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the pretrained YOLOv8 model
model = YOLO("best.pt")  # 确保这是正确的模型路径

# Read an image
image = cv2.imread("ultralytics/cfg/datasets/teeth-data/images/test/color_1.png")  # 确保这是正确的图像路径

# Run inference on the image
results = model.predict(image, save=False)

# 遍历预测结果中的每个结果对象
for r in results:
    # 复制原始图像以便修改
    img = np.copy(r.orig_img)
    # 提取并处理图像名称
    img_name = Path(r.path).stem

    # 遍历每个结果中的对象，这些对象可能代表不同的检测到的实体
    for ci, c in enumerate(r):
        # 获取检测到的对象的标签名称
        label = c.names[c.boxes.cls.tolist().pop()]

        # 创建一个与原图大小相同的黑色掩码
        b_mask = np.zeros(img.shape[:2], np.uint8)

        # 从检测对象中提取轮廓并转换为整数坐标
        # contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        #  Extract contour result
        contour = c.masks.xy.pop()
        #  Changing the type
        contour = contour.astype(np.int32)
        #  Reshaping
        contour = contour.reshape(-1, 1, 2)
        # 在黑色掩码上绘制白色的轮廓，实现填充效果
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # 选择一种处理方式:

        # 选项1: 将对象与黑色背景隔离
        # 将单通道的黑白掩码转换为三通道格式
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        # 使用掩码与原图进行按位与操作，仅保留掩码区域的像素
        isolated = cv2.bitwise_and(mask3ch, img)
        # 裁剪图像使其只包括目标区域所需的其他步骤。
        #  Bounding box coordinates
        x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
        print(f"{img_name}_{label}: {x1, y1, x2, y2}")
        # Crop image to object region
        iso_crop = isolated[y1:y2, x1:x2]

        # OPTION-2: 隔离具有透明背景的对象（当保存为 PNG 时）
        # isolated = np.dstack([img, b_mask])


        # 将处理后的图像保存到文件
        # cv2.imwrite(f"{img_name}_{label}.png", isolated)
        cv2.imshow(f"{img_name}_{label}", iso_crop)
        cv2.waitKey(0)

