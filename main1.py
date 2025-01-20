# import sys
# sys.path.append("autodl-tmp/yolov8/")

from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # Load a model
    # # 直接使用预训练模型创建模型
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'ultralytics/cfg/datasets/Mydata.yaml'}, epochs=500, imgsz=640, batch=32)
    #
    # #使用yaml配置文件来创建模型，并导入预训练权重
    # model = YOLO('ultralytics/cfg/models/myyaml/yolov8-Slimneck-SPPELAN-ECA2.yaml') # build a new model from YAML
    # model.load('weights/yolov8n.pt')
    # model.train(**{'cfg': 'ultralytics/cfg/default.yaml', 'data': 'ultralytics/cfg/datasets/Mydata-Aug.yaml'},
    #             epochs=1000, imgsz=640, batch=64, name='train')  # name：是此次训练结果保存的文件夹   数据集是我自己的数据集

# #     # 模型验证：用验证集
#     model = YOLO('runs/detect/train/weights/best.pt')
#     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'name':'val', 'batch':32}) #模型验证用验证集
#     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'split':'test', 'iou':0.9}) #模型验证用测试集

    # 模型推理：
    model = YOLO('runs/detect/train50/weights/best.pt')
    model.predict(source='test/lettuce1/0081.jpg', name='predict', **{'save':True})

    # from ultralytics import YOLO
    #
    # # Load a model
    # model = YOLO("runs/segment/train/weights/best.pt")  # pretrained YOLOv8n model
    #
    # # Run batched inference on a list of images
    # results = model(["data/Images/0002_1.jpg", "data/Images/0002_2.jpg"])  # return a list of Results objects

    # # Process results list
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     result.show()  # display to screen
    #     result.save(filename="result.jpg")  # save to disk