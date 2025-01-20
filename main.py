# import sys
# sys.path.append("autodl-tmp/yolov8/")

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    # # # 直接使用预训练模型创建模型
    model = YOLO('yolov8n.pt')
    model.train(**{'cfg':'ultralytics/cfg/default.yaml', 'data':'D:\code\mydata/bug.yaml'}, epochs=200, imgsz=640, batch=32)

    # #使用yaml配置文件来创建模型，并导入预训练权重
    # model = YOLO('ultralytics/cfg/models/myyaml/yolov8-Slimneck-SPPELAN-ECA-NEW2.yaml') # build a new model from YAML
    # model.load('weights/yolov8n.pt')
    # model.train(**{'cfg': 'ultralytics/cfg/default.yaml', 'data': 'ultralytics/cfg/datasets/Mydata-Aug.yaml'},
    #             epochs=1000, imgsz=640, batch=64, name='train')  # name：是此次训练结果保存的文件夹   数据集是我自己的数据集

# #     # 模型验证：用验证集
#     model = YOLO('runs/detect/train/weights/best.pt')
#     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'name':'val', 'batch':32}) #模型验证用验证集
#     model.val(**{'data':'ultralytics/models/yolo/detect/mydata/traffic.yaml', 'split':'test', 'iou':0.9}) #模型验证用测试集

    # 模型推理：
    # model = YOLO('runs/detect/train50/weights/best.pt')
    # model.predict(source='D:/y/ultralytics-main/test/lettuce2', name='predict', **{'save':True})

# from ray import tune
#
# from ultralytics import YOLO
#
# # Define a YOLO model
# model = YOLO("yolov8n.pt")
#
# # Run Ray Tune on the model
# result_grid = model.tune(data="coco128.yaml",
#                          space={"lr0": tune.uniform(1e-5, 1e-1)},
#                          epochs=5,
#                          use_ray=True)






# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO('ultralytics/cfg/models/v8/yolov8-C2f-FasterBlock.yaml')
#     # model.load('yolov8n.pt') # loading pretrain weights
#     model.train(data=r'替换数据集yaml文件地址',
#                 # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
#                 cache=False,
#                 imgsz=640,
#                 epochs=150,
#                 single_cls=False,  # 是否是单类别检测
#                 batch=4,
#                 close_mosaic=10,
#                 workers=0,
#                 device='0',
#                 optimizer='SGD', # using SGD
#                 # resume='', # 如过想续训就设置last.pt的地址
#                 amp=False,  # 如果出现训练损失为Nan可以关闭amp
#                 project='runs/train',
#                 name='exp',
#                 )