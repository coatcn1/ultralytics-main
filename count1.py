import argparse
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

track_history = defaultdict(list)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


counting_regions = [
    {
        'name': 'YOLOv8 Polygon Region',
        'polygon': Polygon([(50, 80), (250, 20), (450, 80), (400, 350), (100, 350)]),
        'counts': 0,
        'dragging': False,
        'region_color': (255, 42, 4),
        'text_color': (255, 255, 255)
    },
    {
        'name': 'YOLOv8 Rectangle Region',
        'polygon': Polygon([(200, 250), (440, 250), (440, 550), (200, 550)]),
        'counts': 0,
        'dragging': False,
        'region_color': (37, 255, 225),
        'text_color': (0, 0, 0)
    }
]

def mouse_callback(event, x, y, flags, param):
    global current_region  # 声明为全局变量

    if event == cv2.EVENT_LBUTTONDOWN:
        for region in counting_regions:
            if region['polygon'].contains(Point((x, y))):
                current_region = region
                current_region['dragging'] = True
                current_region['offset_x'] = x
                current_region['offset_y'] = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if current_region is not None and current_region['dragging']:
            dx = x - current_region['offset_x']
            dy = y - current_region['offset_y']
            current_region['polygon'] = Polygon([
                (p[0] + dx, p[1] + dy) for p in current_region['polygon'].exterior.coords])
            current_region['offset_x'] = x
            current_region['offset_y'] = y

    elif event == cv2.EVENT_LBUTTONUP:
        if current_region is not None and current_region['dragging']:
            current_region['dragging'] = False

def run(weights='runs/detect/train112/weights/best.pt', source=None, device='cpu', view_img=False, save_img=False, exist_ok=False, classes=None, line_thickness=2, track_thickness=2, region_thickness=2):
    vid_frame_count = 0

    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    model = YOLO(weights)
    model.to('cuda') if device == '0' else model.to('cpu')

    names = model.model.names

    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    save_dir = increment_path(Path('ultralytics_rc_output') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        vid_frame_count += 1

        results = model.track(frame, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                track = track_history[track_id]
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                for region in counting_regions:
                    if region['polygon'].contains(Point((bbox_center[0], bbox_center[1]))):
                        region['counts'] += 1  # 累加计数

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow('Ultralytics YOLOv8 Region Counter Movable')
                cv2.setMouseCallback('Ultralytics YOLOv8 Region Counter Movable', mouse_callback)
            cv2.imshow('Ultralytics YOLOv8 Region Counter Movable', frame)

        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()

    # 打印最终计数结果
    for region in counting_regions:
        print(f"Region: {region['name']}, Total Counts: {region['counts']}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/coatcn/workspace/yolocode/112/best.pt', help='initial weights path')  # 修改为新的路径
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--source', type=str, default='/home/coatcn/workspace/yolocode/video/VID_20240223_152846.mp4', help='video file path')  # 修改为新的路径
    parser.add_argument('--view-img', action='store_true', default=True, help='show results')
    parser.add_argument('--save-img', action='store_true', default=True, help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--line-thickness', type=int, default=2, help='bounding box thickness')
    parser.add_argument('--track-thickness', type=int, default=None, help='Tracking line thickness')
    parser.add_argument('--region-thickness', type=int, default=4, help='Region thickness')

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
