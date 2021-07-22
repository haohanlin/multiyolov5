import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torch.nn.functional as F

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np


Cityscapes_COLORMAP = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
]

Cityscapes_IDMAP = [
    [7],
    [8],
    [11],
    [12],
    [13],
    [17],
    [19],
    [20],
    [21],
    [22],
    [23],
    [24],
    [25],
    [26],
    [27],
    [28],
    [31],
    [32],
    [33],
]

Cityscapes_Class = ["road", "sidewalk", "building", "wall", "fence",
               "pole", "traffic light", "traffic sign", "vegetation",
               "terrain", "sky", "person", "rider", "car", "truck",
               "bus", "train", "motorcycle", "bicyle"]


def label2image(pred, COLORMAP=Cityscapes_COLORMAP):
    colormap = np.array(COLORMAP, dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]

def trainid2id(pred, IDMAP=Cityscapes_IDMAP):
    colormap = np.array(IDMAP, dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    if opt.submit:
        sub_dir = str(save_dir) + "/results/"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA 原始代码,cpu用float32,gpu用float16
    # half = False  # 强制禁用float16推理, 20和30系列显卡有tensor cores float16, 10系列卡不开cudnn.benchmark速度反而降
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model 加载模型
    stride = int(model.stride.max())  # model strides 模型最大步数
    imgsz = check_img_size(imgsz, s=stride)  # check img_size 
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader 数据加载
    vid_path, vid_writer, s_writer = None, None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
                                # 开启后第一次推理会把各种后端算法测试一遍,后续推理都用最快的算法,会有较明显加速
                                # 算法速度不仅与复杂度有关,也与输入规模相关,因此要求后续输入同尺寸,原版仅在视频测试时开启,想测真实速度应该开启
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        cudnn.benchmark = False
        dataset = LoadImages(source, img_size=imgsz, stride=stride)  # 跑的是这个

    
    if opt.submit or opt.save_as_video:  # 提交和做视频必定是同尺寸
        cudnn.benchmark = True
        
    # Get names and colors
    # 目标检测的类别名字
    names = model.module.names if hasattr(model, 'module') else model.names
    print('---whx--- names ',names)
    # 不同的名字类别对应不同的颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    print('---whx--- colors ',colors)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once 运行一次
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:  #加载的图片数据
        img = torch.from_numpy(img).to(device) #numpy数据转换成 tensor
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:  #返回tensor的维度（整数）
            img = img.unsqueeze(0) #unsqueeze(0)后就会在0的位置加了一维就变成一行三列（1,3） unsqueeze主要起到升维的作用，后续图像处理可以更好地进行批操作

        # Inference
        #上下文管理器中执行此操作 不记录操作，将梯度设置为零 
        # 否则，我们的梯度会记录所有已发生操作的运行记录即loss.backward()将梯度添加到已存储的内容中，而不是替换它们）
        with torch.no_grad():
            t1 = time_synchronized()
            out = model(img, augment=opt.augment) #模型加载图片计算输出数据
            pred = out[0][0] # 目标检测结果
            seg = out[1]  # [0]分割结果
        # Apply NMS 非最大值抑制。 主要作用赛选目标框
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections 检测过程
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path 路径
            save_path = str(save_dir / p.name)  # img.jpg生成保存路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string  
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh 归一化增益whwh
            print('---whx--- det:',det.shape,im0.shape)
            if len(det):
                # Rescale boxes from img_size to im0 size 从img_size缩放到im0大小  缩放坐标，对应大图片
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique(): #c 表示检测到几个类别分别是什么
                    print('---whx--- c:',c ) 
                    n = (det[:, -1] == c).sum()  # detections per class  计算出现的每个类别有几个 n 
                    print('---whx---- n:',n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string 连成字符串

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image 把框叠加到图 
                        # 图片类别标签 和置信度
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # xyxy 是坐标 im0是原图  label标签  color框的颜色 line_thickness线条粗细
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.5f}s)')
            # seg = seg[0]
            # 分割 插入
            print('---whx---',seg.shape)
            #seg 为[1, 19, 512, 1024] 要采用插值，才能达到原图大小
            seg = F.interpolate(seg, (im0.shape[0], im0.shape[1]), mode='bilinear', align_corners=True)[0]
            #seg 为[19, 1024, 2048] 原图
            #把图片按分割标签进行绘制颜色
            mask = label2image(seg.max(axis=0)[1].cpu().numpy(), Cityscapes_COLORMAP)[:, :, ::-1]
            # print('----whx---- seg : ',seg.max(axis=0)[1])
            # print('----whx---- seg 222: ',seg.max(axis=0)[1].cpu().numpy())
            print('---whx--- mask:',mask)
            dst = cv2.addWeighted(mask, 0.4, im0, 0.6, 0)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0) #保存目标检测模型
                cv2.imshow("segmentation", mask) #保存分割模型
                cv2.imshow("mix", dst) #保存合成模型
                cv2.waitKey(0)  # 1 millisecond

            if opt.submit:
                sub_path = sub_dir+str(p.name)
                sub_path = sub_path[:-4] + "_pred.png"
                result = trainid2id(seg.max(axis=0)[1].cpu().numpy(), Cityscapes_IDMAP)
                cv2.imwrite(sub_path, result)
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    cv2.imwrite(save_path[:-4]+"_mask"+save_path[-4:], mask)
                    cv2.imwrite(save_path[:-4]+"_dst"+save_path[-4:], dst)

                else: # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, dst.shape[1], dst.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(dst)#(im0)
            if opt.save_as_video:
                if not s_writer:
                    fps, w, h = 30, dst.shape[1], dst.shape[0]
                    s_writer = cv2.VideoWriter(str(save_dir)+"out.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                s_writer.write(dst)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    if s_writer != None:
        s_writer.release()
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-as-video', action='store_true', help='save same size images as a video')
    parser.add_argument('--submit', action='store_true', help='get submit file in folder submit')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
