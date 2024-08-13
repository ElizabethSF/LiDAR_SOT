import os, argparse, json, numpy as np, yaml, multiprocessing, shutil
import sot_3d, sot_3d.utils as utils
from sot_3d.data_protos import BBox
from data_loader import ExampleLoader
from copy import deepcopy
from sot_3d.visualization import Visualizer2D

import open3d as o3d


parser = argparse.ArgumentParser()
# paths
parser.add_argument('--bench_list', type=str, default='./benchmark/vehicle/bench_list.json', 
    help='the path of benchmark object list')
parser.add_argument('--data_folder', type=str, default='../datasets/waymo/sot/',
    help='store the data')
parser.add_argument('--result_folder', type=str, default='../TrackingResults/',
    help='path to store the tracking results')
parser.add_argument('--config_path', type=str, default='config.yaml', help='config path')
# running configurations
parser.add_argument('--name', type=str, default='debug', help='name of this experiments')
parser.add_argument('--process', type=int, default=1, help='multiprocessing for acceleration')
parser.add_argument('--skip', action='store_true', default=False, help='skip the tracklets already finish')
parser.add_argument('--visualize', action='store_true', default=False)
# debug mode
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--max_len', type=int, default=200)
# parse arguments
args = parser.parse_args()

data_index = "1"


def find_bboxes(id, data_folder, segment_name, start_frame, end_frame):
    """ In the SOT, the beginning frame is a GT BBox
        This function is for finding the gt bbox
    Args:
        id (str): id of the tracklet
        data_folder (str): root for data storage
        segment_name (str): the segment to look at
        start_frame (int): which frame to search
    Return
        BBox (numpy array): [x, y, z, h, l, w, h]
    """
    gt_info = np.load(os.path.join(data_folder, 'gt_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    ego_info = np.load(os.path.join(data_folder, 'ego_info', '{:}.npz'.format(segment_name)), 
        allow_pickle=True)
    bboxes, ids = gt_info['bboxes'][start_frame], gt_info['ids'][start_frame]
    index = ids.index(id)
    start_bbox = bboxes[index]

    gts = list()
    for i in range(start_frame, end_frame + 1):
        frame_bboxes = gt_info['bboxes'][i]
        frame_ids = gt_info['ids'][i]
        index = frame_ids.index(id)
        bbox = frame_bboxes[index]
        bbox = BBox.array2bbox(bbox)

        ego_matrix = ego_info[str(i)]
        bbox = BBox.bbox2world(ego_matrix, bbox)
        gts.append(bbox)

    return start_bbox, gts


from math import sqrt

def calculate_distance(bbox1, bbox2):
    """Calculate the Euclidean distance between the centers of two bounding boxes."""
    
    distance = sqrt((bbox1[0] - bbox2[0]) ** 2 + (bbox1[1] - bbox2[1]) ** 2)
    return distance

def calculate_heading_error(bbox1, bbox2):
    
    heading_error = abs(bbox1[3] - bbox2[3])*180/3.1415926
    
    heading_error = min(heading_error, abs(360 - heading_error))
    
    heading_error = min(heading_error, abs(180 - heading_error))
    
    
    return heading_error

def compare_to_gt(cur_frame_idx, frame_result, gts=None):
    """ For the detail of frame_result, refer to the get_frame_result in tracker.py
    """
    result = deepcopy(frame_result)
    max_frame_key = max(list(frame_result.keys()))
    for i in range(max_frame_key + 1):
        frame_idx = cur_frame_idx - (max_frame_key - i)
        bbox0, bbox1 = frame_result[i]['bbox0'], frame_result[i]['bbox1']
        result[i]['bbox0'] = BBox.bbox2array(bbox0).tolist()
        result[i]['bbox1'] = BBox.bbox2array(bbox1).tolist()

        if gts:
            gt_bbox0, gt_bbox1 = gts[frame_idx - 1], gts[frame_idx]
            iou_2d, iou_3d = sot_3d.utils.iou3d(bbox1, gt_bbox1)
            
            result[i]['gt_bbox0'] = BBox.bbox2array(gt_bbox0).tolist()
            result[i]['gt_bbox1'] = BBox.bbox2array(gt_bbox1).tolist()
            result[i]['gt_motion'] = (BBox.bbox2array(gt_bbox1) - BBox.bbox2array(gt_bbox0))[:4].tolist()
            result[i]['iou2d'] = iou_2d
            result[i]['iou3d'] = iou_3d
            
            # Calculate distance between the centers of the bounding boxes
            result[i]['loc_error'] = calculate_distance(BBox.bbox2array(gt_bbox1), BBox.bbox2array(bbox1))
            result[i]['heading_error'] = calculate_heading_error(BBox.bbox2array(gt_bbox1), BBox.bbox2array(bbox1))
            
            
    return result



def id_track(configs, id, segment_name, frame_range, data_folder):
    """ ID tracking, prepare the data loader and call the tracker_api
    """
    # initialize the data loader
    data_loader = ExampleLoader(configs=configs, id=id, segment_name=segment_name, 
        data_folder=data_folder, frame_range=frame_range)
    # find the starting bbox
    start_bbox, gts = find_bboxes(id=id, data_folder=data_folder, 
        segment_name=segment_name, start_frame=frame_range[0], end_frame=frame_range[1])
    # run the tracker
    tracking_results = tracker_api(configs=configs, id=id, start_bbox=start_bbox,
        start_frame=frame_range[0], data_loader=data_loader, track_len=frame_range[1]-frame_range[0]+1,
        gts=gts, visualize=args.visualize)
    return tracking_results


def frame_result_visualization(frame_result, pc):
    visualizer = Visualizer2D(figsize=(12, 12))
    bbox0, bbox1 = frame_result['bbox0'], frame_result['bbox1']
    gt_bbox0, gt_bbox1 = frame_result['gt_bbox0'], frame_result['gt_bbox1']
    bbox1, gt_bbox1 = BBox.array2bbox(bbox1), BBox.array2bbox(gt_bbox1)
    visualizer.handler_box(bbox1, color='light_blue')
    visualizer.handler_box(gt_bbox1, color='red')
    vis_pc = utils.pc_in_box_2D(gt_bbox1, pc, 4.0)
    visualizer.handler_pc(vis_pc)
    visualizer.show()
    visualizer.close()


def tracker_api(configs, id, start_bbox, start_frame, data_loader, track_len, gts=None, visualize=False):
    """ api for the tracker
    Args:
        configs: model configuration read from config.yaml
        id (str): each tracklet has an id
        start_bbox ([x, y, z, yaw, l, w, h]): the beginning location of this id
        data_loader (an iterator): iterator returning data of each incoming frame
    Return:
        {
            frame_number0: pred_bbox0,
            frame_number1: pred_bbox1,
            ...
            frame_numberN: pred_bboxN
        }
    """
    mean_loc_error = 0
    mean_heading_error = 0
    
    tracker = sot_3d.Tracker(id=id, configs=configs, start_bbox=start_bbox, start_frame=start_frame, track_len=track_len)
    tracklet_result = dict()
    for frame_index in range(track_len):
        print('////////////////////////////////////////')
        print('Processing {:} {:} / {:}'.format(id, frame_index + 1, track_len))
        # initialize a tracker
        frame_data = next(data_loader)
        # if the first frame, add the start_bbox
        input_bbox = None        
        
        if frame_index == 0:
            input_bbox = BBox.bbox2world(frame_data['ego'], BBox.array2bbox(start_bbox))
        
        input_data = sot_3d.FrameData(ego_info=frame_data['ego'], pc=frame_data['pc'], start_bbox=input_bbox,
            terrain=frame_data['terrain'], dets=frame_data['dets'])
        
        # run the frame level tracking
        frame_output = tracker.track(input_data)
        # the frame 0 may produce no output
        if not frame_output:
            continue

        # if gt is not None, we may compare our prediction with gt
        frame_result = compare_to_gt(frame_index, frame_output, gts)
        max_frame_key = max(list(frame_result.keys()))
        for i in range(max_frame_key + 1):
            print('BBox0    : {:}'.format(frame_result[i]['bbox0']))
            print('BBox1    : {:}'.format(frame_result[i]['bbox1']))
            print('Motion   : {:}'.format(frame_result[i]['motion']))
            if gts:
                print('GT BBox0 : {:}'.format(frame_result[i]['gt_bbox0']))
                print('GT BBox1 : {:}'.format(frame_result[i]['gt_bbox1']))
                print('GT Motion: {:}'.format(frame_result[i]['gt_motion']))
                print('IOUS     : {:}  {:}'.format(frame_result[i]['iou2d'], frame_result[i]['iou3d']))

                
                mean_loc_error = (mean_loc_error * frame_index + frame_result[i]['loc_error'])/(frame_index + 1)
                mean_heading_error = (mean_heading_error * frame_index + frame_result[i]['heading_error'])/(frame_index + 1)
                
                
                print('LOC ERROR: {:}'.format(frame_result[i]['loc_error']))
                print('Heading ERROR: {:}'.format(frame_result[i]['heading_error']))
                print('LOC ERROR Mean: {:}'.format(mean_loc_error))
                print('Heading ERROR Mean: {:}'.format(mean_heading_error))
                
                
            print('\n')
        tracklet_result[frame_index + start_frame] = frame_result[max_frame_key]

        if visualize and frame_index%20==0:
            frame_result_visualization(frame_result[max_frame_key], tracker.input_data.pc)
        
    return tracklet_result



def myDataLoader():
    for i in range(200):
        name_str = "{:0>4d}".format(i+5)
        file_path_str = "ourdata/segmented_cloud/data_" + data_index + "/"
        
        pcd = o3d.io.read_point_cloud(file_path_str + name_str + ".pcd")
        
        print(name_str)
        
        points = np.asarray(pcd.points)
        
        ego_matrix = np.eye(4)
        
        yield {
            'ego': ego_matrix,
            'pc': points,  # 示例点云数据
            'dets': None,  # 示例检测结果
            'terrain': None  # 示例地面
        }
        
def load_gts():
    gts = []
    
    num = 0
    
    start_bbox = None
    
    with open("ourdata/gt/gt_" + data_index + ".txt", 'r') as file:
        for line in file:
            num = num +1
                

            if num > 200:
                break
            
            # 将每行的数据转换为浮点数列表
            bbox_line = list(map(float, line.strip().split()))
            
            gt_bbox = np.zeros(7)
            
            gt_bbox[0]= bbox_line[1]
            gt_bbox[1]= bbox_line[2]
            gt_bbox[2]= bbox_line[3]
            
            gt_bbox[3]= bbox_line[6]
            
            # 4.602, 1.90, 1.645
            gt_bbox[4]= 4.602
            gt_bbox[5]= 1.90
            gt_bbox[6]= 1.645
                        
            if num == 1:
                start_bbox = gt_bbox
            
            bbox = BBox.array2bbox(gt_bbox)
            
            gts.append(bbox)
            
    return gts,start_bbox
        
        

if __name__ == '__main__':
    if not os.path.exists(os.path.join(args.result_folder, args.name)):
        os.makedirs(os.path.join(args.result_folder, args.name))
    result_folder = os.path.join(args.result_folder, args.name)
    summary_folder = os.path.join(result_folder, 'summary')
    if not os.path.exists(os.path.join(summary_folder)):
        os.makedirs(summary_folder)

    ## 准备api的各项输入

    # id (没啥用)
    id= "99"
    
    # 表示运行的是哪一组数据 1~2
    data_index = "1"
    
    # config
    f = open("config.yaml", 'r')
    configs = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    
    # dataloader [pointcloud, ego]
    data_loader = myDataLoader()
    
    
    # 包围盒的真值数据和初始包围盒真值
    gts, start_bbox = load_gts()
    
    # run the tracker
    # visualize决定是否可视化结果
    tracker_api(configs, id, start_bbox, 0, data_loader, 200, gts=gts, visualize=True)