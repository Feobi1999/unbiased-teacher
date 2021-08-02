import json
import torch
import os
from detectron2.structures import BoxMode, RotatedBoxes, pairwise_iou_rotated
farea = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
def bbox_iou(box1, box2):
    area1 = farea(box1)
    area2 = farea(box2)
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

# load gt
gt_anno = json.load(open('/media/sda2/mzhe/datasets/Cityscapes/cocoAnnotations/cityscapes_val_caronly_cocostyle.json'))
gt = {img['id']:{'bbox':[], 'label': []} for img in gt_anno['images']}
for anno in gt_anno['annotations']:
    if anno['iscrowd'] == 1:
        continue
    gt[anno['image_id']]['bbox'].append([anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]])
    gt[anno['image_id']]['label'].append(anno['category_id'])
for img in gt:
    gt[img]['bbox'] = torch.Tensor(gt[img]['bbox']).view(-1, 4)
    gt[img]['label'] = torch.Tensor(gt[img]['label']).view(-1)

#load pred
def load_pred(fname):
    pred = {img['id']:{'bbox':[], 'label': [], 'score': []} for img in gt_anno['images']}
    pred_anno = json.load(open(fname))
    # import pdb
    # pdb.set_trace()
    for anno in pred_anno:
        pred[anno['image_id']]['bbox'].append([anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]])
        pred[anno['image_id']]['label'].append(anno['category_id'])
        pred[anno['image_id']]['score'].append(anno['score'])
    for img in pred:
        pred[img]['bbox'] = torch.Tensor(pred[img]['bbox']).view(-1, 4)
        pred[img]['label'] = torch.Tensor(pred[img]['label']).view(-1)
        pred[img]['score'] = torch.Tensor(pred[img]['score']).view(-1)
    return pred
#cs_pred = load_pred('cs_coco_mh_cshead.json')
#coco_pred = load_pred('cs_coco_mh_cocohead.json')
cs_pred = load_pred('/media/sda2/mzhe/unbiased-teacher/fft_sim10k/inference/coco_instances_results.json')
coco_pred = load_pred('/media/sda2/mzhe/unbiased-teacher/ubteacher_sim10k/inference/coco_instances_results.json')

sum_pred = {img:{'bbox': torch.cat((cs_pred[img]['bbox'], coco_pred[img]['bbox'])), 'label': torch.cat((cs_pred[img]['label'], coco_pred[img]['label'])), 'score': torch.cat((cs_pred[img]['score'], coco_pred[img]['score']))} for img in cs_pred}
def merge_pred(preds):
    res = {}
    for img in preds:
        if int(img) % 100 == 0:
            print('merging:%d'%int(img))
        res[img] = {}
        res[img]['bbox'] = []
        res[img]['score'] = []
        res[img]['label'] = []
        for cls in range(1, 9):
            cls_ind = (preds[img]['label'] == cls)
            if cls_ind.sum() == 0:
                continue
            boxes = preds[img]['bbox'][cls_ind]
            scores = preds[img]['score'][cls_ind]
            scores, sort_ind = scores.sort(descending=True)
            boxes = boxes[sort_ind]
            ious = bbox_iou(boxes, boxes)
            picked = [False] * len(ious)
            for i in range(ious.size(0)):
                if not picked[i]:
                    picked[i] = True
                else:
                    continue
                for j in range(i + 1, ious.size(0)):
                    if (ious[i, j] > 0.5) & (ious[i, j] <0.7 ):
                        res[img]['bbox'].append((boxes[i] + boxes[j]) / 2)
                        res[img]['score'].append((scores[i] + scores[j]) / 2)
                        res[img]['label'].append(cls)
                        picked[j] = True
                        continue
                res[img]['bbox'].append(boxes[i])
                res[img]['score'].append(scores[i])
                res[img]['label'].append(cls)
        res[img]['bbox'] = torch.stack(res[img]['bbox']).view(-1, 4)
        res[img]['label'] = torch.Tensor(res[img]['label']).view(-1)
        res[img]['score'] = torch.Tensor(res[img]['score']).view(-1)
    return res



def instances_to_json(instances):
    num_instance = len(instances)
    results = []
    if num_instance == 0:
        return []
    for i in range(num_instance):
        boxes = instances[i]['bbox'].numpy()
        if boxes.shape[1] == 4:
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances[i]['score'].tolist()
        classes = instances[i]['label'].tolist()

        num_boxes = len(boxes)
        for k in range(num_boxes):
            result = {
                "image_id": i,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }

            results.append(result)
    return results
merged_preds = merge_pred(sum_pred)
final_res = instances_to_json(merged_preds)
#test under different confident scores


output_dir = 'merge_prediction'
if not os.path.exists(output_dir):  #判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(output_dir)
file_path = os.path.join(output_dir, "coco_instances_results.json")

with open(file_path, "w") as f:
    f.write(json.dumps(final_res))
    f.flush()
# import pdb
# pdb.set_trace()





