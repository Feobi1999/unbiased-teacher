import json
import torch

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

import pdb
pdb.set_trace()
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
                    if ious[i, j] > 0.6:
                        res[img]['bbox'].append((boxes[i] + boxes[j]) / 2)
                        res[img]['score'].append((scores[i] + scores[j]) / 2)
                        res[img]['label'].append(cls)
                        picked[j] = True
                        continue
                res[img]['bbox'].append(boxes[i])
                res[img]['score'].append(scores[i] / 2)
                res[img]['label'].append(cls)
        res[img]['bbox'] = torch.stack(res[img]['bbox']).view(-1, 4)
        res[img]['label'] = torch.Tensor(res[img]['label']).view(-1)
        res[img]['score'] = torch.Tensor(res[img]['score']).view(-1)
    return res
merged_pred = merge_pred(sum_pred)
#test under different confident scores
num_classes = 1

def merge_predict(pred1,pred2,thresh=0.8):
    pred_concat=[]
    assert len(pred1) != len(pred2)
    for i in len(pred1):


        bbox1 = pred1[i]['bbox']
        score1 = pred1[i]['score']
        label1 = pred1[i]['label']

        bbox2 = pred2[i]['bbox']
        score2 = pred2[i]['score']
        label2 = pred2[i]['label']

        bbox_with_score = torch.concat((bbox1,score1),1)
        bbox_with_score2 = torch.concat((bbox2,score2),1)
        bbox_concat = torch.concat((bbox_with_score,bbox_with_score2),0)
        ''' operate nms '''
        keep_inds = nms(bbox_concat, thresh = thresh)
        post_nms_pred = bbox_concat[keep_inds]

        len = post_nms_pred.shape[0]
        print("post total bboxes:", len)
        for l in range(len):

            anno['image_id'] = i
            anno['bbox'] = post_nms_pred[:,:4]
            anno['category_id'] = 1
            anno['score'] = post_nms_pred[:,4]
            pred_concat.append(anno)

def match(pred, gt, c_th=0.05, area=[0, 9999999]):
    thresh = 0.5
    tp = []
    fn = []
    fp = []
    match_res = []
    match_score = []
    for img in gt:
        #tp.append(0)
        #fn.append(0)
        #fp.append(0)
        ##dealing
        keep_ind = pred[img]['score'] >= c_th
        pred[img]['bbox'] = pred[img]['bbox'][keep_ind]
        pred[img]['label'] = pred[img]['label'][keep_ind]
        pred[img]['score'] = pred[img]['score'][keep_ind]
        ##
        if len(gt[img]['bbox']) == 0:
            tp.append(0)
            fn.append(0)
            fp.append(len(pred[img]['bbox']))
            continue
        if len(pred[img]['bbox']) == 0:
            tp.append(0)
            fn.append(len(gt[img]['bbox']))
            fp.append(0)
            match_res.append(torch.zeros(len(gt[img]['bbox'])))
            match_score.append(torch.zeros(len(gt[img]['bbox'])))
            continue
        ious = bbox_iou(gt[img]['bbox'], pred[img]['bbox'])
        scores = torch.zeros(len(gt[img]['bbox']))
        for i in range(ious.size(0)):
            for j in range(ious.size(1)):
                if ious[i, j] >= thresh and gt[img]['label'][i] == pred[img]['label'][j]:
                    ious[i, j] = 1
                    if pred[img]['score'][j] > scores[i]:
                        scores[i] = pred[img]['score'][j]
                else:
                    ious[i, j] = 0
        match_res.append((ious.sum(dim=1) > 0).float())
        match_score.append(scores)
        tp.append((ious.sum(dim=1) > 0).sum().item())
        fn.append((ious.sum(dim=1) == 0).sum().item())
        fp.append((ious.sum(dim=0) == 0).sum().item())
    print('total ratio: %.3f'%(sum(tp) / (sum(tp) + sum(fn))))
    avg_f = lambda x: sum(x) / len(x)
    print('avg ratio: %.3f'%(avg_f([tp[i] / (tp[i] + fn[i]) for i in range(len(tp)) if tp[i] + fn[i] > 0])))
    print('fp / img: %.1f'%(sum(fp) / len(fp)))
    return torch.cat(match_res).float(), torch.cat(match_score)
'''
res_sum, score_sum = match(sum_pred, gt)
res_sum, score_sum = match(sum_pred, gt, 0.2)
res_sum, score_sum = match(sum_pred, gt, 0.3)
res_sum, score_sum = match(sum_pred, gt, 0.35)
res_sum, score_sum = match(sum_pred, gt, 0.4)
res, score = match(merged_pred, gt)
res, score = match(merged_pred, gt, 0.2)
res, score = match(merged_pred, gt, 0.3)
'''
res_sum, score_sum = match(sum_pred, gt, 0.1)
res, score = match(merged_pred, gt, 0.1)
res_sum, score_sum = match(sum_pred, gt, 0.15)
res, score = match(merged_pred, gt, 0.15)

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
                    if ious[i, j] > 0.6:
                        res[img]['bbox'].append((boxes[i] + boxes[j]) / 2)
                        res[img]['score'].append((scores[i] + scores[j]) / 2)
                        res[img]['label'].append(cls)
                        picked[j] = True
                        continue
                res[img]['bbox'].append(boxes[i])
                res[img]['score'].append(scores[i] / 2)
                res[img]['label'].append(cls)
        res[img]['bbox'] = torch.stack(res[img]['bbox']).view(-1, 4)
        res[img]['label'] = torch.Tensor(res[img]['label']).view(-1)
        res[img]['score'] = torch.Tensor(res[img]['score']).view(-1)
    return res