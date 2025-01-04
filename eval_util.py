"""
Copied from MMAction2
https://github.com/open-mmlab/mmaction2/blob/master/mmaction/core/evaluation/eval_detection.py
"""
import numpy as np
import sys
import pdb


def box_iou(boxes1, boxes2):
    areas1 = boxes1[1] - boxes1[0]
    areas2 = boxes2[1] - boxes2[0]

    left = np.maximum(boxes1[0], boxes2[0])
    right = np.minimum(boxes1[1], boxes2[1])

    inter = np.clip(right - left, 0, None)
    union = areas1 + areas2 - inter

    iou = inter / union

    return iou


def CalculateAveragePrecision(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1+i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])

    return ap


def compute_vdsfx_sfx_map(gt_gra_id, gra_gt_qid2data, gra_pred_qid2data, rank_k, tiou_threshold):
    class_ap = []
    for gt_gra in gt_gra_id:
        dects = []
        for gra_pred in gra_pred_qid2data:
            for i, choose_id in enumerate(gra_pred["choose_list"][:rank_k]):
                if choose_id == gt_gra:
                    dects.append([
                        gra_pred["video-id"],
                        gra_pred["score_list"][i],
                        gra_pred["t-start"],
                        gra_pred["t-end"]
                        ])
                    break
        if len(dects) == 0:
            class_ap.append(0)
            continue

        gts = {}
        npos = 0
        for gra_gt in gra_gt_qid2data:
            if gra_gt["id_gt"] == gt_gra:
                npos += 1
                if gra_gt["video-id"] not in gts.keys():
                    gts[gra_gt["video-id"]] = []
                gts[gra_gt["video-id"]].append([
                    gra_gt["video-id"],
                    gra_gt["t-start"],
                    gra_gt["t-end"]
                ])
        dects = sorted(dects, key=lambda conf: conf[1], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with amount of gts for each image
        det = {}
        for key in gts.keys():
            det[key] = np.zeros(len(gts[key]))
        # Loop through detections
        for d in range(len(dects)):
            if dects[d][0] in gts.keys():
                gt = gts[dects[d][0]]
            else:
                gt = []
            iouMax = sys.float_info.min
            for j in range(len(gt)):
                boxes1 = [dects[d][2], dects[d][3]]
                boxes2 = [gt[j][1], gt[j][2]]
                iou = box_iou(boxes1, boxes2)
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            # Assign detection as true positive/don't care/false positive
            if iouMax >= tiou_threshold:
                if det[dects[d][0]][jmax] == 0:
                    TP[d] = 1  # count as true positive
                    det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                else:
                    FP[d] = 1  # count as false positive
            else:
                FP[d] = 1  # count as false positive
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        ap = CalculateAveragePrecision(rec, prec)
        class_ap.append(ap)
        
    return np.mean(np.array(class_ap))


def compute_vdsfx_vid_map(qid_list, gra_gt_qid2data, gra_pred_qid2data, rank_k, tiou_threshold):
    qid_ap = []
    for qid in qid_list:
        dects = []
        for gra_pred in gra_pred_qid2data:
            if gra_pred["video-id"] == qid and 'sfx0' not in gra_pred["choose_list"][:rank_k]:
                dects.append([
                             gra_pred["choose_list"][0],
                             gra_pred["score_list"][0],
                             gra_pred["t-start"],
                             gra_pred["t-end"],
                            ])
        if len(dects) == 0:
            qid_ap.append(0)
            continue
        
        gts = []
        for gra_gt in gra_gt_qid2data:
            if gra_gt["video-id"] == qid:
                gts.append([
                    gra_gt["id_gt"],
                    gra_gt["t-start"],
                    gra_gt["t-end"]
                ])
        
        dects = sorted(dects, key=lambda conf: conf[1], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        det = np.zeros(len(gts))

        for d in range(len(dects)):
            iouMax = sys.float_info.min
            for j in range(len(gts)):
                if dects[d][0] == gts[j][0]:
                    boxes1 = [dects[d][2], dects[d][3]]
                    boxes2 = [gts[j][1], gts[j][2]]
                    iou = box_iou(boxes1, boxes2)
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j
            if iouMax >= tiou_threshold:
                if det[jmax] == 0:
                    TP[d] = 1
                    det[jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / len(gts)
        prec = np.divide(acc_TP, (acc_FP + acc_TP))   
        ap = CalculateAveragePrecision(rec, prec)
        qid_ap.append(ap)
        
    return np.mean(np.array(qid_ap))


def compute_kmd_vid_map(qid_list, gra_gt_qid2data, gra_pred_qid2data, tiou_threshold):
    qid_ap = []
    for qid in qid_list:
        dects = []
        for gra_pred in gra_pred_qid2data:
            if gra_pred["video-id"] == qid:
                dects.append([
                             gra_pred["score"],
                             gra_pred["t-start"],
                             gra_pred["t-end"],
                            ])
        if len(dects) == 0:
            qid_ap.append(0)
            continue
        
        gts = []
        for gra_gt in gra_gt_qid2data:
            if gra_gt["video-id"] == qid:
                gts.append([
                    gra_gt["t-start"],
                    gra_gt["t-end"]
                ])
        
        dects = sorted(dects, key=lambda conf: conf[0], reverse=True)
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        det = np.zeros(len(gts))

        for d in range(len(dects)):
            iouMax = sys.float_info.min
            for j in range(len(gts)):
                boxes1 = [dects[d][1], dects[d][2]]
                boxes2 = [gts[j][0], gts[j][1]]
                iou = box_iou(boxes1, boxes2)
                if iou > iouMax:
                    iouMax = iou
                    jmax = j
            if iouMax >= tiou_threshold:
                if det[jmax] == 0:
                    TP[d] = 1
                    det[jmax] = 1
                else:
                    FP[d] = 1
            else:
                FP[d] = 1
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / len(gts)
        prec = np.divide(acc_TP, (acc_FP + acc_TP))   
        ap = CalculateAveragePrecision(rec, prec)
        qid_ap.append(ap)
        
    return np.mean(np.array(qid_ap))


def compute_vdsfx_sfx(submission, ground_truth, rank_list, iou_thds=np.array([0.5, 0.75])):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    moment_pred_qid2data = []
    for d in submission:
        for i in range(len(d["moment_pred_windows"])):
            moment_pred_qid2data.append({
                "video-id": d["qid"],
                "t-start": d["moment_pred_windows"][i][0],
                "t-end": d["moment_pred_windows"][i][1],
                "choose_list": d["matched_sfx"][i],
                "score_list": d["moment_score"][i]
            })

    moment_gt_qid2data = []
    gt_sfx_id = []
    for d in ground_truth:
        for i in range(len(d["moment_gt_windows"])):
            gt_sfx_id.append(d["gt_sfx"][i])
            moment_gt_qid2data.append({
                "video-id": d["qid"],
                "t-start": d["moment_gt_windows"][i][0],
                "t-end": d["moment_gt_windows"][i][1],
                "id_gt": d["gt_sfx"][i]
            })

    gt_sfx_id = list(set(gt_sfx_id))
    vdsfx_sfx_map = np.zeros((len(iou_thds), len(rank_list)))
    for i in range(len(iou_thds)):
        tiou_threshold = iou_thds[i]
        for j, rank_k in enumerate(rank_list):
            vdsfx_sfx_map[i][j] = compute_vdsfx_sfx_map(gt_sfx_id, moment_gt_qid2data, moment_pred_qid2data, rank_k, tiou_threshold)

    return vdsfx_sfx_map


def compute_vdsfx_vid(submission, ground_truth, rank_list, iou_thds=np.array([0.5, 0.75])):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    moment_pred_qid2data = []
    for d in submission:
        for i in range(len(d["moment_pred_windows"])):
            moment_pred_qid2data.append({
                "video-id": d["qid"],
                "t-start": d["moment_pred_windows"][i][0],
                "t-end": d["moment_pred_windows"][i][1],
                "choose_list": d["matched_sfx"][i],
                "score_list": d["moment_score"][i]
            })

    moment_gt_qid2data = []
    qid_list = []
    for d in ground_truth:
        for i in range(len(d["moment_gt_windows"])):
            qid_list.append(d["qid"])
            moment_gt_qid2data.append({
                "video-id": d["qid"],
                "t-start": d["moment_gt_windows"][i][0],
                "t-end": d["moment_gt_windows"][i][1],
                "id_gt": d["gt_sfx"][i]
            })

    qid_list = list(set(qid_list))
    vdsfx_vid_map = np.zeros((len(iou_thds), len(rank_list)))
    for i in range(len(iou_thds)):
        tiou_threshold = iou_thds[i]
        for j, rank_k in enumerate(rank_list):
            vdsfx_vid_map[i][j] = compute_vdsfx_vid_map(qid_list, moment_gt_qid2data, moment_pred_qid2data, rank_k, tiou_threshold)

    return vdsfx_vid_map


def compute_kmd_vid(submission, ground_truth, iou_thds=np.array([0.5, 0.75])):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    moment_pred_qid2data = []
    for d in submission:
        for i in range(len(d["moment_pred_windows"])):
            moment_pred_qid2data.append({
                "video-id": d["qid"],
                "t-start": d["moment_pred_windows"][i][0],
                "t-end": d["moment_pred_windows"][i][1],
                "score": d["moment_pred_windows"][i][2]
            })

    moment_gt_qid2data = []
    qid_list = []
    for d in ground_truth:
        for i in range(len(d["moment_gt_windows"])):
            moment_gt_qid2data.append({
                "video-id": d["qid"],
                "t-start": d["moment_gt_windows"][i][0],
                "t-end": d["moment_gt_windows"][i][1]
            })
            qid_list.append(d["qid"])

    qid_list = list(set(qid_list))
    kmd_vid_map = np.zeros(len(iou_thds))
    for i in range(len(iou_thds)):
        tiou_threshold = iou_thds[i]
        kmd_vid_map[i] = compute_kmd_vid_map(qid_list, moment_gt_qid2data, moment_pred_qid2data, tiou_threshold)

    return kmd_vid_map


def eval_vdsfx_sfx(submission, ground_truth, rank_list):
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    shared_qids = pred_qids.intersection(gt_qids)
    submission = [e for e in submission if e["qid"] in shared_qids]
    ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]
    map_vdsfx_sfx = compute_vdsfx_sfx(submission, ground_truth, rank_list)

    return map_vdsfx_sfx


def eval_vdsfx_vid(submission, ground_truth, rank_list):
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    shared_qids = pred_qids.intersection(gt_qids)
    submission = [e for e in submission if e["qid"] in shared_qids]
    ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]
    map_vdsfx_vid = compute_vdsfx_vid(submission, ground_truth, rank_list)

    return map_vdsfx_vid


def eval_kmd_vid(submission, ground_truth):
    pred_qids = set([e["qid"] for e in submission])
    gt_qids = set([e["qid"] for e in ground_truth])
    shared_qids = pred_qids.intersection(gt_qids)
    submission = [e for e in submission if e["qid"] in shared_qids]
    ground_truth = [e for e in ground_truth if e["qid"] in shared_qids]
    map_kmd_vid = compute_kmd_vid(submission, ground_truth)

    return map_kmd_vid

