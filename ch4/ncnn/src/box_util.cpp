#include "box_util.h"
#include <algorithm>
#include <iostream>
#include <fstream>

const int PriorBoxParameter_CodeType_CORNER = 0;
const int PriorBoxParameter_CodeType_CENTER_SIZE = 1;
const int PriorBoxParameter_CodeType_CORNER_SIZE = 2;

bool SortScorePairDescend(const std::pair<float, int>& pair1,
    const std::pair<float, int>& pair2) {
    return pair1.first > pair2.first;
}

bool SortScorePairDescend2(const std::pair<float, std::pair<int, int>>& pair1,
    const std::pair<float, std::pair<int, int>>& pair2) {
    return pair1.first > pair2.first;
}

float BBoxSize(const NormalizedBBox& bbox, const bool normalized) {
    if (bbox.xmax < bbox.xmin || bbox.ymax < bbox.ymin) {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0;
    }
    else {
        float width = bbox.xmax - bbox.xmin;
        float height = bbox.ymax - bbox.ymin;
        if (normalized) {
            return width * height;
        }
        else {
            // If bbox is not within range [0, 1].
            return (width + 1) * (height + 1);
        }
    }
}

float BBoxSize(const float* bbox, const bool normalized) {
    if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0;
    }
    else {
        const float width = bbox[2] - bbox[0];
        const float height = bbox[3] - bbox[1];
        if (normalized) {
            return width * height;
        }
        else {
            // If bbox is not within range [0, 1].
            return (width + 1) * (height + 1);
        }
    }
}

void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold,
    const int top_k, std::vector<std::pair<float, int> >* score_index_vec) {
    // Generate index score pairs.
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            score_index_vec->push_back(std::make_pair(scores[i], i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
        SortScorePairDescend);

    // Keep top_k scores if needed.
    if (top_k > -1 && top_k < score_index_vec->size()) {
        score_index_vec->resize(top_k);
    }
}

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
    NormalizedBBox* intersect_bbox) {
    if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
        bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
        // Return [0, 0, 0, 0] if there is no intersection.
        intersect_bbox->xmin = 0;
        intersect_bbox->ymin = 0;
        intersect_bbox->xmax = 0;
        intersect_bbox->ymax = 0;
    }
    else {
        intersect_bbox->xmin = (std::max(bbox1.xmin, bbox2.xmin));
        intersect_bbox->ymin = (std::max(bbox1.ymin, bbox2.ymin));
        intersect_bbox->xmax = (std::min(bbox1.xmax, bbox2.xmax));
        intersect_bbox->ymax = (std::min(bbox1.ymax, bbox2.ymax));
    }
}

float JaccardOverlap(const float* bbox1, const float* bbox2) {
    if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] ||
        bbox2[1] > bbox1[3] || bbox2[3] < bbox1[1]) {
        return 0;
    }
    else {
        const float inter_xmin = std::max(bbox1[0], bbox2[0]);
        const float inter_ymin = std::max(bbox1[1], bbox2[1]);
        const float inter_xmax = std::min(bbox1[2], bbox2[2]);
        const float inter_ymax = std::min(bbox1[3], bbox2[3]);

        const float inter_width = inter_xmax - inter_xmin;
        const float inter_height = inter_ymax - inter_ymin;
        const float inter_size = inter_width * inter_height;

        const float bbox1_size = BBoxSize(bbox1);
        const float bbox2_size = BBoxSize(bbox2);

        return inter_size / (bbox1_size + bbox2_size - inter_size);
    }
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
    const bool normalized) {
    NormalizedBBox intersect_bbox;
    IntersectBBox(bbox1, bbox2, &intersect_bbox);
    float intersect_width, intersect_height;
    if (normalized) {
        intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
        intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
    }
    else {
        intersect_width = intersect_bbox.xmax - intersect_bbox.xmin + 1;
        intersect_height = intersect_bbox.ymax - intersect_bbox.ymin + 1;
    }
    if (intersect_width > 0 && intersect_height > 0) {
        float intersect_size = intersect_width * intersect_height;
        float bbox1_size = BBoxSize(bbox1);
        float bbox2_size = BBoxSize(bbox2);
        return intersect_size / (bbox1_size + bbox2_size - intersect_size);
    }
    else {
        return 0.;
    }
}

void ApplyNMSFast(std::vector<NormalizedBBox>& bboxes,
    const std::vector<float>& scores, const float score_threshold,
    const float nms_threshold, const float eta, const int top_k,
    std::vector<int>* indices) 
{
    // Sanity check.
    //CHECK_EQ(bboxes.size(), scores.size())
       // << "bboxes and scores have different size.";

    //// Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int> > score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);
    //// Do nms.
    float adaptive_threshold = nms_threshold;
    indices->clear();
    while (score_index_vec.size() != 0) {
        const int idx = score_index_vec.front().second;
        bool keep = true;
        for (int k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            }
            else {
                break;
            }
        }
        if (keep) {
            indices->push_back(idx);
        }
        score_index_vec.erase(score_index_vec.begin());
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
        }
    }
}

void DecodeBBox(
    const NormalizedBBox& prior_bbox, const std::vector<float>& prior_variance,
    const int code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const NormalizedBBox& bbox,
    NormalizedBBox* decode_bbox) {
    if (code_type == PriorBoxParameter_CodeType_CORNER) {
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            decode_bbox->xmin = (prior_bbox.xmin + bbox.xmin);
            decode_bbox->ymin = (prior_bbox.ymin + bbox.ymin);
            decode_bbox->xmax = (prior_bbox.xmax + bbox.xmax);
            decode_bbox->ymax = (prior_bbox.ymax + bbox.ymax);
        }
        else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox->xmin = (
                prior_bbox.xmin + prior_variance[0] * bbox.xmin);
            decode_bbox->ymin = (
                prior_bbox.ymin + prior_variance[1] * bbox.ymin);
            decode_bbox->xmax = (
                prior_bbox.xmax + prior_variance[2] * bbox.xmax);
            decode_bbox->ymax = (
                prior_bbox.ymax + prior_variance[3] * bbox.ymax);
        }
    }
    else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
        float prior_width = prior_bbox.xmax - prior_bbox.xmin;
       // CHECK_GT(prior_width, 0);
        float prior_height = prior_bbox.ymax - prior_bbox.ymin;
       // CHECK_GT(prior_height, 0);
        float prior_center_x = (prior_bbox.xmin + prior_bbox.xmax) / 2.;
        float prior_center_y = (prior_bbox.ymin + prior_bbox.ymax) / 2.;

        float decode_bbox_center_x, decode_bbox_center_y;
        float decode_bbox_width, decode_bbox_height;
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to retore the offset
            // predictions.
            decode_bbox_center_x = bbox.xmin * prior_width + prior_center_x;
            decode_bbox_center_y = bbox.ymin * prior_height + prior_center_y;
            decode_bbox_width = exp(bbox.xmax) * prior_width;
            decode_bbox_height = exp(bbox.ymax) * prior_height;
        }
        else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox_center_x =
                prior_variance[0] * bbox.xmin * prior_width + prior_center_x;
            decode_bbox_center_y =
                prior_variance[1] * bbox.ymin * prior_height + prior_center_y;
            decode_bbox_width =
                exp(prior_variance[2] * bbox.xmax) * prior_width;
            decode_bbox_height =
                exp(prior_variance[3] * bbox.ymax) * prior_height;
        }

        decode_bbox->xmin = (decode_bbox_center_x - decode_bbox_width / 2.);
        decode_bbox->ymin = (decode_bbox_center_y - decode_bbox_height / 2.);
        decode_bbox->xmax = (decode_bbox_center_x + decode_bbox_width / 2.);
        decode_bbox->ymax = (decode_bbox_center_y + decode_bbox_height / 2.);
    }
    else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
        float prior_width = prior_bbox.xmax - prior_bbox.xmin;
        //CHECK_GT(prior_width, 0);
        float prior_height = prior_bbox.ymax - prior_bbox.ymin;
        //CHECK_GT(prior_height, 0);
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to add the offset
            // predictions.
            decode_bbox->xmin = (prior_bbox.xmin + bbox.xmin * prior_width);
            decode_bbox->ymin = (prior_bbox.ymin + bbox.ymin * prior_height);
            decode_bbox->xmax = (prior_bbox.xmax + bbox.xmax * prior_width);
            decode_bbox->ymax = (prior_bbox.ymax + bbox.ymax * prior_height);
        }
        else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox->xmin = (
                prior_bbox.xmin + prior_variance[0] * bbox.xmin * prior_width);
            decode_bbox->ymin = (
                prior_bbox.ymin + prior_variance[1] * bbox.ymin * prior_height);
            decode_bbox->xmax = (
                prior_bbox.xmax + prior_variance[2] * bbox.xmax * prior_width);
            decode_bbox->ymax = (
                prior_bbox.ymax + prior_variance[3] * bbox.ymax * prior_height);
        }
    }
    else {
        //LOG(KLOG_FATAL) << "Unknown LocLossType.";
    }
    float bbox_size = BBoxSize(*decode_bbox);
    decode_bbox->size = (bbox_size);
    /*if (clip_bbox) {
        ClipBBox(*decode_bbox, decode_bbox);
    }*/
}

void DecodeBBoxes(
    const std::vector<NormalizedBBox>& prior_bboxes,
    const std::vector<std::vector<float> >& prior_variances,
    const int code_type, const bool variance_encoded_in_target,
    const bool clip_bbox, const std::vector<NormalizedBBox>& bboxes,
    std::vector<NormalizedBBox>* decode_bboxes) {
    //CHECK_EQ(prior_bboxes.size(), prior_variances.size());
    //CHECK_EQ(prior_bboxes.size(), bboxes.size());
    int num_bboxes = prior_bboxes.size();
    if (num_bboxes >= 1) {
        //CHECK_EQ(prior_variances[0].size(), 4);
    }
    decode_bboxes->clear();
    for (int i = 0; i < num_bboxes; ++i) {
        NormalizedBBox decode_bbox;
        DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
            variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
        decode_bboxes->push_back(decode_bbox);
    }
}

void DecodeBBoxesAll(const std::vector<LabelBBox>& all_loc_preds,
    const std::vector<NormalizedBBox>& prior_bboxes,
    const std::vector<std::vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const int code_type, const bool variance_encoded_in_target,
    const bool clip, std::vector<LabelBBox>* all_decode_bboxes) {
    //CHECK_EQ(all_loc_preds.size(), num);
    all_decode_bboxes->clear();
    all_decode_bboxes->resize(num);
    for (int i = 0; i < num; ++i) {
        // Decode predictions into bboxes.
        LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
        for (int c = 0; c < num_loc_classes; ++c) {
            int label = share_location ? -1 : c;
            if (label == background_label_id) {
                // Ignore background class.
                continue;
            }
            if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
                // Something bad happened if there are no predictions for current label.
               // LOG(KLOG_FATAL) << "Could not find location predictions for label " << label;
            }
            const std::vector<NormalizedBBox>& label_loc_preds =
                all_loc_preds[i].find(label)->second;
            DecodeBBoxes(prior_bboxes, prior_variances,
                code_type, variance_encoded_in_target, clip,
                label_loc_preds, &(decode_bboxes[label]));
        }
    }
}

void GetLocPredictions(const float* loc_data, const int num,
    const int num_preds_per_class, const int num_loc_classes,
    const bool share_location, std::vector<LabelBBox>* loc_preds){
    loc_preds->clear();
    if (share_location) {
        if (num_loc_classes != 1)
            std::cout << "check num_loc_classes failed" << std::endl;
    }
    loc_preds->resize(num);
    for (int i = 0; i < num; ++i) {
        LabelBBox& label_bbox = (*loc_preds)[i];
        for (int p = 0; p < num_preds_per_class; ++p) {
            int start_idx = p * num_loc_classes * 4;
            for (int c = 0; c < num_loc_classes; ++c) {
                int label = share_location ? -1 : c;
                if (label_bbox.find(label) == label_bbox.end()) {
                    label_bbox[label].resize(num_preds_per_class);
                }
                label_bbox[label][p].xmin = (loc_data[start_idx + c * 4]);
                label_bbox[label][p].ymin = (loc_data[start_idx + c * 4 + 1]);
                label_bbox[label][p].xmax = (loc_data[start_idx + c * 4 + 2]);
                label_bbox[label][p].ymax = (loc_data[start_idx + c * 4 + 3]);
            }
        }
        loc_data += num_preds_per_class * num_loc_classes * 4;
    }
}

void GetConfidenceScores(const float* conf_data, const int num,
    const int num_preds_per_class, const int num_classes,
    std::vector<std::map<int, std::vector<float> > >* conf_preds) {
    conf_preds->clear();
    conf_preds->resize(num);
    for (int i = 0; i < num; ++i) {
        std::map<int, std::vector<float> >& label_scores = (*conf_preds)[i];
        for (int p = 0; p < num_preds_per_class; ++p) {
            int start_idx = p * num_classes;
            for (int c = 0; c < num_classes; ++c) {
                label_scores[c].push_back(conf_data[start_idx + c]);
            }
        }
        conf_data += num_preds_per_class * num_classes;
    }
}

void GetConfidenceScores(const float* conf_data, const int num,
    const int num_preds_per_class, const int num_classes,
    const bool class_major, std::vector<std::map<int, std::vector<float> > >* conf_preds) {
    conf_preds->clear();
    conf_preds->resize(num);
    for (int i = 0; i < num; ++i) {
        std::map<int, std::vector<float> >& label_scores = (*conf_preds)[i];
        if (class_major) {
            for (int c = 0; c < num_classes; ++c) {
                label_scores[c].assign(conf_data, conf_data + num_preds_per_class);
                conf_data += num_preds_per_class;
            }
        }
        else {
            for (int p = 0; p < num_preds_per_class; ++p) {
                int start_idx = p * num_classes;
                for (int c = 0; c < num_classes; ++c) {
                    label_scores[c].push_back(conf_data[start_idx + c]);
                }
            }
            conf_data += num_preds_per_class * num_classes;
        }
    }
}


void GetPriorBBoxes(const float* prior_data, const int num_priors,
    std::vector<NormalizedBBox>* prior_bboxes,
    std::vector<std::vector<float> >* prior_variances)
{
    prior_bboxes->clear();
    prior_variances->clear();
    for (int i = 0; i < num_priors; ++i) {
        int start_idx = i * 4;
        NormalizedBBox bbox;
        bbox.xmin = (prior_data[start_idx]);
        bbox.ymin = (prior_data[start_idx + 1]);
        bbox.xmax = (prior_data[start_idx + 2]);
        bbox.ymax = (prior_data[start_idx + 3]);
        float bbox_size = BBoxSize(bbox);
        bbox.size = (bbox_size);
        prior_bboxes->push_back(bbox);
    }

    for (int i = 0; i < num_priors; ++i) {
        int start_idx = (num_priors + i) * 4;
        std::vector<float> var;
        for (int j = 0; j < 4; ++j) {
            var.push_back(prior_data[start_idx + j]);
        }
        prior_variances->push_back(var);
    }

}
