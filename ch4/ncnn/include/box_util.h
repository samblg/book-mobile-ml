#ifndef BBOX_UTIL_H_
#define BBOX_UTIL_H_

#include "facebox.h"


#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <utility>
#include <vector>

typedef std::map<int, std::vector<NormalizedBBox> > LabelBBox;

// Compute bbox size.
float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true);
float BBoxSize(const float* bbox, const bool normalized = true);

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
    const bool normalized = true);

bool SortScorePairDescend(const std::pair<float, int>& pair1,
    const std::pair<float, int>& pair2);

bool SortScorePairDescend2(const std::pair<float, std::pair<int, int>>& pair1,
    const std::pair<float, std::pair<int, int>>& pair2);

void ApplyNMSFast(std::vector<NormalizedBBox>& bboxes,
    const std::vector<float>& scores, const float score_threshold,
    const float nms_threshold, const float eta, const int top_k,
    std::vector<int>* indices);

void DecodeBBoxesAll(const std::vector<LabelBBox>& all_loc_preds,
    const std::vector<NormalizedBBox>& prior_bboxes,
    const std::vector<std::vector<float> >& prior_variances,
    const int num, const bool share_location,
    const int num_loc_classes, const int background_label_id,
    const int code_type, const bool variance_encoded_in_target,
    const bool clip, std::vector<LabelBBox>* all_decode_bboxes);

void GetLocPredictions(const float* loc_data, const int num,
    const int num_preds_per_class, const int num_loc_classes,
    const bool share_location, std::vector<LabelBBox>* loc_preds);

void GetConfidenceScores(const float* conf_data, const int num,
    const int num_preds_per_class, const int num_classes,
    std::vector<std::map<int, std::vector<float> > >* conf_preds);

void GetConfidenceScores(const float* conf_data, const int num,
    const int num_preds_per_class, const int num_classes,
    const bool class_major, std::vector<std::map<int, std::vector<float> > >* conf_preds);

void GetPriorBBoxes(const float* prior_data, const int num_priors,
    std::vector<NormalizedBBox>* prior_bboxes,
    std::vector<std::vector<float> >* prior_variances);


#endif