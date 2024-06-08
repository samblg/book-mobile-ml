#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

namespace caffe {
	using std::vector;
	using std::string;

	const float DEDUP_BOXES = 0.0625f;

	struct bbox_fu
	{
		float bbox1;  //left
		float bbox2;  //top
		float bbox3;  //right
		float bbox4;  //bottom
		float score;
	};

	struct bbox_fu_nms
	{
		float bbox1;
		float bbox2;
		float bbox3;
		float bbox4;
		float score;
		int index;
		float area;
	};

    inline bool compare_bbox_score(const bbox_fu& t1, const bbox_fu& t2)
	{
		return t1.score >= t2.score;
	}

    inline bool rindcom(const bbox_fu_nms& t1, const bbox_fu_nms& t2)
	{
		return t1.score <= t2.score;
	}

    inline bool d_rindcom(const bbox_fu_nms& t1, const bbox_fu_nms& t2)
	{
		return t1.score >= t2.score;
	}

    bool nms2(vector<bbox_fu>&src, vector<bbox_fu>&dst, float overlap);
    void _whctrs(vector<float> anchor, float& w, float& h, float& x_ctr, float& y_ctr);
    vector<vector<float> > _mkanchors(vector<float> ws, vector<float> hs, float x_ctr, float y_ctr);
    vector<vector<float> > _ratio_enum(vector<float> anchor, vector<float> ratios);
    vector<vector<float> > _scale_enum(vector<float> anchor, vector<float> scales);
    vector<vector<float> > generate_anchors(std::vector<float> scales, float base_size = 16, float ratios = 1);
	/********************** bbox_transform *********************/
    vector<bbox_fu> bbox_transform_inv(vector<vector<float> > boxes, vector<vector<float> > deltas, vector<float> scores,
        int MaxWidth, int MaxHeight, int minSize, int flag);
}
