#ifndef FACEBOX_H
#define FACEBOX_H

class NormalizedBBox
{
public:
    NormalizedBBox(){};
    ~NormalizedBBox(){};
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    int label;
    bool difficult;
    float score;
    float size;
    
};

#endif