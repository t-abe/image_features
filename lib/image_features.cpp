#include "image_features.h"

using namespace std;

namespace image_features {

LBP::LBP(const int r) : r_(r)
{
}

vector<float> LBP::get_impl(const Image& image) const {
    vector<float> fv(dim(), 0);
    const cv::Mat_<uchar>& mask = image.mask;
    const cv::Mat_<uchar>& gray = image.gray;
    cv::Mat_<uchar> lbpimage = lbp(gray);
    for(int y=0; y < lbpimage.rows; ++y){
        for(int x=0; x < lbpimage.cols; ++x){
            if(!mask.empty() && mask(y, x) == 0) continue;
            fv[lbpimage(y, x)] += 1;
        }
    }
    l2normalize(fv.begin(), fv.end());
    return fv;
}

cv::Mat_<uchar> LBP::lbp(const cv::Mat_<uchar>& src) const {
    static const int anc[2*8] = {-1,-1, 0,-1, +1,-1, +1,0,
                                 +1,+1, 0,+1, -1,+1, -1,0}; // x, y, x, y, ...
    static const double sqrt2 = sqrt(2.0);
    const int& r = r_;
    cv::Mat_<uchar> lbpimage = cv::Mat_<uchar>::zeros(src.size());
    for(int y=0; y < src.rows; ++y){
        for(int x=0; x < src.cols; ++x){
            std::vector<int> transition;
            for(int i=0; i < 8; i++){
                cv::Point t(anc[i*2+0], anc[i*2+1]);
                if(t.x != 0 && t.x != 0){
                    t.x *= (int)(sqrt2 * r);
                    t.y *= t.x;
                } else {
                    t.x *= r;
                    t.y *= r;
                }
                t.x += x;
                t.y += y;
                if(0 <= t.x && t.x < src.cols && 0 <= t.y && t.y < src.rows){
                    transition.push_back(src(t) > src(y, x) ? 1 : 0);
                } else {
                    transition.push_back(i==0 ? 0 : transition[i-1]);
                }
            }
            for(int i=0; i < transition.size(); i++){
                lbpimage(y, x) <<= 1;
                lbpimage(y, x) += transition[i];
            }
        }
    }
    return lbpimage;
}


} // image_features
