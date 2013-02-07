#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <cfloat>
#include <string>

namespace image_features {

void l2normalize(std::vector<float>::iterator begin,
                 std::vector<float>::iterator end)
{
    float norm = 0;
    for(auto it=begin; it != end; ++it) norm += (*it) * (*it);
    norm = sqrt(norm);
    for(auto it=begin; it != end; ++it) *it /= norm;
}

void l1normalize(std::vector<float>::iterator begin,
                 std::vector<float>::iterator end)
{
    float norm = 0;
    for(auto it=begin; it != end; ++it) norm += *it;
    for(auto it=begin; it != end; ++it) *it /= norm;
}

struct Image {
    cv::Mat_<uchar> gray;
    cv::Mat_<cv::Vec3b> bgr;
    cv::Mat_<uchar> mask;

    Image(const cv::Mat_<uchar>& gray,
          const cv::Mat_<cv::Vec3b>& bgr,
          const cv::Mat_<uchar>& mask)
        : gray(gray), bgr(bgr), mask(mask)
    {}
    Image(const std::string image_filename,
          const std::string mask_filename = std::string())
        : gray(), bgr(), mask()
    {
        cv::Mat image = cv::imread(image_filename);
        if(image.empty()) throw std::runtime_error("Cannot open image: " + image_filename);
        if(image.channels() == 1){
            image.copyTo(gray);
        } else {
            image.copyTo(bgr);
            gray = cv::Mat_<uchar>(image.size());
            cv::cvtColor(image, gray, CV_BGR2GRAY);
        }

        if(!mask_filename.empty()){
            mask = cv::imread(mask_filename, 0);
            if(mask.empty()) throw std::runtime_error("Cannot open image: " + mask_filename);
            mask = mask > 128;
        }
    }
    ~Image() {}

    bool has_bgr() const {
        return bgr.empty();
    }

    bool has_mask() const {
        return mask.empty();
    }
};

class ImageFeature {
public:
    virtual ~ImageFeature(){}
    std::vector<float> get(const Image& image) const {
        return get_impl(image);
    }
    void train(const std::vector<Image>& images) {
        train_impl(images);
    }
    virtual int dim() const = 0;
private:
    virtual std::vector<float> get_impl(const Image& image) const = 0;
    virtual void train_impl(const std::vector<Image>& images) = 0;
};

class LBP : public ImageFeature {
    int r_;
public:
    LBP(const int r);
    ~LBP(){};
    int dim() const { return 256; }
private:
    std::vector<float> get_impl(const Image& image) const;
    void train_impl(const std::vector<Image>& images){};
    cv::Mat_<uchar> lbp (const cv::Mat_<uchar>& src) const;
};

} // image_features
