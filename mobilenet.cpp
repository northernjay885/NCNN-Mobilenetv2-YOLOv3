#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"
//#include "mat.h"
#include "platform.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_yolov3(const cv::Mat& image, std::vector<Object>& objects)
{
    const float probThreshold {0.5};
    //load model and pretrained weights
    ncnn::Net net;
    net.load_param("mobilenetv2_yolov3.param");
    net.load_model("mobilenetv2_yolov3.bin");

    const int input_size = 352;

    int img_width = image.cols;
    int img_height = image.rows;

    ncnn::Mat ncnnImage = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_width, img_height, input_size, input_size);

    //subtract range mean value, norm to (-1, 1)
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
    ncnnImage.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(4);

    ex.input("data", ncnnImage);

    ncnn::Mat detectRes;
    ex.extract("detection_out", detectRes);

    objects.clear();

    for(int i = 0; i < detectRes.h; i++)
    {
        const float* values = detectRes.row(i);
        
        if(values[1] >= probThreshold)
        {
            Object object;
            object.label = values[0];
            object.prob = values[1];
            object.rect.x = values[2] * img_width;
            object.rect.y = values[3] * img_height;
            object.rect.width = values[4] * img_width - object.rect.x;
            object.rect.height = values[5] * img_height - object.rect.y;

            objects.push_back(object);
        }

    }

    return 0;

}

static void draw_objects(const cv::Mat& inputImage, const std::vector<Object>& objects)
{
    static const char* class_name[] = {"background",
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"};
    
    cv::Mat image = inputImage.clone();

    for(size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        cv::rectangle(image, obj.rect, cv::Scalar(50,205,50));

        std::string text {""};
        text += class_name[obj.label];
        text += std::to_string(obj.prob * 100);
        text += "%";

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, &baseline);

        int x = obj.rect.x;
        int y = obj.rect.y;
        if(y < 0)
            y = 0;
        if(x + textSize.width > image.cols)
            x = image.cols - textSize.width;
        
        //y + textSize.height for bottom left corner
        cv::rectangle(image, cv::Rect(x, y, textSize.width, textSize.height + 5), cv::Scalar(50,205,50), -1);
        cv::putText(image, text, cv::Point(x, y + textSize.height), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255,255,255));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}


int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr<<"Usage: "<<argv[0]<<" [imagepath]\n";
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat image = cv::imread(imagepath, cv::IMREAD_COLOR);
    if(image.empty())
    {
        std::cerr<<"no image has been read!"<<std::endl;
        return -1;
    }

    //detection
    std::vector<Object> objects;
    detect_yolov3(image, objects);

    //draw box
    draw_objects(image, objects);

    return 0;

}