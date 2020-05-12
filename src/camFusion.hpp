#ifndef camFusion_hpp
#define camFusion_hpp

#include <stdio.h>
#include <vector>
#include <iomanip>
#include <opencv2/core.hpp>
#include "dataStructures.h"


void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame);

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true);

void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> &kptMatches, double frameRate, double &TTC, cv::Mat *visImg=nullptr);
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC);

/* Helper Functions */
double avg_int_vector(std::vector<int> const& v);
double avg_double_vector(std::vector<double> const& v);
void showROILidarStats(std::vector<int> &numLidarPtsList, bool showDetailStats);
void showDetectMatchingStats(std::string cfg, bool bWriteToCSV, std::vector<int> &detections_list, std::vector<int> &matches_list, 
                            std::vector<double> &t_detKeypoints, std::vector<double> &t_descKeypoints, std::vector<double> &t_matchDescriptors);
void showTTC(std::vector<double> &ttcLidar_list, std::vector<double> &ttcCamera_list, double sensorFrameRate, double threshold_pct=0.1);

#endif /* camFusion_hpp */
