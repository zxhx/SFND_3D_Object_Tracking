/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

// Check the string is in the vector
bool valid_input(std::vector<string> v, string key)
{
    bool error = false;
    if (find(v.begin(), v.end(), key) == v.end())
    {
		cout << key << " not found. Please try again with one of the following:" << endl;
        for (auto item : v)
            cout << item << " ,";
        error = true;
    }
    return error;
}

bool isNumber(string s) 
{ 
    for (int i = 0; i < s.length(); i++) 
        if (isdigit(s[i]) == false) 
            return false; 
  
    return true; 
} 

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // Data location
    string dataPath = "../";
    string saveImgPath = "../saved_images/";

    // Camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    
    // Image data range
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 50;   // last file index to load
    int imgStepWidth = 1;
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    /* CAMERA OBJECT DETECTION CONFIGURATION */
    // Possible object detection options
    vector<string> list_detectorType{"SHITOMASI","HARRIS","FAST","BRISK","ORB","AKAZE","SIFT"};
    vector<string> list_descriptorType{"BRISK","ORB","AKAZE","SIFT","BRIEF","FREAK"};
    vector<string> list_matcherType{"MAT_BF","MAT_FLANN"};
    vector<string> list_selectorType{"SEL_NN","SEL_KNN"};

    // Default object detection parameters
    string detectorType     = "SHITOMASI";// SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    string descriptorType   = "BRISK";    // BRISK, ORB, AKAZE, SIFT, BRIEF, FREAK
    string matcherType      = "MAT_BF";   // MAT_BF, MAT_FLANN
    string selectorType     = "SEL_NN";   // SEL_NN, SEL_KNN

    bool bWriteToCSV = false;                   // save results to CSV

    /* LIDAR CONFIGURABLE PARAMETERS */
    int minLidarPtsForBb = 100;                 // minimum number of Lidar for bounding boxes

    /* SAVE AUGMENTED IMAGES (WITH TTC RESULTS) */
    bool bSaveTTCImg = false;                   // save TTC frame
    bool bVis = false;                          // visualize results

    const bool bShowROILidarStats = true;       // show statistics of lidar points within the ROI for all frames
    const bool bShowROILidarsPts = false;       // show number of lidar points within the ROI
    const bool bVerbose = false;                // show major execution steps
    const bool bShowTimer = false;              // show timers for keypoint detection, extraction and matching per frame

    /* TOOLS FOR EVALUATING DESCRIPTORS MATCHING PERFORMANCE
       ALL BOOLEAN SHOULD BE DEFAULT TO FALSE; */
    const bool bLimitKpts = false;              // limit number of detected keypoints. Should only enable for debugging
    const int maxKeypoints = 100;               // limit number of detected keypoints (for debugging only)
    const bool bShowMatching = false;           // show image descriptor matching
    const bool bSaveMatching = false;           // save image descriptor matching
    if (bLimitKpts) cout << " NOTE: Keypoints have been limited!" << endl;

    // Input handling
    bool inputError = false;
    for (int i = 1; i < argc; ++i)
    {
        // Run default parameters if no input arguments are supplied
        if (argc == 0) break;

        /* INPUT ARGUMENTS */
        string arg = argv[i];
        if ( (arg == "--detector") || (arg == "-det") ) {
            if (i + 1 < argc)
            {
                detectorType = argv[i+1];
                inputError = valid_input(list_detectorType,detectorType);
            }
        } else if ( (arg == "--descriptor") || (arg == "-des") ) {
            if (i + 1 < argc)
            {
                descriptorType = argv[i+1];
                inputError = inputError || valid_input(list_descriptorType,descriptorType);
            }
        } else if ( (arg == "--matcher") || (arg == "-match") ) {
            if (i + 1 < argc)
            {
                matcherType = argv[i+1];
                inputError = inputError || valid_input(list_matcherType,matcherType);
            }
        } else if ( (arg == "--selector") || (arg == "-sel") ) {
            if (i + 1 < argc)
            {
                selectorType = argv[i+1];
                inputError = inputError || valid_input(list_selectorType,selectorType);
            }
        }

        /* [OPTIONAL] INPUT ARGUMENTS */
        // Check if saving augmented TTC image is required
        if ( (arg == "--saveTTC") || (arg == "-saveTTC") ) {
            bSaveTTCImg = true;
        }

        // Check if minimum number of Lidar points are specified
        else if ( (arg == "--minLidarPts") || (arg == "-minLidar") ) {
            if (i + 1 < argc)
            {
                if (isNumber(argv[i+1]))
                {
                    minLidarPtsForBb = atoi(argv[i+1]);
                } else {
                    cerr << argv[i+1] << " is not an integer. Run the program again with an integer" << endl;
                    return 1;
                }
            }
        }

        // Check if image start index is specified
        else if ( (arg == "--start") || (arg == "-start") ) {
            if (i + 1 < argc)
            {
                if (isNumber(argv[i+1]))
                {
                    imgStartIndex = atoi(argv[i+1]);
                } else {
                    cerr << argv[i+1] << " is not an integer. Run the program again with an integer" << endl;
                    return 1;
                }
            }
        }

        // Check if image end index is specified
        else if ( (arg == "--end") || (arg == "-end") ) {
            if (i + 1 < argc)
            {
                if (isNumber(argv[i+1]))
                {
                    imgEndIndex = atoi(argv[i+1]);
                } else {
                    cerr << argv[i+1] << " is not an integer. Run the program again with an integer" << endl;
                    return 1;
                }
            }
        }
        
    }

    if(inputError)
    {
        cerr << "Invalid input arguments. Please run the program again" << endl;
        return 1;
    }

    /* INIT VARIABLES AND DATA STRUCTURES */

    int match_counter = 0;
    vector<int> detections_list, matches_list;
    vector<double> t_detKeypoints, t_descKeypoints, t_matchDescriptors;
    vector<int> numLidarPtsList;

    cout << endl;
    cout << "-------------------------------------" << endl;
    cout << "Camera Object detection configuration" << endl;
    cout << "-------------------------------------" << endl;
    cout << "- Detector   = " << detectorType << endl;
    cout << "- Descriptor = " << descriptorType << endl;
    cout << "- Matcher    = " << matcherType << endl;
    cout << "- Selector   = " << selectorType << endl;
    cout << endl;

    // Parameters
    // const int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

    // Results of Lidar and Camera based TTC
    vector<double> ttcLidar_list, ttcCamera_list;

    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second [Hz] for Lidar and camera
    cout << "Processing frame index from " << imgStartIndex << " to " << imgEndIndex << " every " << imgStepWidth << " frame(s)" << endl << endl; 

    // Object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // Calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        string text = "#" + to_string(imgIndex);
        cout << "Detecting preceding vehicle in Frame " << setw(3) << text << "...";
        if (imgIndex == 0) cout << endl;

        /* LOAD IMAGE INTO BUFFER */

        // Assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // Load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // Push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        if (bVerbose) cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;  // confident threshold [0..1]
        float nmsThreshold = 0.4;   // non maximum suppression threshold [0..1]
        // YOLO object detection
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        if (bVerbose) cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // Load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // Remove Lidar points based on distance properties in [m]
        float minZ = -1.5, maxZ = -0.9; // slightly above ground surface
        float minX = 2.0, maxX = 20.0;  // front and rear of the ego car
        float maxY = 2.0;               // focus on ego lane
        float minR = 0.1;               // minimum reflectivity [0..1]; 1 = highest reflectivity
        float maxR = 1.0;               // maximum reflectivity [0..1]; 1 = highest reflectivity
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR, maxR);

        // Remove Lidar points outside the ego lane
        const double laneWidth = 4.0; // Assumed road lane width [m]
        cropLidarPointsEgoLane(lidarPoints,laneWidth);
    
        // Update the filtered Lidar points in the current frame
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        if (bVerbose) cout << "#3 : CROP LIDAR POINTS done" << endl;


        /* CLUSTER LIDAR POINT CLOUD */

        // Associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = false;
        if(bVis)
        {
            bool bWait = true;
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(1000, 2000), bWait);
        }
        bVis = false;

        if (bVerbose) cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        
        
        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
        // continue; // skips directly to the next image without processing what comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // Convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // Extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        detKeypointsModern(keypoints, imgGray, detectorType,t_detKeypoints,bShowTimer,bVis);
        detections_list.push_back(keypoints.size());

        // Optional : limit number of keypoints (helpful for debugging and learning)
        if (bLimitKpts)
        {
            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 100 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
        }

        // Push keypoints and descriptor for current frame to end of data buffer
        
        (dataBuffer.end() - 1)->keypoints = keypoints;

        if (bVerbose) cout << "#5 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */
        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, t_descKeypoints, bShowTimer);

        // Push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        if (bVerbose) cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

        // Wait until at least two images have been processed
        if (dataBuffer.size() > 1)
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            cv::Mat matchImg = (dataBuffer.end() - 1)->cameraImg.clone();
            matchDescriptors((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 1)->cameraImg, matchImg,
                             (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType, t_matchDescriptors, bShowTimer);

            // Increment match counter once keypoints matching is completed
            match_counter++;

            // Store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            if (bVerbose)  cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            /* TRACK 3D OBJECT BOUNDING BOXES */

            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            // Associate bounding boxes between current and previous frame using keypoint matches
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1));
            //// EOF STUDENT ASSIGNMENT

            // Store matches in current data frame
            // Write bounding box best matches into the currnet data buffer
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            if (bVerbose) cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // Loop over all BB match pairs
            // For each bounding box in current data frame
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
            {
                // Find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;

                // For each bounding box in current frame
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                {
                    // Check whether previous match partner corresponds to this BB 
                    // in the current frame
                    if (it1->second == it2->boxID)
                    {
                        // Pointer to a bounding box in current frame
                        currBB = &(*it2);
                        break;
                    }
                }

                // For each bounding box in previous frame
                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                {
                    // Check whether current match partner corresponds to this BB
                    // in the previous frame
                    if (it1->first == it2->boxID)
                    {
                        // Pointer to a bounding box in previous frame
                        prevBB = &(*it2);
                        break;
                    }
                }

                // Store the number of Lidar points enclosed by the bounding box
                if (currBB->lidarPoints.size() > 0) numLidarPtsList.push_back(currBB->lidarPoints.size());

                // Compute TTC for current match
                // Only compute TTC if we have sufficent number of Lidar points for robust computation
                if( currBB->lidarPoints.size() > minLidarPtsForBb && prevBB->lidarPoints.size() > minLidarPtsForBb )
                {
                    // Remove the Lidar points outliers below the bumper
                    cropLidarPointsAboveBumper(prevBB->lidarPoints);
                    cropLidarPointsAboveBumper(currBB->lidarPoints);

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> Compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar; 
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> Assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> Compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);    

                    // Keep track of number of matches found within ROI
                    matches_list.push_back(currBB->kptMatches.size());

                    cout << "found " + to_string(currBB->kptMatches.size()) + " matches within ROI" << endl;

                    // Visual the keypoint matches with `maxKeypoints`
                    if (bLimitKpts)
                    {
                        // Overlay the keypoint matches enclosed by the ROI
                        cv::Mat matchImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints, 
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints, 
                                        currBB->kptMatches, matchImg, 
                                        cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                        
                        // Show matching keypoint within region of interest
                        if (bShowMatching)
                        {
                            string windowName = "Matching keypoints inside the region of interest";
                            cv::namedWindow(windowName, 7);
                            cv::imshow(windowName, matchImg);
                            cv::waitKey(0);
                        }

                        // Save descriptors matching
                        if (bSaveMatching)
                        {
                            string text = detectorType + "_" + descriptorType + "_" + matcherType + "_" + selectorType + "_ROI_match#" + to_string(match_counter);
                            text += " - Found " + to_string(currBB->kptMatches.size()) + " matches";
                            const string matchImgName = saveImgPath + text + imgFileType;

                            // Get text size
                            cv::Size textSize = cv::getTextSize(text,cv::FONT_HERSHEY_PLAIN,3,2,0);
                            // Center text
                            cv::Point textOrg((matchImg.cols - textSize.width)/2,40);
                            // Overlay rectangle box
                            cv::rectangle(matchImg, textOrg + cv::Point(-10,textSize.height*0.2),
                                          textOrg + cv::Point(textSize.width, -textSize.height*1.2),
                                          cv::Scalar::all(0),cv::FILLED);
                            putText(matchImg, text, textOrg, cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar::all(255), 2, 8);

                            // Overlay text with black background
                            // cv::Rect rect(80, 10, 1200, 40); // [x,y,width,height]
                            // cv::rectangle(matchImg, rect, cv::Scalar(0, 0, 0),cv::FILLED);
                            // cv::putText(matchImg, text, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(255,255,255));
                            cv::imwrite(matchImgName,matchImg);
                        }

                        // Skip calculation of TTC when keypoint detectors are limited
                        // as it may not have sufficient keypoint matches to satisfy the minimum distanec requirement
                        // of the camera based calculation
                        continue;
                    }

                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    //// EOF STUDENT ASSIGNMENT

                    bVis = false;
                    cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                    if (bVis || bSaveTTCImg)
                    {
                        // Show overlay of the Lidar points over the image
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        // Show overlay of detected rectangle for the ROI
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        // Augment the image with information with time to collision
                        // computed from lidar and camera
                        string text = "TTC Lidar : " + to_string(ttcLidar) + " s, TTC Camera : " + to_string(ttcCamera) + " s" ;

                        // Get text size
                        cv::Size textSize = cv::getTextSize(text,cv::FONT_HERSHEY_PLAIN,3,2,0);
                        // Center text
                        cv::Point textOrg((visImg.cols - textSize.width)/2,40);

                        // Overlay text with black background
                        // cv::Rect rect(80, 20, 800, 40); // [x,y,width,height]
                        // cv::rectangle(visImg, rect, cv::Scalar(0,0,0), cv::FILLED);
                        cv::rectangle(visImg, textOrg + cv::Point(0,textSize.height*1.0),
                                        textOrg + cv::Point(textSize.width, -textSize.height*1.0),
                                        cv::Scalar::all(0),cv::FILLED);
                        putText(visImg, text, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255,255,255), 2);
                    }

                    // Visualize Lidar points over camera image
                    if (bVis)
                    {
                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    bVis = false;

                    if (bSaveTTCImg)
                    {
                        string ttc_ImgName = detectorType + "_" + descriptorType + "_" + matcherType + "_" + selectorType + "_" + to_string(imgIndex) + "_[TTC_Lidar=" + to_string(ttcLidar) + "s]" + "_[TTC_Camera=" + to_string(ttcCamera) + "s]";
                        ttc_ImgName += " - " + to_string(currBB->kptMatches.size()) + " matches";
                        string ttc_ImgPath = "../report/" + ttc_ImgName + imgFileType;
                        cv::imwrite(ttc_ImgPath,visImg);
                    }

                    // Store the TTC results for each frame
                    ttcLidar_list.push_back(ttcLidar);
                    ttcCamera_list.push_back(ttcCamera);

                } // EOF TTC computation
            } // EOF loop over all BB matches
        }
    } // EOF loop over all images
    
    /* LIDAR POINTS WITHIN ROI ANALYSIS */
    if (bShowROILidarStats) showROILidarStats(numLidarPtsList,bShowROILidarsPts);

    /* LIDAR AND CAMERA BASED TTC RESULTS */
    showTTC(ttcLidar_list,ttcCamera_list,sensorFrameRate);

    /* DETECTION AND MATCHING RESULTS*/
    const string cfg = detectorType + ", " + descriptorType + ", " + matcherType + ", " + selectorType;
    showDetectMatchingStats(cfg, bWriteToCSV, detections_list, matches_list, t_detKeypoints, t_descKeypoints, t_matchDescriptors);

    return 0;
}
