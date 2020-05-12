# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 

1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 

2. Second, you will compute the TTC based on Lidar measurements. 

3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 

4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course.  


## Dependencies for Running Locally

* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1.0
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.



## FP.1 Match 3D Objects

* **Target**:  Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the IDs of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.  

* **Implement**: matchBoundingBoxes() function implement: 

  ```c++
  void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
  {    
      int p = prevFrame.boundingBoxes.size();
      int c = currFrame.boundingBoxes.size();
      int pt_counts[p][c] = { };
      for (auto it = matches.begin(); it != matches.end() - 1; ++it)     
      {
          cv::KeyPoint query = prevFrame.keypoints[it->queryIdx];
          auto query_pt = cv::Point(query.pt.x, query.pt.y);
          bool query_found = false;
  
          cv::KeyPoint train = currFrame.keypoints[it->trainIdx];
          auto train_pt = cv::Point(train.pt.x, train.pt.y);
          bool train_found = false;
  
          std::vector<int> query_id, train_id;
          for (int i = 0; i < p; i++) 
          {
              if (prevFrame.boundingBoxes[i].roi.contains(query_pt))            
               {
                  query_found = true;
                  query_id.push_back(i);
               }
          }
          for (int i = 0; i < c; i++) 
          {
              if (currFrame.boundingBoxes[i].roi.contains(train_pt))            
              {
                  train_found= true;
                  train_id.push_back(i);
              }
          }
  
          if (query_found && train_found) 
          {
              for (auto id_prev: query_id)
                  for (auto id_curr: train_id)
                       pt_counts[id_prev][id_curr] += 1;
          }
      }
     
      for (int i = 0; i < p; i++)
      {  
           int max_count = 0;
           int id_max = 0;
           for (int j = 0; j < c; j++)
               if (pt_counts[i][j] > max_count)
               {  
                    max_count = pt_counts[i][j];
                    id_max = j;
               }
            bbBestMatches[i] = id_max;
      } 
  }
  ```

  

## FP.2 Compute Lidar-based TTC

* **Target**: Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

* **Implement**: In order to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors, here only consider Lidar points within ego lane, then get the mean distance to get stable output.

  ```c++
  void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                       std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
  {		     
      int lane_wide = 4;
      std::vector<float> ppx;
      std::vector<float> pcx;
      for(auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
      {
          if(abs(it->y) < lane_wide/2) 
              ppx.push_back(it->x);
      }
      for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
      {
          if(abs(it->y) < lane_wide/2) 
              pcx.push_back(it->x);
      }
  
      float min_px, min_cx;
      int p_size = ppx.size();
      int c_size = pcx.size();
      if(p_size > 0 && c_size > 0)
      {
          for(int i=0; i<p_size; i++)
          {
              min_px += ppx[i];
          }
  
          for(int j=0; j<c_size; j++)
          {
              min_cx += pcx[j];
          }
      }
      else 
      {
          TTC = NAN;
          return;
      }
  
      min_px = min_px /p_size;
      min_cx = min_cx /c_size;
      std::cout<<"lidar_min_px:"<<min_px<<std::endl;
      std::cout<<"lidar_min_cx:"<<min_cx<<std::endl;
  
      float dt = 1/frameRate;
      TTC = min_cx * dt / (min_px - min_cx);
  }
  ```

## FP.3 Associate Keypoint Correspondences with Bounding Boxes

* **Target**: Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

* **Implement**: 

  ```c++
  void clusterKptMatchesWithROI(BoundingBox &boundingBox_c, BoundingBox &boundingBox_p, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
  {
      double dist_mean = 0;
      std::vector<cv::DMatch>  kptMatches_roi;
  
      float shrinkFactor = 0.15;
      cv::Rect smallerBox_c, smallerBox_p;
      // shrink
      smallerBox_c.x = boundingBox_c.roi.x + shrinkFactor * boundingBox_c.roi.width / 2.0;
      smallerBox_c.y = boundingBox_c.roi.y + shrinkFactor * boundingBox_c.roi.height / 2.0;
      smallerBox_c.width = boundingBox_c.roi.width * (1 - shrinkFactor);
      smallerBox_c.height = boundingBox_c.roi.height * (1 - shrinkFactor);
  
      smallerBox_p.x = boundingBox_p.roi.x + shrinkFactor * boundingBox_p.roi.width / 2.0;
      smallerBox_p.y = boundingBox_p.roi.y + shrinkFactor * boundingBox_p.roi.height / 2.0;
      smallerBox_p.width = boundingBox_p.roi.width * (1 - shrinkFactor);
      smallerBox_p.height = boundingBox_p.roi.height * (1 - shrinkFactor);
  
      //get the matches within curr_boundingBox and pre_boundingBox
      for(auto it=kptMatches.begin(); it != kptMatches.end(); ++ it)
      {
          cv::KeyPoint train = kptsCurr.at(it->trainIdx);
          auto train_pt = cv::Point(train.pt.x, train.pt.y);
  
          cv::KeyPoint query = kptsPrev.at(it->queryIdx); 
          auto query_pt = cv::Point(query.pt.x, query.pt.y);
  
          // check wether point is within current bounding box
          if (smallerBox_c.contains(train_pt) && smallerBox_p.contains(query_pt)) 
              kptMatches_roi.push_back(*it);           
      }
  
      //caculate the mean distance of all the matches within boundingBox 
      for(auto it=kptMatches_roi.begin(); it != kptMatches_roi.end(); ++ it)
      {
          dist_mean += cv::norm(kptsCurr.at(it->trainIdx).pt - kptsPrev.at(it->queryIdx).pt); 
      }
      if(kptMatches_roi.size() > 0)
          dist_mean = dist_mean/kptMatches_roi.size();
      else return;
  
      //keep the matches  distance < dist_mean * 1.5
      double threshold = dist_mean*1.5;
      for  (auto it = kptMatches_roi.begin(); it != kptMatches_roi.end(); ++it)
      {
         float dist = cv::norm(kptsCurr.at(it->trainIdx).pt - kptsPrev.at(it->queryIdx).pt);
         if (dist< threshold)
             boundingBox_c.kptMatches.push_back(*it);
      }
  
      std::cout<<"curr_bbx_matches_size: "<<boundingBox_c.kptMatches.size()<<std::endl;
  }
  ```

  

## FP.4 Compute Camera-based TTC

* **Target**:  Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

* **Implement**: 

  ```c++
  // Compute time-to-collision (TTC) based on keypoint correspondences in successive images
  void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                        std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
  {
      vector<double> distRatios; 
      for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
      {
          cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
          cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
         
          for (auto it2 = it1 + 1; it2 != kptMatches.end(); ++it2)
          {  
              double minDist = 100.0; 
              cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
              cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
              // compute distances and distance ratios
              double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
              double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
              if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
              { 
                  double distRatio = distCurr / distPrev;
                  distRatios.push_back(distRatio);
              }
          }
      }  
      if (distRatios.size() == 0)
      {
          TTC = NAN;
          return;
      }
  
      std::sort(distRatios.begin(), distRatios.end());
  
      /* 
      int num_ration = distRatios.size();
      int crop_head_tail = floor(distRatios.size() / 10.0);
      double medDistRatio = 0;
      for (auto it = distRatios.begin() + crop_head_tail; it != distRatios.end() - crop_head_tail; ++it)
      {
          medDistRatio += *it;
      }
      medDistRatio = medDistRatio/(num_ration - 2*crop_head_tail);
      */
  
      long medIndex = floor(distRatios.size() / 2.0);
      double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
  
      double dT = 1 / frameRate;
      TTC = -dT / (1 - medDistRatio);
  }
  ```

## FP.5 Performance Evaluation 1

* **Target**: Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

* **Implement**: 

  Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

  TTC from Lidar is not correct because of some outliers and some unstable points from preceding vehicle's front mirrors, those need to be filtered out . Here we adapt a bigger shrinkFactor = 0.2, to get more reliable and stable lidar points. Then get a more accurate results.

  ```
  -------------------------------------
  Camera Object detection configuration
  -------------------------------------
  - Detector   = AKAZE
  - Descriptor = ORB
  - Matcher    = MAT_BF
  - Selector   = SEL_NN
  
  -----------------------------------------------------------------------------
   Time-To-Collision (TTC) frame analysis 
   TTC calculation updates every 0.1 s
  
   * indicates if the TTC is negative or it is 10% greater than the last frame
  -----------------------------------------------------------------------------
  Frame #   1 -  TTC Lidar: 12.972159s,  TTC Camera : 13.402956s   Est. error: 3%
  Frame #   2 -  TTC Lidar: 12.264038s,  TTC Camera : 14.289965s   Est. error: 15%
  Frame #   3 - *TTC Lidar: 13.916132s,  TTC Camera : 13.789529s   Est. error: 1%     <-- Frame 3 (Condition: not decreasing monotonically)
  Frame #   4 -  TTC Lidar:  7.115722s,  TTC Camera : 14.783032s   Est. error: 70%
  Frame #   5 - *TTC Lidar: 16.251088s,  TTC Camera : 16.057552s   Est. error: 1%     <-- Frame 5 (Condition: not decreasing monotonically)
  Frame #   6 -  TTC Lidar: 12.421338s,  TTC Camera : 13.890458s   Est. error: 11%
  Frame #   7 - *TTC Lidar: 34.340420s, *TTC Camera : 15.605360s   Est. error: 75%    <-- Frame 7 ((Condition: not decreasing monotonically)
  Frame #   8 -  TTC Lidar:  9.343759s,  TTC Camera : 14.255249s   Est. error: 42%
  Frame #   9 - *TTC Lidar: 18.131756s,  TTC Camera : 13.665328s   Est. error: 28%    <-- Frame 9 (Condition: not decreasing monotonically)
  Frame #  10 -  TTC Lidar: 18.031756s,  TTC Camera : 11.331379s   Est. error: 46%
  Frame #  11 -  TTC Lidar:  3.832443s,  TTC Camera : 12.367197s   Est. error: 105%
  Frame #  12 - *TTC Lidar: -10.853745s,  TTC Camera : 12.145637s   Est. error: 3561% <-- Frame 12 (Condition: negative value)
  Frame #  13 - *TTC Lidar:  9.223069s,  TTC Camera : 10.830024s   Est. error: 16%
  Frame #  14 - *TTC Lidar: 10.967763s,  TTC Camera : 10.459322s   Est. error: 5%     <-- Frame 14 (Condition: not decreasing monotonically)
  Frame #  15 -  TTC Lidar:  8.094218s,  TTC Camera :  9.797382s   Est. error: 19%
  Frame #  16 -  TTC Lidar:  3.175354s,  TTC Camera : 10.218116s   Est. error: 105%
  Frame #  17 - *TTC Lidar: -9.994236s,  TTC Camera :  8.703067s   Est. error: 2896%  <-- Frame 17 (Condition: negative value)
  Frame #  18 - *TTC Lidar:  8.309779s,  TTC Camera :  8.920069s   Est. error: 7%
  ```

  

## FP.6 Performance Evaluation 2

* **Target**: Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe the observations again and also look into potential reasons.

* **Implement**:  

  Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

  when get a robust clusterKptMatchesWithROI can get a stable TTC from Camera. if the result get unstable, It's probably the worse keypints matches.

  The TOP3 detector / descriptor combinations as the best choices for the purpose of detecting keypoints on vehicles are: 

  * SHITOMASI/BRISK
  * SHITOMASI/BRIEF
  * SHITOMASI/ORB
