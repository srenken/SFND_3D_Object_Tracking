
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Compare Distance of LidarPoint in x-direction
bool compareByDistanceX(const LidarPoint &a, const LidarPoint &b)
{
    return a.x < b.x;
}

// Get Median of X-Distance; Parameter must be a sorted vector by x-distance
double getDistanceXMedian(std::vector<LidarPoint> lidarPoints)
{
    size_t size = lidarPoints.size();

    if(size == 0)
    {
        return 0;  
    }
    else
  {
    if (size % 2 == 0)
    {
      return (lidarPoints.at(size / 2 - 1).x + lidarPoints.at(size / 2).x) / 2;
    }
    else 
    {
      return lidarPoints.at(size / 2).x;
    }
  }
}

// Get Q1 of X-Distance; Parameter must be a sorted vector by x-distance
double getDistanceXQ1(std::vector<LidarPoint> lidarPoints)
{
    size_t size = lidarPoints.size();

    if(size == 0)
    {
        return 0;  
    }
    else
  {
    if (size % 4 == 0)
    {
      return (lidarPoints.at(size / 4 - 1).x + lidarPoints.at(size / 4).x) / 2;
    }
    else 
    {
      return lidarPoints.at(size / 4).x;
    }
  }
}

// Get Q3 of X-Distance; Parameter must be a sorted vector by x-distance
double getDistanceXQ3(std::vector<LidarPoint> lidarPoints)
{
    size_t size = lidarPoints.size();

    if(size == 0)
    {
        return 0;  
    }
    else
  {
    if ((size * 3) % 4 == 0)
    {
      return (lidarPoints.at(size * 3 / 4 - 1).x + lidarPoints.at(size * 3 / 4).x) / 2;
    }
    else 
    {
      return lidarPoints.at(size * 3 / 4).x;
    }
  }
}


// Function to calculate the median of a vector
double calculateMedian(const std::vector<double>& data) {
    // Make a copy of the data to avoid modifying the original vector
    std::vector<double> sortedData = data;

    // Sort the data
    std::sort(sortedData.begin(), sortedData.end());

    // Calculate the median
    size_t size = sortedData.size();
    if (size % 2 == 0) {
        // If the size is even, take the average of the middle two values
        return (sortedData[size / 2 - 1] + sortedData[size / 2]) / 2.0;
    } else {
        // If the size is odd, return the middle value
        return sortedData[size / 2];
    }
}



// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // Temp results
    std::vector<cv::DMatch> bb_kptMatches;
    std::vector<double> match_distance;

    // Loop over all keyPoint Matches
    for(auto itrKeyPointMatches = kptMatches.begin(); itrKeyPointMatches != kptMatches.end(); ++itrKeyPointMatches)
    {
        cv::KeyPoint prevKpt = kptsPrev[itrKeyPointMatches->queryIdx];
        cv::KeyPoint currKpt = kptsCurr[itrKeyPointMatches->trainIdx];
        
        // Check if Keypoint within BoundingBox
        if (boundingBox.roi.contains(currKpt.pt))
        {
            // Add to vector of matches with BoundingBox
            bb_kptMatches.push_back(*itrKeyPointMatches);

            // Calc euclidean distance to prev Keypoint and add distance to vector match_distance
            double kpts_distance = cv::norm(currKpt.pt - prevKpt.pt);
            match_distance.push_back(kpts_distance);
        }

    }

    // robust mean (median) of distanes
    double median_distance = calculateMedian(match_distance);

    // Filter all matches, where distance is too far off the median 
    for(size_t i = 0; i < bb_kptMatches.size(); ++i)
    {
        const cv::DMatch& bb_kptMatch = bb_kptMatches[i];
        const double match_dist = match_distance[i];

        if ( match_dist < median_distance * 1.3 )
        {
            boundingBox.kptMatches.push_back(bb_kptMatch);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios

    double median_distanceRatio = calculateMedian(distRatios);

    double dT = 1 / frameRate;
    TTC = -dT / (1 - median_distanceRatio);

}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // Sort LiDAR Point Vectors by distance in X direction
    sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), compareByDistanceX);
    sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), compareByDistanceX);

    double medianXPrev = getDistanceXMedian(lidarPointsPrev);
    double medianXCurr = getDistanceXMedian(lidarPointsCurr);

    double minXPrev = 1e9, minXCurr = 1e9;
    
    double minXValuePrev = lidarPointsPrev.at(0).x;
    double minXValueCurr = lidarPointsCurr.at(0).x;

    double Q1XPrev = getDistanceXQ1(lidarPointsPrev);
    double Q1XCurr = getDistanceXQ1(lidarPointsCurr);

    double Q3XPrev = getDistanceXQ3(lidarPointsPrev);
    double Q3XCurr = getDistanceXQ3(lidarPointsCurr);

    double IQRXPrev = Q3XPrev - Q1XPrev;
    double IQRXCurr = Q3XCurr - Q1XCurr;

    double outlierXTresholdPrev = Q1XPrev - (1.5 * IQRXPrev);
    double outlierXTresholdCurr = Q1XCurr - (1.5 * IQRXCurr);

    cout << "Previous Frame -> Median X = " << medianXPrev << ";  outlier Threshold = " << outlierXTresholdPrev << ";  Min X = " << minXValuePrev << endl;
    cout << "Current Frame -> Median X = " << medianXCurr << ";  outlier Threshold = " << outlierXTresholdCurr << ";  Min X = " << minXValueCurr << endl;

    // find closest Lidar point (x-direction)
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        // only consider points above the outlier threshold (Q1 - 1.5 * IQR)
        if (it->x >= outlierXTresholdPrev)
        {
            minXPrev = minXPrev > it->x ? it->x : minXPrev;
        }
        
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        // only consider points above the outlier threshold (Q1 - 1.5 * IQR)
        if (it->x >= outlierXTresholdCurr)
        {
            minXCurr = minXCurr > it->x ? it->x : minXCurr;
        }
    }

    // compute TTC from both measurements
    TTC = minXCurr * (1 / frameRate) / (minXPrev - minXCurr);

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // BoxID_prev, BoxID_curr, count of matches for same bounding boxes
    std::map<std::pair<int, int>, int> matchCandiates;

    std::multimap<int, int> boxIDprevFrame;
    std::multimap<int, int> boxIDCurrFrame;

    std::vector<BoundingBox> prevBoundingBoxes, currBoundingBoxes;

    prevBoundingBoxes = prevFrame.boundingBoxes;
    currBoundingBoxes = currFrame.boundingBoxes;

    // Loop over all keyPoint Matches
    for(auto itrKeyPointMatches = matches.begin(); itrKeyPointMatches != matches.end(); ++itrKeyPointMatches)
    {
        int prevFrame_KeyP_Idx = itrKeyPointMatches->queryIdx;
        int currFrame_KeyP_Idx = itrKeyPointMatches->trainIdx;

        

        // Loop over BoundingBoxes in prevFrame and check if KeyPoint is contained
        for(auto itrPrevBBoxes = prevBoundingBoxes.begin(); itrPrevBBoxes != prevBoundingBoxes.end(); ++itrPrevBBoxes)
        {
            // Loop over BoundingBoxes in currFrame and check if KeyPoint is contained
            for(auto itrCurrBBoxes = currBoundingBoxes.begin(); itrCurrBBoxes != currBoundingBoxes.end(); ++itrCurrBBoxes)
            {
                // Are the matched KeyPoints part of the ROI in their respective frames?
                if( itrPrevBBoxes->roi.contains( prevFrame.keypoints.at(prevFrame_KeyP_Idx).pt) && 
                    itrCurrBBoxes->roi.contains( currFrame.keypoints.at(currFrame_KeyP_Idx).pt) )
                {

                    // Check if key exists
                    std::map<std::pair<int, int>, int>::iterator it_matchCandiates = matchCandiates.find(std::pair<int,int>(itrPrevBBoxes->boxID, itrCurrBBoxes->boxID));

                    if (it_matchCandiates != matchCandiates.end() )
                    {
                        // Increment counter
                        it_matchCandiates->second = it_matchCandiates->second + 1;
                    }
                    else
                    {
                        // Add BoxID Pair and set counter to 1
                        matchCandiates.insert(std::pair<std::pair<int, int>, int>(std::pair<int, int>(itrPrevBBoxes->boxID, itrCurrBBoxes->boxID), int(1)) );
                    }
                }
            
            }
            
        }

        
    }

    // Loop over BoundingBoxes in prevFrame, to find the BoundingBox Match Candidate with the hightest count of matching keypoints
    for(auto itrPrevBBoxes = prevBoundingBoxes.begin(); itrPrevBBoxes != prevBoundingBoxes.end(); ++itrPrevBBoxes)
    {   
        std::pair<int,int> bestMatch = make_pair(99999,99999);
        int bestMatch_count = 0;

        for (const auto &map_entry: matchCandiates)
        {
            auto key_pair = map_entry.first;
            if( key_pair.first == itrPrevBBoxes->boxID && map_entry.second > bestMatch_count)
            {
                bestMatch = key_pair;
                bestMatch_count = map_entry.second;
            }

        }

        if( bestMatch_count > 0 )
        {
            bbBestMatches.insert(bestMatch);
        }
        
    }

}
