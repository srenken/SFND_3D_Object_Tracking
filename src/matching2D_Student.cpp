#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        // In case of Binary Descriptor use Hamming distance else L2_NORM (Sum of squared differences)
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2; 
       
        // Brute Force Matching
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F || descRef.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        // FLANN matching
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector< vector<cv::DMatch>> knnMatches;

        matcher->knnMatch(descSource, descRef, knnMatches, 2); // Finds the 2 best matches for each descriptor
  
        // descriptor distance ratio test

        float minDescDistRatio = 0.8;
        for(auto itr = knnMatches.begin(); itr != knnMatches.end(); ++itr)
        {
                        
            if (itr->at(0).distance < minDescDistRatio * itr->at(1).distance)
            {
                matches.push_back(itr->at(0));
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // descriptorTypes: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT

    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        int bytes = 32;
        bool use_orientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        int nfeatures = 500;
		float scaleFactor = 1.2f;
		int nlevels = 8;
		int edgeThreshold = 31;
		int firstLevel = 0;
		int WTA_K = 2;
		cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
		int patchSize = 31;
		int fastThreshold = 20;

        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);
    
    }
    else if(descriptorType.compare("FREAK") == 0)
    {

        bool  	orientationNormalized = true;
		bool  	scaleNormalized = true;
		float  	patternScale = 22.0f;
		int  	nOctaves = 4;

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);

    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
		int  	descriptor_size = 0;
		int  	descriptor_channels = 3;
		float  	threshold = 0.001f;
		int  	nOctaves = 4;
		int  	nOctaveLayers = 4;
		cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;

        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);
    }
    else if(descriptorType.compare("SIFT") == 0)
    {

        int  	nfeatures = 0;
		int  	nOctaveLayers = 3;
		double  	contrastThreshold = 0.04;
		double  	edgeThreshold = 10;
		double  	sigma = 1.6;

        extractor = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 5);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


// Detect keypoints in image using the HARRIS detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Timer Start
    double t = (double)cv::getTickCount();

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);


    // Perform non-maximum suppression to extract the strongest corners
    // in a local neighborhood around each pixel

    // define size of sliding window
    int sw_size = 6;                  // should be odd so we can center it on a pixel and have symmetry in all directions
    int sw_dist = floor(sw_size / 2); // number of pixels to left/right and top/down to investigate

    // create output image
    cv::Mat result_img = cv::Mat::zeros(img.rows, img.cols, CV_8U);

    // loop over all pixels in the corner image
    for (int r = sw_dist; r < img.rows - sw_dist - 1; r++) // rows
    {
        for (int c = sw_dist; c < img.cols - sw_dist - 1; c++) // cols
        {

            int response = (int)img.at<float>(r, c);
            //std::cout << response << std::endl;
            if (response > minResponse)
            {
                // loop over all pixels within sliding window around the current pixel 
                // -->> to find the local max value
                unsigned int max_val{0}; // keeps track of strongest response
                for (int rs = r - sw_dist; rs <= r + sw_dist; rs++)
                {
                    for (int cs = c - sw_dist; cs <= c + sw_dist; cs++)
                    {
                        // check wether max_val needs to be updated
                        unsigned int new_val = (int)img.at<float>(rs, cs);
                        max_val = max_val < new_val ? new_val : max_val;
                    }
                }

                // check wether current pixel is local maximum
                if (response == max_val)
                {
                    
                        result_img.at<unsigned int>(r, c) = max_val;
                        cv::KeyPoint keypoint((float)c, (float)r, (float)sw_size, (float)max_val);
                        keypoints.push_back(keypoint);
                    
                    
                }
            }
            
                
        }
    }

    // Timer Stop
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris Corner Detectoer detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  	  
    if (bVis)
    {
        // visualize results
        cv::Mat img_keypoints;
        cv::drawKeypoints( dst_norm_scaled, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


        std::string windowName = "Harris Corner Detector Response with Keypoints";
        cv::namedWindow(windowName, 5);
        cv::imshow(windowName, result_img);
        cv::waitKey(0);
    }



}



// Detect keypoints in image using Modern Detectors (FAST, BRISK, ORB, AKAZE, and SIFT)
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    // Timer Start
    double t = (double)cv::getTickCount();

    // Select Detector based on the set detectorType and perform detection
    if (detectorType.compare("SURF") == 0)
    {
        // Detect keypoints using SURF Detector
        int minHessian = 400;
        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
        detector->detect( img, keypoints );
    }
    else if(detectorType.compare("SIFT") == 0)
    {
        // Detect keypoints using SIFT Detector
        cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
        detector->detect(img, keypoints); 
    }
    else if(detectorType.compare("FAST") == 0)
    {
        // Detect keypoints using FAST Detector
        cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
        detector->detect(img, keypoints); 
    }
    else if(detectorType.compare("BRISK") == 0)
    {
        // Detect keypoints using BRISK Detector
        cv::Ptr<cv::BRISK> detector = cv::BRISK::create();
        detector->detect(img, keypoints); 
    }
    else if(detectorType.compare("ORB") == 0)
    {
        // Detect keypoints using ORB Detector
        cv::Ptr<cv::ORB> detector = cv::ORB::create();
        detector->detect(img, keypoints); 
    }
    else if(detectorType.compare("AKAZE") == 0)
    {
        // Detect keypoints using AKAZE Detector
        cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
        detector->detect(img, keypoints); 
    }
    else
    {
        cout << "!!! Defined detectorType invalid !!!" << endl;
        return;
    }
    

    // Timer Stop
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  	  
    if (bVis)
    {
        // visualize results
        cv::Mat img_keypoints;
        cv::drawKeypoints( img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


        std::string windowName = detectorType + " Detector Response with Keypoints";
        cv::namedWindow(windowName, 5);
        cv::imshow(windowName, img_keypoints);
        cv::waitKey(0);
    }



}