#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <dirent.h>
#include <string.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <syslog.h>
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"
#include "opencv2/opencv.hpp"

#ifndef BOOL
	#define BOOL bool
#endif

using namespace std;
using namespace cv;

// Input and output folder locations

const char * inDir = "/home/pi/faces/in";
const char * outDir = "/home/pi/faces/out";
const char * debugDir = "/home/pi/faces/debug";

// Haar Cascade file, used for Face Detection

const char *faceCascadeFilename =
  "/home/pi/facerecog/haarcascade_frontalface_alt.xml";
const char *lblCascadeFilename =
  "/home/pi/facerecog/lbpcascade_frontalface.xml";
const char *faceDataXml = "/home/pi/facerecog/facedata.xml";

// Set to 0 if you dont want images of the Eigenvectors
// saved to files (for debugging)

int SAVE_EIGENFACE_IMAGES = 1;

// You might get better recognition accuracy if you enable this

//#define USE_MAHALANOBIS_DISTANCE

// Global variables

IplImage ** faceImgArr        = 0; // array of face images
CvMat    *  personNumTruthMat = 0; // array of person numbers

// Array of person names (indexed by the person number)

vector<string> personNames;

// Default dimensions for faces in the face recognition database

int faceWidth = 120;
int faceHeight = 90;

// The number of people in the training set

int nPersons                  = 0;

// The number of training images

int nTrainFaces               = 0;

// The number of eigenvalues

int nEigens                   = 0;

// The average image

IplImage * pAvgTrainImg       = 0;

// Eigenvectors

IplImage ** eigenVectArr      = 0;

// Eigenvalues

CvMat * eigenValMat           = 0;

// Projected training faces

CvMat * projectedTrainFaceMat = 0;

CvMemStorage* storage         = 0;
static IplImage * small_img   = 0;
double scale = 3;

CascadeClassifier faceCascade;

#define PAD_FACE 40
#define PAD_FACE_2 80

// Function prototypes

int getdir(string dir, vector<string> &files);
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int findNearestNeighbor(
  float * projectedTestFace, float *pConfidence
);
bool recognizeFromFile(
  const char *inputFile,
  const char *outputFile,
  CvHaarClassifierCascade *cascade,
	float * projectedTestFace,
	CvMat *trainPersonNumMat
);
IplImage* resizeImage(
  const IplImage *origImg, int newWidth, int newHeight
);
CvRect cropRect(const CvRect rect, int w, int h);
IplImage* cropImage(const IplImage *img, const CvRect region);
CvRect detectFaceInImage(
  IplImage* img, CascadeClassifier &cascade
);
void execute();

// Get the list of files in a directory

int getdir(string dir, vector<string> &files)
{
  DIR *dp;
  struct dirent *dirp;

  if((dp = opendir(dir.c_str())) == NULL)
  {
    char msg[200];
		snprintf(
      msg, 
      sizeof(msg)-1,
      "Error(%d) opening %s",
      errno,
      dir.c_str()
    );
    syslog(LOG_INFO, msg);
    return errno;
  }

  while ((dirp = readdir(dp)) != NULL)
  {
    files.push_back(string(dirp->d_name));
  }

  closedir(dp);
  
  sort(files.begin(), files.end());

  return 0;
}

// Open the training data from the file 'facedata.xml'

int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	// Create a file-storage interface

	CvFileStorage* fileStorage =
    cvOpenFileStorage(
      faceDataXml, 0, CV_STORAGE_READ
    );
	if (!fileStorage)
	{
    syslog(LOG_INFO, "Unable to open training database.");
		return 0;
	}

	// Load the names

	personNames.clear();	// Make sure it starts as empty
	
  nPersons = cvReadIntByName(fileStorage, 0, "nPersons", 0);
	if (nPersons == 0)
	{
    syslog(LOG_INFO, "No people found in training database.");
		return 0;
	}

	// Load each person's name

	for (int i=0; i < nPersons; i++) 
	{
		string sPersonName;
		char varname[200];
		snprintf(varname, sizeof(varname)-1, "personName_%d", (i+1));
		sPersonName = cvReadStringByName(fileStorage, 0, varname);
		personNames.push_back(sPersonName);
	}

	// Load the data

	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat =
    (CvMat*)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat =
    (CvMat*)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat =
    (CvMat*)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg =
    (IplImage*)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr =
    (IplImage**)cvAlloc(nTrainFaces*sizeof(IplImage *));
	
  for (int i=0; i < nEigens; i++)
	{
		char varname[200];
		snprintf(varname, sizeof(varname)-1, "eigenVect_%d", i);
		eigenVectArr[i] =
      (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	// release the file-storage interface

  cvReleaseFileStorage(&fileStorage);

  syslog(LOG_INFO, "Face recognition database loaded.");

	return 1;
}

// Find the most likely person based on a detection.
// Returns the index, and stores the confidence value
// into pConfidence.

int findNearestNeighbor(
  float * projectedTestFace, float *pConfidence
)
{
	double leastDistSq = DBL_MAX;
	int iNearest = 0;

	for (int iTrain=0; iTrain < nTrainFaces; iTrain++)
	{
		double distSq = 0;

		for (int i=0; i < nEigens; i++)
		{
			float d_i =
        projectedTestFace[i] -
        projectedTrainFaceMat->data.fl[iTrain * nEigens + i];

#ifdef USE_MAHALANOBIS_DISTANCE
      // Mahalanobis distance (might give better results
      // than Eucalidean distance)
			
      distSq += d_i*d_i / eigenValMat->data.fl[i]; 
#else
      // Euclidean distance

			distSq += d_i*d_i;
#endif
		}

		if (distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}

	// Return the confidence level based on the Euclidean distance,
	// so that similar images should give a confidence between
  // 0.5 to 1.0, and very different images should give a
  // confidence between 0.0 to 0.5

	*pConfidence =
    1.0f -
    sqrt(
      leastDistSq / (float)(nTrainFaces * nEigens)
    ) / 255.0f;

	// Return the found index

	return iNearest;
}

// Make sure the given rectangle is completely within the given
// image dimensions

// Creates a new image copy that is of a desired size

IplImage* resizeImage(
  const IplImage *origImg, int newWidth, int newHeight
)
{
	int origWidth = 0;
	int origHeight = 0;
	
  if (origImg != NULL)
  {
		origWidth = origImg->width;
		origHeight = origImg->height;
	}
	if (
    origImg == NULL ||
    newWidth <= 0 || newHeight <= 0 ||
    origWidth <= 0 || origHeight <= 0
  )
	{
    syslog(
      LOG_INFO, "ERROR in resizeImage: Bad desired image size."
    );
    closelog();
		exit(1);
	}

	// Scale the image to the new dimensions, even if the aspect
  // ratio will be changed

  IplImage *outImg =
    cvCreateImage(
      cvSize(newWidth, newHeight),
      origImg->depth,
      origImg->nChannels
    );

	if (newWidth > origImg->width && newHeight > origImg->height)
	{
		// Make the image larger
		
    cvResetImageROI((IplImage*)origImg);
    
    // CV_INTER_CUBIC or CV_INTER_LINEAR is good for enlarging

    cvResize(origImg, outImg, CV_INTER_LINEAR);
	}
	else
	{
		// Make the image smaller

		cvResetImageROI((IplImage*)origImg);
    
    // CV_INTER_AREA is good for shrinking/decimation,
    // but bad at enlarging
		
    cvResize(origImg, outImg, CV_INTER_AREA);
	}

  return outImg;
}

CvRect cropRect(const CvRect rectIn, int w, int h)
{
	CvRect roi = CvRect(rectIn);
	
  // Make sure the displayed image is within the viewing dimensions

  // Limit the bottom-right from past the image

  if (roi.x + roi.width > w)
		roi.width = w - roi.x;
	if (roi.y + roi.height > h)
		roi.height = h - roi.y;
	
  // Limit the top-left from before the image
  
  if (roi.x < 0)
		roi.x = 0;
	if (roi.y < 0)
		roi.y = 0;
	
  // Limit the top-left from after the image

  if (roi.x > w-1)
		roi.x = w-1;
	if (roi.y > h-1)
		roi.y = h-1;
	
  // Limit the negative sizes

  if (roi.width < 0)
		roi.width = 0;
	if (roi.height < 0)
		roi.height = 0;
	
  // Limit the large sizes

  if (roi.width > w)
		roi.width = w - roi.x;
	if (roi.height > h)
		roi.height = h - roi.y;
	
  return roi;
}

// Returns a new image, a cropped version of the original

IplImage* cropImage(const IplImage *img, const CvRect region)
{
  //syslog(LOG_INFO, "cropImage begins");

	CvSize size;
	size.height = img->height;
	size.width = img->width;

	if (img->depth != IPL_DEPTH_8U)
	{
    syslog(
      LOG_INFO,
      "ERROR in cropImage: unknown image depth given in cropImage()."
     );
    closelog();
		exit(1);
	}

	// First create a new (color or greyscale) IPL Image and copy
  // contents of img into it
	
  IplImage *imageTmp =
    cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(img, imageTmp, NULL);

	// Create a new image of the detected region
	// Set region of interest to that surrounding the face

  CvRect checked = cropRect(region, img->width, img->height);

  cvSetImageROI(imageTmp, checked);
	
  // Copy region of interest (i.e. face) into a new iplImage
  // (imageRGB) and return it
	
  size.width = checked.width;
	size.height = checked.height;

  IplImage *imageRGB =
    cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
	cvCopy(imageTmp, imageRGB, NULL);	// Copy just the region.

  cvReleaseImage(&imageTmp);

  //syslog(LOG_INFO, "cropImage ends");

  return imageRGB;		
}

// Perform face detection on the input image, using the given Haar
// cascade classifier. Assumes greyscale for input images.

// Returns a rectangle for the detected region in the given image.

CvRect detectFaceInImage(
  IplImage* img, CascadeClassifier &cascade
)
{
  //syslog(LOG_INFO, "detectFaceInImage begins");

  CvRect found_face;
  
  if (!small_img)
  {
    small_img =
      cvCreateImage(
        cvSize(
          cvRound(img->width / scale),
          cvRound(img->height / scale)
        ),
        8, 1
      );
  }
 
  cvResize(img, small_img, CV_INTER_LINEAR);
  cvEqualizeHist(small_img, small_img);

  Mat imgMat(small_img);

  // Detect objects in the small grayscale image.
  
  vector<Rect> objects;
  cascade.detectMultiScale(
    imgMat,
    objects,
    1.1f,
    2,
    CASCADE_FIND_BIGGEST_OBJECT,
    Size(20, 20)
  );

  if (objects.size() > 0)
  {
    // Found at least one face

    Rect r = (Rect)objects.at(0);

    found_face.x = (int)((double)r.x * scale);
    found_face.y = (int)((double)r.y * scale);
    found_face.width = (int)((double)r.width * scale);
    found_face.height = (int)((double)r.height * scale);
  }
  else
  {
    // Couldn't find the face

    found_face = cvRect(-1,-1,-1,-1);
  }
    
  return found_face;
}

string convert(const string& line)
{ 
  int len = line.length();  
  string nLine = "";

  if (len)
  {  
    nLine += line[0];  

    for (int i = 1; i < len; ++i)
    {
      if (isupper(line[i]))
        nLine += ' ';  
      nLine += line[i];
    }
  }
  return nLine;
} 

// Recognize the person in the supplied image

bool recognizeFromFile(
  const char *inputFile,
  const char *outputFile,
  const char *debugFile,
  CvHaarClassifierCascade *cascade,
	float * projectedTestFace,
	CvMat *trainPersonNumMat
)
{
  //syslog(LOG_INFO, "recognizeFromFile begins");

	// Load the image directly as greyscale

	IplImage *checkImg =
    cvLoadImage(inputFile, CV_LOAD_IMAGE_GRAYSCALE);

	if (!checkImg)
  {
		char msg[200];
		snprintf(
      msg,
      sizeof(msg)-1,
      "ERROR in recognizeFromFile(): Bad input image file: %s",
      inputFile
    );
    syslog(LOG_INFO, msg);
    return false;
	}

  // We'll try to detect the face using LBP

	CvRect faceRect = detectFaceInImage(checkImg, faceCascade);

  // Make sure a valid face was detected
	
  if (faceRect.width > 0)
	{
    // Get the detected face image

		IplImage *faceImg = cropImage(checkImg, faceRect);
		
    // Make sure the image is the same dimensions as the
    // training images
		
    IplImage *sizedImg =
      resizeImage(faceImg, faceWidth, faceHeight);

    // Give the image a standard brightness and contrast,
    // in case it was too dark or low contrast

    // Create an empty greyscale image

    IplImage *equalizedImg =
      cvCreateImage(cvGetSize(sizedImg), 8, 1);
		cvEqualizeHist(sizedImg, equalizedImg);

		if (!equalizedImg)
		{
      syslog(
        LOG_INFO,
        "ERROR in recognizeFromFile(): no input image."
      );
      closelog();
			exit(1);
		}

    char imageFile[200];
    strcpy(imageFile, debugFile);
    char *dot = strrchr(imageFile, '.');
    *dot = 0;

    cvSaveImage(imageFile, equalizedImg);

		// If the face rec database has been loaded, then try to
    // recognize the person currently detected

		if (nEigens > 0)
		{
			// Project the test image onto the PCA subspace

			cvEigenDecomposite(
				equalizedImg,
				nEigens,
				eigenVectArr,
				0, 0,
				pAvgTrainImg,
				projectedTestFace
      );

			// Check which person it is most likely to be

      float confidence;
			int iNearest =
        findNearestNeighbor(projectedTestFace, &confidence);

			int nearest = trainPersonNumMat->data.i[iNearest];

      // Write results to the output file

      char msg[200];
		  snprintf(
        msg,
        sizeof(msg)-1,
        "Recognised %s with a confidence of %f",
        (nearest > 0 ? personNames[nearest-1].c_str() : "nobody"),
        confidence
      );
      syslog(LOG_INFO, msg);

      string message =
        (nearest > 0 ?
          convert(personNames[nearest-1]) :
          "Unrecognized"
        );
      std::ofstream o1(outputFile);
      o1 << message.c_str() << std::endl;

      std::ofstream o2(debugFile);
      o2 << message.c_str() << std::endl;
    }

		// Free the resources used for this frame

		cvReleaseImage(&faceImg);
	  cvReleaseImage(&sizedImg);
	  cvReleaseImage(&equalizedImg);
	}
	cvReleaseImage(&checkImg);

  return true;
}

int main(void)
{
  // Our process ID and Session ID

  pid_t pid, sid;
        
  // Fork off the parent process
  
  pid = fork();
  if (pid < 0)
  {
    exit(EXIT_FAILURE);
  }
  
  // If we got a good PID, then we can exit the parent process
  
  if (pid > 0)
  {
    exit(EXIT_SUCCESS);
  }

  // Change the file mode mask
  
  umask(0);
                
  // Open any logs here

  openlog("facerecd", LOG_PID|LOG_CONS, LOG_USER);

  // Create a new SID for the child process

  sid = setsid();
  if (sid < 0)
  {
    // Log the failure
    
    syslog(LOG_INFO, "Unable to get SID.");
    closelog();
    exit(EXIT_FAILURE);
  }

  // Change the current working directory

  if (chdir("/") < 0)
  {
		// Log the failure

    syslog(LOG_INFO, "Unable to change working directory.");
    closelog();
    exit(EXIT_FAILURE);
  }
        
  // Close out the standard file descriptors

  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  close(STDERR_FILENO);
        
  // Daemon-specific initialization goes here
  
  struct stat st = {0};

  if (stat(inDir, &st) == -1)
  {
    mkdir(inDir, 0700);
  }
  if (stat(outDir, &st) == -1)
  {
    mkdir(outDir, 0700);
  }
  if (stat(debugDir, &st) == -1)
  {
    mkdir(debugDir, 0700);
  }

  execute();

  exit(EXIT_SUCCESS);
}

void execute()
{
	// Load the previously saved training data

	CvMat *trainPersonNumMat = 0;
	if (loadTrainingData(&trainPersonNumMat))
  {
		faceWidth = pAvgTrainImg->width;
		faceHeight = pAvgTrainImg->height;
	}
  else
  {
    closelog();
    return;
  }

	// Project the test images onto the PCA subspace

	float *projectedTestFace =
    (float *)cvAlloc(nEigens*sizeof(float));

	// Load the HaarCascade classifier for face detection

  faceCascade.load(lblCascadeFilename);

	if (faceCascade.empty())
	{
    syslog(
      LOG_INFO,
      "ERROR in main loop: Could not load face detection classifier."
    );
    closelog();
		exit(1);
	}

  CvHaarClassifierCascade* cascade =
	  (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);

	if (!cascade)
	{
    syslog(
      LOG_INFO,
      "ERROR in main loop: Could not load face detection classifier."
    );
    closelog();
		exit(1);
	}

  vector<string> files = vector<string>();
  const char *contents = NULL;

  /* The Big Loop */

  while (1)
  {
    // Get the files in our "in" directory

    files.clear();
		getdir(inDir, files);
    
    if ((int)files.size() >= 3)
      contents = files[2].c_str();
    else
      contents = NULL;

    if (contents != NULL)
		{
      char input[256];
      input[0] = 0;
      strcat(input, inDir);
      strcat(input, "/");
      strcat(input, contents);

      char *last = strrchr(input, '/');

      char output[256];
      output[0] = 0;
      strcat(output, outDir);
      strcat(output, "/");
      strcat(output, last + 1);
      strcat(output, ".txt");

      char debug[256];
      debug[0] = 0;
      strcat(debug, debugDir);
      strcat(debug, "/");
      strcat(debug, last + 1);
      strcat(debug, ".txt");

      if (
        recognizeFromFile(
          input,
          output,
          debug,
          cascade,
          projectedTestFace,
          trainPersonNumMat
        )
      )
      {
        remove(input);
      }
    }
    sleep(0.5); /* wait half a second */
  }
	
  cvReleaseHaarClassifierCascade(&cascade);

  if (storage)
  {
    cvReleaseMemStorage(&storage);
  }
  if (cascade)
  {
    cvReleaseHaarClassifierCascade(&cascade);
  }

  cvReleaseImage(&small_img);
  
  closelog();
}