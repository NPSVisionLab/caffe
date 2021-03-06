/******************************************************************************
 * CVAC Software Disclaimer
 * 
 * This software was developed at the Naval Postgraduate School, Monterey, CA,
 * by employees of the Federal Government in the course of their official duties.
 * Pursuant to title 17 Section 105 of the United States Code this software
 * is not subject to copyright protection and is in the public domain. It is 
 * an experimental system.  The Naval Postgraduate School assumes no
 * responsibility whatsoever for its use by other parties, and makes
 * no guarantees, expressed or implied, about its quality, reliability, 
 * or any other characteristic.
 * We would appreciate acknowledgement and a brief notification if the software
 * is used.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above notice,
 *       this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Naval Postgraduate School, nor the name of
 *       the U.S. Government, nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without
 *       specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE NAVAL POSTGRADUATE SCHOOL (NPS) AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NPS OR THE U.S. BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *****************************************************************************/
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <Ice/Communicator.h>
#include <Ice/Initialize.h>
#include <Ice/ObjectAdapter.h>
#include <util/FileUtils.h>
#include <util/DetectorDataArchive.h>
#include <util/ServiceManI.h>
#include <util/OutputResults.h>

#include <highgui.h>
#include <stdlib.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

#include "CaffeDetectI.h"

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

//using namespace caffe;
using namespace cvac;
using namespace Ice;



///////////////////////////////////////////////////////////////////////////////
// This is called by IceBox to get the service to communicate with.
extern "C"
{
  /**
   * Create the detector service via a ServiceManager.  The 
   * ServiceManager handles all the icebox interactions.  Pass the constructed
   * detector instance to the ServiceManager.  The ServiceManager obtains the
   * service name from the config.icebox file as follows. Given this
   * entry:
   * IceBox.Service.Caffe_Detector=CaffeDetector:create --Ice.Config=config.service
   * ... the name of the service is BOW_Detector.
   */
  ICE_DECLSPEC_EXPORT IceBox::Service* create(CommunicatorPtr communicator)
  {
    CaffeDetectI *detector = new CaffeDetectI();
    ServiceManagerI *sMan = new ServiceManagerI( detector, detector );
    detector->setServiceManager( sMan );
    return sMan;
  }
}

///////////////////////////////////////////////////////////////////////////////

CaffeDetectI::CaffeDetectI()
  : callback(NULL)
  , mServiceMan(NULL)
  , gotModel(false)
{
    mHasMemoryData = false;
    mHasImageData = false;
    mWidth = 0;
    mHeight = 0;
}

CaffeDetectI::~CaffeDetectI()
{
}

void CaffeDetectI::setServiceManager(ServiceManagerI *sman)
{
  mServiceMan = sman;
}

void CaffeDetectI::starting()
{
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(mServiceMan->getServiceName().c_str());
  m_CVAC_DataDir = mServiceMan->getDataDir();	

  /*
  // check if the config.service file contains a trained model; if so, read it.
  string modelfile = mServiceMan->getModelFileFromConfig();

  if (modelfile.empty())
  {
    localAndClientMsg(VLogger::DEBUG, NULL, "No trained model file specified in service config.\n" );
  }
  else
  {
    localAndClientMsg(VLogger::DEBUG, NULL, "Will read trained model file as specified in service config: %s\n",
                      modelfile.c_str());
    if (pathAbsolute(modelfile) == false)
        modelfile =  m_CVAC_DataDir + "/" + modelfile;
        
    gotModel = readModelFile( modelfile, Ice::Current() );
    if (!gotModel)
    {
      localAndClientMsg(VLogger::WARN, NULL, "Failed to read pre-configured trained model "
                        "from: %s; will continue but now require client to send trained model\n",
                        modelfile.c_str());
    }
  }
  */
}  

bool CaffeDetectI::createLookupTbl(string lookupFileName)
{
    ifstream tempf;
    tempf.open(lookupFileName.c_str()); 
    if (tempf.is_open() == false)
    {
        localAndClientMsg(VLogger::WARN, NULL, 
                         "Failed to read lookup file %s ", 
                          lookupFileName.c_str());
        return false;
    }
    mLookup.clear();
    string lineInput;
    while (getline(tempf, lineInput))
    {
        size_t idx = lineInput.find_first_of(' ');
	if (idx != string::npos)
	    lineInput.erase(0, idx+1);
        mLookup.push_back(lineInput);
    }
    return true;

}

string CaffeDetectI::getClientDirectory(const ::Ice::Current& current)
{
    std::string connectName = getClientConnectionName(current);
    std::string clientName = mServiceMan->getSandbox()->createClientName(mServiceMan->getServiceName(),
                                                                           connectName);
      
    std::string clientDir = mServiceMan->getSandbox()->createClientDir(clientName);
    return clientDir;
}

bool CaffeDetectI::readModelFile( string model, string clientDir, 
                                  string fileListName)
{
    std::string zipfilename = model;
    DetectorDataArchive dda;
    dda.unarchive(zipfilename, clientDir);
    model = dda.getFile(MODELID);
    localAndClientMsg( VLogger::DEBUG, NULL,
                       "loading model %s\n", model.c_str());
    if (model.empty())
    {
        localAndClientMsg( VLogger::WARN, NULL,
	 "unable to load model from archive file %s\n", zipfilename.c_str());
        return false;
    }
    string weights = dda.getFile(WEIGHTID);

    // Create the caffe_net from the model
    if (mDetectorProps->gpu == true)
	Caffe::set_mode(Caffe::GPU);
    else
	Caffe::set_mode(Caffe::CPU);
   
    Caffe::set_phase(Caffe::TEST);
    //For relative path names to work in the .prototext file we need
    //to change the directory to this clientDir.  At the end we will 
    //change back.
    string curDir = getCurrentWorkingDirectory();
    chdir(clientDir.c_str());
    bool res = true;

    caffe::NetParameter netParams;
    caffe::ReadNetParamsFromTextFileOrDie(model, &netParams);
    // Find the IMAGE_DATA layer and change the file name
    int layer_id;
    int batch_size;
    for (layer_id = 0; layer_id < netParams.layers_size(); layer_id++)
    {
	caffe::LayerParameter* layer_param =
				 netParams.mutable_layers(layer_id);
	if (layer_param->has_image_data_param())
	{
	    // IMAGE_DATA has a major problem.  It does not return
	    // an error of a bad file in the file list unless its the
	    // first one and if its the first one it terminates the 
	    // program.  Both are bad so generate warning until
	    // this gets fixed in Caffe.
	    localAndClientMsg( VLogger::WARN, NULL,
	     "IMAGE_DATA will fail if an imagefile is missing or corrupt!");

	    caffe::ImageDataParameter* dp = 
			     layer_param->mutable_image_data_param();
	    dp->set_source(fileListName);
	    mHasImageData = true;
	    break;
	}
	if (layer_param->has_memory_data_param())
	{
	    const caffe::MemoryDataParameter& dp = 
			     layer_param->memory_data_param();
	    batch_size = dp.batch_size();
	    if (batch_size != 1)
	    {
		localAndClientMsg( VLogger::ERROR, NULL,
		 "Only batch_size of 1 is supported\n" );
		return false;
	    }
	    mHasMemoryData = true;
	    mWidth = dp.width();
	    mHeight = dp.height();
	    break;
	}
    }
    mCaffe_net = new caffe::Net<float>(netParams);

    if (!weights.empty())
    {
        localAndClientMsg( VLogger::DEBUG, NULL,
                       "loading pretrained data %s\n", weights.c_str());
        mCaffe_net->CopyTrainedLayersFrom(weights);
    } else
    {
        localAndClientMsg( VLogger::WARN, NULL,
                       "No weight file provided!");
        res = false;
    }
    string lookup = dda.getFile(LOOKUPID);
    if (!lookup.empty())
	createLookupTbl(lookup);
    // change back to original directory
    chdir(curDir.c_str());
    return res;
}

// Client verbosity
bool CaffeDetectI::initialize( const DetectorProperties& detprops,
                                 const FilePath& model,
                                 const Current& current)
{
  // Create DetectorPropertiesI class to allow the user to modify detection
  // parameters
  mDetectorProps = new DetectorPropertiesI();
  mDetectorProps->load(detprops);

  // Get the default CVAC data directory as defined in the config file
  localAndClientMsg(VLogger::DEBUG, NULL, "Initializing CaffeDetector...\n");
  Ice::PropertiesPtr iceprops = (current.adapter->getCommunicator()->getProperties());
  string verbStr = iceprops->getProperty("CVAC.ServicesVerbosity");
  if (!verbStr.empty())
  {
    getVLogger().setLocalVerbosityLevel( verbStr );
  }

  if(model.filename.empty())
  {
    if (!gotModel)
    {
        localAndClientMsg(VLogger::ERROR, callback, "No trained model available, aborting.\n" );
        return false;
    }
    // ok, go on with pre-configured model
  }
  else
  {
    if (gotModel)
    {
        localAndClientMsg(VLogger::WARN , callback, "Detector Preconfigured with a model file so ignoring passed in model %s.\n",
                          model.filename.c_str() );
    }else
    {
        string modelfile = getFSPath( model, m_CVAC_DataDir );
        localAndClientMsg( VLogger::DEBUG_1, NULL, "initializing with %s\n", modelfile.c_str());
        //For now lets wait to read the model file for when we process
        //the run set so the prototxt file has all the images on initialization
        //of the net.

        bool res = true;
        if (!res)
        {
          localAndClientMsg(VLogger::ERROR, callback,
                        "Failed to initialize because explicitly specified trained model "
                        "cannot be found or loaded: %s\n", modelfile.c_str());
          return false;
        }
    }
  }

  localAndClientMsg(VLogger::INFO, NULL, "CaffeDetector initialized.\n");
  return true;
}

std::string CaffeDetectI::getName(const ::Ice::Current& current)
{
  return mServiceMan->getServiceName();
}

std::string CaffeDetectI::getDescription(const ::Ice::Current& current)
{
  return "Caffe Detector";
}

void CaffeDetectI::setVerbosity(::Ice::Int verbosity, const ::Ice::Current& current)
{
}

DetectorProperties CaffeDetectI::getDetectorProperties(const ::Ice::Current& current)
{
  return DetectorPropertiesI();
}

/** Scans the detection cascade across each image in the RunSet
 * and returns the results to the client
 */
void CaffeDetectI::process( const Identity &client,
                              const RunSet& runset,
                              const FilePath& model,
                              const DetectorProperties& detprops,
                              const Current& current)
{
  callback = DetectorCallbackHandlerPrx::uncheckedCast(
            current.con->createProxy(client)->ice_oneway());

  bool initRes = initialize( detprops, model, current );
  if (initRes == false)
      return;
  //////////////////////////////////////////////////////////////////////////
  // Setup - RunsetConstraints
  cvac::RunSetConstraint mRunsetConstraint;  
  mRunsetConstraint.excludeLostFrames = false;
  mRunsetConstraint.excludeOccludedFrames = false;
  #ifdef WIN32
  mRunsetConstraint.spacesInFilenamesPermitted = true;
  #else
  mRunsetConstraint.spacesInFilenamesPermitted = false;
  #endif
  mRunsetConstraint.addType("png");
  mRunsetConstraint.addType("tif");
  mRunsetConstraint.addType("jpg");
  mRunsetConstraint.addType("jpeg");
  // End - RunsetConstraints

  //////////////////////////////////////////////////////////////////////////
  // Start - RunsetWrapper
  mServiceMan->setStoppable();  
  localAndClientMsg(VLogger::DEBUG, NULL, "Creating RunsetWrapper.\n");
  cvac::RunSetWrapper mRunsetWrapper(&runset,m_CVAC_DataDir,mServiceMan);
  mServiceMan->clearStop();
  if(!mRunsetWrapper.isInitialized())
  {
    localAndClientMsg(VLogger::ERROR, callback,
      "RunsetWrapper is not initialized, aborting.\n");    
    return;
  }
  // End - RunsetWrapper

  // For some reason OutputResults causes a memory error "bad linked list"
  // when stopIcebox is called so we will output results manually.

  localAndClientMsg(VLogger::DEBUG, NULL, "Creating run set iterator.\n");
  //////////////////////////////////////////////////////////////////////////
  // Start - RunsetIterator
  int nSkipFrames = 150;  //the number of skip frames
  mServiceMan->setStoppable();
  cvac::RunSetIterator mRunsetIterator(&mRunsetWrapper,mRunsetConstraint,
                                       mServiceMan,callback,nSkipFrames);
  localAndClientMsg(VLogger::DEBUG, NULL, "Run set iterator created.\n");
  mServiceMan->clearStop();
  if(!mRunsetIterator.isInitialized())
  {
    localAndClientMsg(VLogger::ERROR, callback,
      "RunSetIterator is not initialized, aborting.\n");
    return;
  } 
  // End - RunsetIterator
 
  // If the user sent us a file list then use that over the Runset
  // since we assume that its a big list and we already created it if
  // the file exists.  If it does not exist then create it.
  // If the user passed in a fileList we need to symbolic link it
  // to the one expected in the net prototext file

  string clientDir = getClientDirectory(current);
  string tempListFile = getTempFilename(clientDir, "filelist");
  bool fileListExists = false;
  string fileListPath;
  if (mDetectorProps->fileList.size() > 0)
  {

      fileListPath = m_CVAC_DataDir + "/" + mDetectorProps->fileList;
      if (fileExists(fileListPath))
          fileListExists = true;
      tempListFile = fileListPath;
  }
  int cnt = 0;
  vector<Result *> resultList;
  vector<Labelable *> labelList;
  if (mDetectorProps->fileList.size() == 0 || fileListExists == false)
  {
      ofstream tempf;
      tempf.open(tempListFile.c_str()); 
      if (tempf.is_open() == false)
      {
	localAndClientMsg(VLogger::ERROR, callback,
	  "Could not open file %s for writing.\n", tempListFile.c_str());
	return;
      }
      mServiceMan->setStoppable();
      localAndClientMsg(VLogger::DEBUG, NULL, "Starting run set iterator.\n");
      while(mRunsetIterator.hasNext())
      {
	if((mServiceMan != NULL) && (mServiceMan->stopRequested()))
	{        
	  mServiceMan->stopCompleted();
	  break;
	}
    
	cvac::Labelable& labelable = *(mRunsetIterator.getNext());
	Result &curres = mRunsetIterator.getCurrentResult();
	resultList.push_back(&curres);
	labelList.push_back(&labelable);
	string fullname;
	FilePath fpath = RunSetWrapper::getFilePath(labelable);
	if (pathAbsolute(fpath.directory.relativePath))
	    fullname = fpath.directory.relativePath + "/" + fpath.filename;
	else
	    fullname = getFSPath( fpath, m_CVAC_DataDir );
	tempf << fullname << " 0" << endl; 
	cnt++;
      }  
      tempf.close();
  }else
  { // We need to count the number of lines in the file (= files)
    // and we need to create a labelable and a result
      ifstream tempf;
      tempf.open(fileListPath.c_str()); 
      if (tempf.is_open() == false)
      {
	 localAndClientMsg(VLogger::ERROR, callback,
	  "Could not open file %s for reading.\n", fileListPath.c_str());
	 return;
      }
      string filen;
      string labelVal;
      while (tempf >> filen >> labelVal)
      {
          cvac::Label label;
	  label.hasLabel = true;
	  label.name = labelVal;
	  string dir = getFileDirectory(filen);
	  string fname = getFileName(filen);
	  cvac::DirectoryPath dpath;
	  dpath.relativePath = dir;
	  cvac::FilePath fpath;
	  fpath.directory = dpath;
	  fpath.filename = fname;
	  cvac::ImageSubstratePtr sub = new ImageSubstrate(0,0, fpath);
	  cvac::LabelablePtr labptr = new Labelable(0,label, sub);
	  labelList.push_back(&(*labptr));
	  cvac::Result  *resptr = new Result();
	  resptr->original = labptr;
	  resultList.push_back(resptr);
	  cnt++;
      }
      tempf.close();
  }
  /*
  if (mDetectorProps->fileList.size() > 0)
  { // Link it to the one in the net file
      if (fileExists(tempListFile))
      {
#ifdef WIN32
          _unlink(tempListFile.c_str());
#else
          unlink(tempListFile.c_str());
#endif

      }
      if (makeSymlinkFile(tempListFile, fileListPath) == false)
      {
	  localAndClientMsg(VLogger::ERROR, callback,
              "Could link list file from %s to %s.\n", 
	       fileListPath.c_str(),
	       tempListFile.c_str());
	  return;
      }
  }
  */
  // We wait to read the model file and initialize the Net until after
  // we create the temp file list.
  string modelfile = getFSPath( model, m_CVAC_DataDir );
  localAndClientMsg(VLogger::DEBUG, NULL, "Reading model file.\n");
  if (readModelFile(modelfile, clientDir, tempListFile) == false)
  {
    localAndClientMsg(VLogger::ERROR, callback,
      "Could not initialize the Net with model file %s for writing.\n", 
       modelfile.c_str());
    return;
  }
  printf("CVAC: initialized the net\n");
  // Go through the runset again.  We run the net for each file
  // and save the result.  Here we assume that the order will be the same!
  int i;
  float loss = 0.0;
  for (i = 0; i < cnt; i++)
  {
      const vector<Blob<float>*>* resultPtr;
      if (mHasMemoryData)
      {
	  /** Read the data in ourselfs.  We have to do that since
	   * caffe does not report any errors loading the file and
	   * it continually tries to load the same one.
	   */
	  Labelable *lptr = labelList[i]; 
	  string fullname;
	  FilePath fpath = RunSetWrapper::getFilePath(*lptr);
	  if (pathAbsolute(fpath.directory.relativePath))
	      fullname = fpath.directory.relativePath + "/" + fpath.filename;
	  else
	      fullname = getFSPath( fpath, m_CVAC_DataDir );
	  caffe::Datum datum;
	  // Pass in a label of zero
	  if (ReadImageToDatum(fullname, 0, mHeight, mWidth, &datum))
	  {
	      std::vector<caffe::Datum> images;
	      float loss = 0.0;
	      images.push_back(datum);
	      boost::dynamic_pointer_cast< caffe::MemoryDataLayer<float> >
	               (mCaffe_net->layers()[0])->AddDatumVector(images);
	      float iter_loss;
	      const vector<Blob<float>*>& result =
		  mCaffe_net->ForwardPrefilled(&iter_loss);
	      resultPtr = &result;
	      loss += iter_loss;
	  }else
	  {
	      continue;  // Skip this file
	  }
      }else
      { // We assume the file was read on via the prototext file
	  vector<caffe::Blob<float>* > bottom_vec;
	  float iter_loss;
	  const vector<Blob<float>*>& result =
                  mCaffe_net->Forward(bottom_vec, &iter_loss);
	  resultPtr = &result;
	  loss += iter_loss;
      }

      //debug
      //printf("CVAC: Net forward ran\n");
      //printf("CVAC: Output result size %d\n", resultPtr->size());
      Result *rptr = resultList[i]; 
      // Since have batch_size configured to 1, each call to Forward
      // will process the next file.  The results will be the
      // label which we current always set to 0 and the
      // output which will be the classification id as defined
      // synset_words.txt in the caffe directory.
      //
      for (int j = 0; j < resultPtr->size(); ++j) 
      {
	  const float* result_vec = (*resultPtr)[j]->cpu_data();
	  for (int k = 0; k < (*resultPtr)[j]->count(); ++k)
	  {
	      const float score = result_vec[k];
	      int idx = (int)score;
	      const std::string& output_name = mCaffe_net->blob_names()[
			mCaffe_net->output_blob_indices()[j]];
	      printf("output=%s, score=%g\n", output_name.c_str(), score);
	      // To get the probabilities
	      // const shared_ptr<Blob<float>>&probs = 
	      //                  mCaffe_net.blob_by_name("prob");
	      // const float *probs_out = probs->cpu_data();
	      if (output_name == "output")
	      {
		  string labelname;
		  if (!mLookup.empty())
		  {
		      // Lookup what the class id is and return that
		      labelname = mLookup[idx];
		      size_t sidx = labelname.find_first_of(',');
		      if (sidx != string::npos)
			  labelname = labelname.substr(0,sidx);
		      printf("classid=%d, labelName=%s\n", idx, 
			   labelname.c_str());
		  }else
		  {
		      char buffer[64];
		      sprintf(buffer, "%d", idx);
		      labelname = buffer;
		      printf("labelName=%s\n", labelname.c_str());
		  }
		  // Send the result back to client
		  LabelablePtr newFound = new Labelable();
		  newFound->lab.hasLabel = true;
		  newFound->lab.name = labelname;
		  newFound->confidence = 1.0f;
		  rptr->foundLabels.push_back(newFound);
		  ResultSet resSet;
		  resSet.results.push_back(*rptr);
		  callback->foundNewResults(resSet);
		  // Remove it since we notified the client
		  rptr->foundLabels.pop_back();
		  //outputres.addResult(*rptr, *lptr, std::vector<cv::Rect>(), 
		  //                   labelname, 1.0f);
	      }
          }
      }
  }

  // We are done so send any final results
  mServiceMan->clearStop();
  delete mCaffe_net;  // Free up resources

  localAndClientMsg(VLogger::DEBUG, NULL, "process complete.\n");
  //////////////////////////////////////////////////////////////////////////
}

/** convert from OpenCV result to CVAC ResultSet
 */

bool CaffeDetectI::cancel(const Identity &client, const Current& current)
{
  localAndClientMsg(VLogger::WARN, NULL, "cancel not implemented.");
  return false;
}

//----------------------------------------------------------------------------
DetectorPropertiesI::DetectorPropertiesI()
{
    verbosity = 0;
    isSlidingWindow = true;
    canSetSensitivity = false;
    canPostProcessNeighbors = true;
    videoFPS = 0;
    nativeWindowSize.width = 0;
    nativeWindowSize.height = 0;
    falseAlarmRate = 0.0;
    recall = 0.0;
    minNeighbors = 0;
    slideScaleFactor = 0.0;
    slideStartSize.width = 0;
    slideStartSize.height = 0;
    slideStopSize.width = 0;
    slideStopSize.height = 0;
    slideStepX = 0.0f;
    slideStepY = 0.0f;

    gpu = true;
    callbackFreq = "labelable";
    iterations = 1;
    fileList = "";
}

void DetectorPropertiesI::load(const DetectorProperties &p) 
{
    verbosity = p.verbosity;
    props = p.props;
    if (p.videoFPS > 0)
        videoFPS = p.videoFPS;
    //Only load values that are not zero
    if (p.nativeWindowSize.width > 0 && p.nativeWindowSize.height > 0)
        nativeWindowSize = p.nativeWindowSize;
    if (p.minNeighbors >= 0)
        minNeighbors = p.minNeighbors;
    if (p.slideScaleFactor > 0)
        slideScaleFactor = p.slideScaleFactor;
    if (p.slideStartSize.width > 0 && p.slideStartSize.height > 0)
        slideStartSize = p.slideStartSize;
    if (p.slideStopSize.width > 0 && p.slideStopSize.height > 0)
        slideStopSize = p.slideStopSize;
    readProps();
}

bool DetectorPropertiesI::readProps()
{
    bool res = true;
    cvac::Properties::iterator it;
    for (it = props.begin(); it != props.end(); it++)
    {
	if (it->first.compare("callbackFrequency") == 0)
	{
	    callbackFreq = it->second;
	    if ((it->second.compare("labelable") != 0) &&
		(it->second.compare("immediate") != 0) &&
		(it->second.compare("final") == 0))
	    {
		localAndClientMsg(VLogger::ERROR, NULL,
			 "callbackFrequency type not supported.\n");
		res = false;
	    }
	}
        if (it->first.compare("gpu") == 0)
        {
            if ((it->second.compare("true") != 0) ||
                (it->second.compare("True") != 0))
            {
                gpu = true;
            }else
                gpu = false;
        }
        if (it->first.compare("fileList") == 0)
        {
            if (it->second.size() != 0)
            {
                fileList = it->second;
            }
        }
        if (it->first.compare("iterations") == 0)
        {
            int cnt = strtol(it->second.c_str(), NULL, 10);
            if (cnt > 0 && cnt != LONG_MAX && cnt != LONG_MIN)
                iterations = cnt;
            else
            {
                localAndClientMsg(VLogger::ERROR, NULL, 
                         "Invalid iterations property.\n");
                res = false;
            }
        }
    }   
    return res;
}
 
bool DetectorPropertiesI::writeProps()
{
    std::stringstream stream;
    stream << iterations;
    bool res = true;
    props.insert(std::pair<string, string>("callbackFrequency", callbackFreq));
    props.insert(std::pair<string, string>("fileList", fileList));
    if (gpu == true)
        props.insert(std::pair<string, string>("gpu", "true"));
    else
        props.insert(std::pair<string, string>("gpu", "false"));
    props.insert(std::pair<string, string>("iterations", stream.str()));
    return res;
}

