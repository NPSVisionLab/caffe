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
#include <sstream>

#include <Ice/Communicator.h>
#include <Ice/Initialize.h>
#include <Ice/ObjectAdapter.h>
#include <util/FileUtils.h>
#include <util/DetectorDataArchive.h>
#include <util/ServiceManI.h>
#include <util/OutputResults.h>

#include <stdlib.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <stdint.h>
#include <sys/stat.h>

#include "CaffeTrainerI.h"

#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

using namespace cvac;
using namespace Ice;

extern bool runProgram(const string &runString, bool wait);

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
   * IceBox.Service.Caffe_Trainer=CaffeTrainer:create --Ice.Config=config.service
   * ... the name of the service is Caffe_Trainer.
   */
  ICE_DECLSPEC_EXPORT IceBox::Service* create(CommunicatorPtr communicator)
  {
      CaffeTrainerI *trainer = new CaffeTrainerI();
      ServiceManagerI *sMan = new ServiceManagerI( trainer, trainer );
      trainer->setServiceManager( sMan );
      return sMan;
  }
}

///////////////////////////////////////////////////////////////////////////////

CaffeTrainerI::CaffeTrainerI()
  : callback(NULL)
  , mServiceMan(NULL)
  
{
}

CaffeTrainerI::~CaffeTrainerI()
{
}

void CaffeTrainerI::setServiceManager(ServiceManagerI *sman)
{
    mServiceMan = sman;
}

void CaffeTrainerI::starting()
{
    m_CVAC_DataDir = mServiceMan->getDataDir();	
    mTrainerProps = new TrainerPropertiesI();
}  

string CaffeTrainerI::getTrainingDirectory(const ::Ice::Current& current)
{
    std::string connectName = getClientConnectionName(current);
    mClientName = mServiceMan->getSandbox()->
             createClientName(mServiceMan->getServiceName(), connectName); 
    std::string trainDir = mServiceMan->getSandbox()->
                                       createTrainingDir(mClientName);
    return trainDir;
}


std::string CaffeTrainerI::getName(const ::Ice::Current& current)
{
    return mServiceMan->getServiceName();
}

std::string CaffeTrainerI::getDescription(const ::Ice::Current& current)
{
    return "Caffe Trainer";
}


TrainerProperties CaffeTrainerI::getTrainerProperties(
                                      const ::Ice::Current& current)
{
    mTrainerProps->writeProps();
    return *mTrainerProps;
}

void CaffeTrainerI::process( const Identity &client,
                              const RunSet& runset,
                              const TrainerProperties& trainProps,
                              const Current& current)
{
    mTrainerProps->load(trainProps);
    callback = TrainerCallbackHandlerPrx::uncheckedCast(
            current.con->createProxy(client)->ice_oneway());

    if (mTrainerProps->solver.size() == 0)
    {
        localAndClientMsg(VLogger::ERROR, callback, 
	                  "Solver must be passed in trainer properties!"
			  );
        return;
    }
    if (mTrainerProps->snapshot.size() > 0 &&
        mTrainerProps->weights.size() > 0)
    {
        localAndClientMsg(VLogger::ERROR, callback, 
		  "Properties must not have both snapshot and weights files!"
		  );
        return;
    }
    string trainDir = getTrainingDirectory(current);
    if (mTrainerProps->useExistingData == false)
    { // Create new data from the runset
        localAndClientMsg(VLogger::INFO, callback, 
		  "Reading in RunSet\n"
		  );
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

	///////////////////////////////////////////////////////////
	// Start - RunsetWrapper
	mServiceMan->setStoppable();  
	cvac::RunSetWrapper mRunsetWrapper(&runset,m_CVAC_DataDir,mServiceMan);
	mServiceMan->clearStop();
	if(!mRunsetWrapper.isInitialized())
	{
	    localAndClientMsg(VLogger::ERROR, callback,
		  "RunsetWrapper is not initialized, aborting.\n");    
	    return;
	}
	// End - RunsetWrapper

	////////////////////////////////////////////////////////////
	// Start - RunsetIterator
	int nSkipFrames = 1;  //the number of skip frames
	mServiceMan->setStoppable();
	cvac::RunSetIterator mRunsetIterator(&mRunsetWrapper,mRunsetConstraint,
                                       mServiceMan,callback,nSkipFrames);
	mServiceMan->clearStop();
	if(!mRunsetIterator.isInitialized())
	{
	    localAndClientMsg(VLogger::ERROR, callback,
		  "RunSetIterator is not initialized, aborting.\n");
	    return;
	} 
	//Write out the filenames and label numbers to filelist.txt
	string trainFilename = "/trainfilelist.txt";
	string valFilename = "/valfilelist.txt";
	string trainFile =  trainDir + trainFilename;
	string valFile =  trainDir + valFilename;
	ofstream trainf;
	ofstream valf;
	trainf.open(trainFile.c_str()); 
	if (trainf.is_open() == false)
	{
	    localAndClientMsg(VLogger::ERROR, callback,
	      "Could not open file %s for writing.\n", trainFile.c_str());
	    return;
	}
	valf.open(valFile.c_str()); 
	if (valf.is_open() == false)
	{
	    localAndClientMsg(VLogger::ERROR, callback,
	      "Could not open file %s for writing.\n", valFile.c_str());
	    return;
	}
	srandom(mTrainerProps->randomSeed);
	mServiceMan->setStoppable();
	while(mRunsetIterator.hasNext())
	{
	    if((mServiceMan != NULL) && (mServiceMan->stopRequested()))
	    {        
	        mServiceMan->stopCompleted();
	        break;
	    }
	    cvac::Labelable& labelable = *(mRunsetIterator.getNext());
	    // If we had to create a symbolic link then we already
	    // have a abs path so don't add m_CVAC_DataDir
	    FilePath fpath =  RunSetWrapper::getFilePath(labelable);
	    string fullname;
	    if (pathAbsolute(fpath.directory.relativePath))
	        fullname = fpath.directory.relativePath + "/" + 
            		           fpath.filename;
            else
	        fullname = getFSPath(fpath, m_CVAC_DataDir );

            long rannum = random();
	    float fran = rannum / (float)RAND_MAX; 
	    if (fran < mTrainerProps->validateRatio)
	    {
                valf << fullname << " " << labelable.lab.name << endl;
	    }else
	    {
		trainf << fullname << " " << labelable.lab.name << endl; 
	    }
	}  
	trainf.close();
	valf.close();
	mServiceMan->clearStop();
	ostringstream optionString;
	if (mTrainerProps->windowSize.width != 0 &&
	    mTrainerProps->windowSize.height != 0)
        {
	    optionString << " --resize_width "  << 
	              mTrainerProps->windowSize.width;
	    optionString << " --resize_height " << 
	              mTrainerProps->windowSize.height;
	}
	if (mTrainerProps->shuffle)
	{
	    optionString << " --shuffle ";
	}
	// Remove the old db directories if they exist
	deleteDirectory(m_CVAC_DataDir + "/" + 
	                    mTrainerProps->trainDBName);
	deleteDirectory(m_CVAC_DataDir + "/" + 
	                    mTrainerProps->validateDBName);

	// Convert the train set
	string runString = "../3rdparty/caffe/bin/convert_imageset ";
        runString.append(optionString.str());
	runString.append(" ");
	runString.append("/"); // Root directory since file_list.txt is abs
	runString.append(" ");
	runString.append(trainFile); 
	runString.append(" ");
	runString.append(m_CVAC_DataDir + "/" + mTrainerProps->trainDBName);
	// Run the caffe program that converts a filelist to database
	localAndClientMsg(VLogger::INFO, callback,
	      "Creating train DB %s.\n", mTrainerProps->trainDBName.c_str());
	if (runProgram(runString, true) == false)
	{
	    localAndClientMsg(VLogger::ERROR, callback,
	      "Could not convert images to db with  %s.\n", runString.c_str());
	    return;
	}
	//Convert the valuate set
	runString = "../3rdparty/caffe/bin/convert_imageset ";
        runString.append(optionString.str());
	runString.append(" ");
	runString.append("/"); // Root directory since file_list.txt is abs
	runString.append(" ");
	runString.append(valFile); 
	runString.append(" ");
	runString.append(m_CVAC_DataDir + "/" + mTrainerProps->validateDBName);
	// Run the caffe program that converts a filelist to database
	localAndClientMsg(VLogger::INFO, callback,
	      "Creating validate DB %s.\n", 
	      mTrainerProps->validateDBName.c_str());
	if (runProgram(runString, true) == false)
	{
	    localAndClientMsg(VLogger::ERROR, callback,
	      "Could not convert images to db with  %s.\n", runString.c_str());
	    return;
	}
	// compute the image mean
	if (mTrainerProps->meanName.size() > 0)
	{
	    string meanName = m_CVAC_DataDir + "/" + mTrainerProps->meanName;
	    localAndClientMsg(VLogger::INFO, callback,
		  "Creating Image mean %s.\n", meanName.c_str());
	    string runString = 
	    "../3rdparty/caffe/bin/compute_image_mean " +
	    m_CVAC_DataDir + "/" + mTrainerProps->trainDBName + " " +
	    meanName + " " + mTrainerProps->dbType;
	    if (runProgram(runString, true) == false)
	    {
		localAndClientMsg(VLogger::ERROR, callback,
		  "Could not create Image mean  %s.\n", meanName.c_str());
		return;
	    }
	}
    }else
    {
        localAndClientMsg(VLogger::INFO, callback, 
		  "Using existing database\n"
		  );
    }
    caffe::SolverParameter solver_param;
    string solverName = m_CVAC_DataDir + "/" + mTrainerProps->solver;
    caffe::ReadProtoFromTextFileOrDie(solverName, &solver_param);
    if (mTrainerProps->gpu)
	Caffe::set_mode(Caffe::GPU);
    else
	Caffe::set_mode(Caffe::CPU);
    shared_ptr<caffe::Solver<float> > solver(
                                caffe::GetSolver<float>(solver_param));
    if (mTrainerProps->snapshot.size() > 0)
    {
        localAndClientMsg(VLogger::INFO, callback,
		   "Continuing training from snapshot %s", 
		    mTrainerProps->snapshot.c_str());
	string sfile =  m_CVAC_DataDir + "/" + mTrainerProps->snapshot;
        solver->Solve(sfile); 
    }else if (mTrainerProps->weights.size() > 0)
    {
        localAndClientMsg(VLogger::INFO, callback,
		   "Finetuning training from weights %s", 
		   mTrainerProps->weights.c_str());
	string wfile =  m_CVAC_DataDir + "/" + mTrainerProps->weights;
	solver->net()->CopyTrainedLayersFrom(wfile);
        solver->Solve();
    }else
    {
        localAndClientMsg(VLogger::INFO, callback,
		   "Beginning training");
        solver->Solve();
    }

    localAndClientMsg(VLogger::INFO, NULL, "creating model file.\n");
    string proto_filename = m_CVAC_DataDir + "/" + mTrainerProps->modelProto;
    if (fileExists(proto_filename) == false)
    {
        localAndClientMsg(VLogger::ERROR, callback,
	   "Could not find model file %s required to generate model file\n",
	   proto_filename.c_str());
        return;
    }
    //delete the solver to release resources
    solver.reset();
    //release the GPU
    //cudaDeviceReset();

    DetectorDataArchive dda;
    std::string clientDir = mServiceMan->getSandbox()->
                                createClientDir(mClientName);
    std::string archiveFilename = getDateFilename(clientDir,  "caffe")+ ".zip";
    dda.setArchiveFilename(archiveFilename);
    // Copy training results into MODELFILE
    string filename(solver_param.snapshot_prefix());
    string model_filename;
    const int kBufferSize = 20;
    char iter_str_buffer[kBufferSize];
    int maxIter = solver_param.max_iter();
    snprintf(iter_str_buffer, kBufferSize, "_iter_%d", maxIter);
    filename += iter_str_buffer;
    model_filename = filename + ".caffemodel";
    dda.addFile(MODELID, proto_filename);
    dda.addFile(WEIGHTID, model_filename);
    string meanName;
    if (mTrainerProps->meanName.size() == 0)
    {
	//TODO fetch the name from the trasform_param of the data layer
        //caffe::NetParameter* np = solver_param.net_param_;
	// For now just use the default imagenet mean
	meanName = m_CVAC_DataDir + "/" + "imagenet_mean.binaryproto";
    }else
	meanName = m_CVAC_DataDir + "/" + mTrainerProps->meanName;
    if (fileExists(meanName) == false)
    {
        localAndClientMsg(VLogger::WARN, callback,
	       "Mean file %s not found\n", meanName.c_str());
    }else
    {
	dda.addFile(MEANID, meanName); 
    }
    dda.createArchive(trainDir);
    mServiceMan->getSandbox()->deleteTrainingDir(mClientName);
    FilePath detectorData;
    detectorData.filename = getFileName(archiveFilename);
    std::string relDir;
    int idx = clientDir.find(m_CVAC_DataDir.c_str(), 0, 
                             m_CVAC_DataDir.length());
    if (idx == 0)
    {
        relDir = clientDir.substr(m_CVAC_DataDir.length() + 1);
    }else
    {
        relDir = clientDir;
    }
    detectorData.directory.relativePath = relDir; 
    callback->createdDetector(detectorData);
    localAndClientMsg(VLogger::INFO, NULL, "training complete.\n");
    //////////////////////////////////////////////////////////////////////////
}


bool CaffeTrainerI::cancel(const Identity &client, const Current& current)
{
  localAndClientMsg(VLogger::WARN, NULL, "cancel not implemented.");
  return false;
}

//----------------------------------------------------------------------------
TrainerPropertiesI::TrainerPropertiesI()
{
    verbosity = 0;
    canSetWindowSize = true;
    windowSize.width = 0;
    windowSize.height = 0;
    falseAlarmRate = 0.0;
    recall = 0.0;
    videoFPS = -1.0;
    gpu = true;
    useExistingData = false;
    shuffle = true;
    validateRatio = 0.2f;
    randomSeed = 1234;
    dbType = "lmdb";
    trainDBName = "train_lmdb";
    validateDBName = "validate_lmdb";
    meanName = "mean.binaryproto";
    modelProto = "deploy.prototxt";

}

void TrainerPropertiesI::load(const TrainerProperties &p) 
{
    verbosity = p.verbosity;
    props = p.props;
    if (p.videoFPS > 0)
        videoFPS = p.videoFPS;
    //Only load values that are not zero
    if (p.windowSize.width > 0 && p.windowSize.height > 0)
        windowSize = p.windowSize;
    readProps();
}

bool TrainerPropertiesI::readProps()
{
    bool res = true;
    cvac::Properties::iterator it;
    for (it = props.begin(); it != props.end(); it++)
    {
	//debug
	printf("%s = %s\n", it->first.c_str(), it->second.c_str());
	if (it->first.compare("useExistingData") == 0)
	{
	    if ((it->second.compare("true") == 0) ||
		(it->second.compare("True") == 0))
	    {
	        useExistingData = true;
	    }else
	        useExistingData = false;
	}
        if (it->first.compare("gpu") == 0)
        {
            if ((it->second.compare("true") == 0) ||
                (it->second.compare("True") == 0))
            {
                gpu = true;
            }else
                gpu = false;
        }
        if (it->first.compare("shuffle") == 0)
        {
            if ((it->second.compare("true") == 0) ||
                (it->second.compare("True") == 0))
            {
                shuffle = true;
            }else
                shuffle = false;
        }
        if (it->first.compare("solver") == 0)
        {
	    solver = it->second;
	}
        if (it->first.compare("snapshot") == 0)
        {
	    snapshot = it->second;
	}
        if (it->first.compare("weights") == 0)
        {
	    weights = it->second;
	}
        if (it->first.compare("dbtype") == 0)
        {
	    dbType = it->second;
	}
        if (it->first.compare("trainDBName") == 0)
        {
	    trainDBName = it->second;
	}
        if (it->first.compare("validateDBName") == 0)
        {
	    validateDBName = it->second;
	}
        if (it->first.compare("validateRatio") == 0)
        {
	    istringstream ss(it->second);
	    if (!(ss >> validateRatio))
	    {
	        localAndClientMsg(VLogger::WARN, NULL, 
		                  "Could not convert validateRatio to float.");
	    }
	}
        if (it->first.compare("randomSeed") == 0)
        {
	    istringstream ss(it->second);
	    if (!(ss >> randomSeed))
	    {
	        localAndClientMsg(VLogger::WARN, NULL, 
		                  "Could not convert randomSeed to int.");
	    }
	}
        if (it->first.compare("meanName") == 0)
        {
	    meanName = it->second;
        }
        if (it->first.compare("modelProto") == 0)
        {
	    modelProto = it->second;
        }
    }   
    return res;
}
 
bool TrainerPropertiesI::writeProps()
{
    bool res = true;
    props.insert(std::pair<string, string>("solver", solver));
    props.insert(std::pair<string, string>("snapshot", snapshot));
    props.insert(std::pair<string, string>("weights", weights));
    props.insert(std::pair<string, string>("dbtype", dbType));
    props.insert(std::pair<string, string>("trainDBName", trainDBName));
    props.insert(std::pair<string, string>("validateDBName", validateDBName));
    props.insert(std::pair<string, string>("meanName", meanName));
    props.insert(std::pair<string, string>("modelProto", modelProto));
    ostringstream ss;
    ss << validateRatio;
    props.insert(std::pair<string, string>("validateRatio", ss.str()));
    ss.str("");
    ss.clear();
    ss << randomSeed;
    props.insert(std::pair<string, string>("randomSeed", ss.str()));
    if (gpu == true)
        props.insert(std::pair<string, string>("gpu", "true"));
    else
        props.insert(std::pair<string, string>("gpu", "false"));
    if (useExistingData == true)
        props.insert(std::pair<string, string>("useExistingData", "true"));
    else
        props.insert(std::pair<string, string>("useExistingData", "false"));
    if (shuffle == true)
        props.insert(std::pair<string, string>("shuffle", "true"));
    else
        props.insert(std::pair<string, string>("shuffle", "false"));
    return res;
}

