#ifndef _CaffeTrainerI_H__
/*****************************************************************************
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
 ****************************************************************************/
#define _CaffeTrainerI_H__

#include <Data.h>
#include <Services.h>

#include <Ice/Ice.h>
#include <IceBox/IceBox.h>
#include <IceUtil/UUID.h>
#include <util/processRunSet.h>
#include <util/ServiceManI.h>
#include <util/RunSetIterator.h>

#include <caffe/caffe.hpp>
#include <cv.h>

namespace { // Need an anonymous namespace to resolve issues with classes of the same name
class TrainerPropertiesI : public cvac::TrainerProperties
{
 public:
  /**
   * Initialize fields for this detector.
   */
  TrainerPropertiesI();
  /**
   * Read the string properties and convert them to member data values.
   */
  bool readProps();
  /**
   * Convert member data values into string properties.
   */
  bool writeProps();
  /**
   * Load the struct's values into our class ignoring uninitialized values
   */
  void load(const TrainerProperties &p);

  // Note: We also use the windowSize.width and height as the value
  // to resize images to.

  bool gpu; // True if gpu is to be used
  bool useExistingData; // True if use existing databases
  bool shuffle; // pass shuffle option to database creator 
  float validateRatio; // 0-1 value of amount of dataset to use for validation
  int randomSeed;  //srand seed to use to randomize validation data selection
  std::string solver; // Name of the prototxt solver file with net definition
  std::string snapshot; // Name of the snapshot file to continue training
  std::string weights; // Name of finetuning weight file
  std::string trainDBName; // Name of the train database
  std::string validateDBName; // Name of the validate database
  std::string meanName; // Name of the mean image file to output
  std::string dbType; // A string lmdb or leveldb
  std::string modelProto; // Name of the prototxt to use for testing.

};

class CaffeTrainerI : public cvac::DetectorTrainer, public cvac::StartStop
{
public:
    CaffeTrainerI();
    ~CaffeTrainerI();

    std::string m_CVAC_DataDir; // Store an absolute path to the detector data files


public:
    virtual void process(const Ice::Identity &client, 
                         const ::cvac::RunSet& runset,
                         const ::cvac::TrainerProperties&,
                         const ::Ice::Current& current);
    virtual bool cancel(const Ice::Identity &client, const ::Ice::Current& current);
    virtual std::string getName(const ::Ice::Current& current);
    virtual std::string getDescription(const ::Ice::Current& current);
    virtual ::cvac::TrainerProperties getTrainerProperties(const ::Ice::Current& current);
    void setServiceManager(cvac::ServiceManagerI *sman);
    virtual void starting();
private:
    
    std::string getTrainingDirectory( const ::Ice::Current& current);

    cvac::ServiceManager    *mServiceMan;
    cvac::TrainerCallbackHandlerPrx callback;
    TrainerPropertiesI   *mTrainerProps;
    std::string mClientName;

};
}
#endif //_CaffeTrainerI_H__
