/*L*****************************************************************************
*
*  Copyright (c) 2015, Smart Surveillance Interest Group, all rights reserved.
*
*  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
*
*  By downloading, copying, installing or using the software you agree to this
*  license. If you do not agree to this license, do not download, install, copy
*  or use the software.
*
*                Software License Agreement (BSD License)
*             For Smart Surveillance Interest Group Library
*                         http://ssig.dcc.ufmg.br
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*
*    1. Redistributions of source code must retain the above copyright notice,
*       this list of conditions and the following disclaimer.
*
*    2. Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*
*    3. Neither the name of the copyright holder nor the names of its
*       contributors may be used to endorse or promote products derived from
*       this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
*  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
*  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
*  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
*  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
*  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
*  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
*  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************L*/

#include <gtest/gtest.h>
// opencv
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
// ssiglib
#include <ssiglib/ml/pls_classifier.hpp>
#include <ssiglib/ml/oblique_decision_tree.hpp>
#include <ssiglib/ml/oblique_random_forest.hpp>

TEST(OBLIQUENODE, BinaryClassification) {	
	cv::Mat_<float> inp;
	cv::Mat labels, dataPos, dataNeg;

	dataPos.create(20, 50, CV_32F);
	cv::randn(dataPos, cv::Mat::zeros(1, 1, CV_32F), cv::Mat::ones(1, 1, CV_32F));
	inp.push_back(dataPos);
	labels.push_back(cv::Mat(std::vector<int>(dataPos.rows, 1), false));


	dataNeg.create(40, 50, CV_32F);
	cv::randn(dataNeg, cv::Mat::zeros(1, 1, CV_32F), cv::Mat::ones(1, 1, CV_32F));
	inp.push_back(dataNeg);
	labels.push_back(cv::Mat(std::vector<int>(dataNeg.rows, -1), false));

	auto classifierType = ssig::PLSClassifier::create();
	classifierType->setNumberOfFactors(2);

	auto classifier = ssig::ObliqueDTClassifier::create();
	classifier->setClassifier(classifierType);
	classifier->setMTry(10);
	classifier->setDepth(2);
	classifier->setFSType(ssig::ObliqueDTClassifier::FeatureSelctionType::RANDOM);
	//classifier->learn(inp, labels);
	

	auto oRF = ssig::ObliqueRF::create();
	oRF->setNumberTree(3);
	oRF->setObliqueTree(classifier);
	oRF->learn(inp, labels);
	
	cv::Mat query1;
	cv::Mat_<float> resp;

	query1.create(1, 50, CV_32F);
	cv::randn(query1, cv::Mat::zeros(1, 1, CV_32F), cv::Mat::ones(1, 1, CV_32F));
	oRF->predict(query1, resp);
}

