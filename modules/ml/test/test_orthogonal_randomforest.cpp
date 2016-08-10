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
#include "ssiglib/ml/orthogonal_random_forest.hpp"

TEST(OrthogonalRF, SampleOrthogonalRF) {
  // Automatically generated stub

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

	auto classifier = ssig::OrthogonalRF::create();

	classifier->setNTrees(3);
	classifier->setDepth(10);
	classifier->setAccuracy(0.1);
	classifier->setMTry(10);

	classifier->learn(inp, labels);

	cv::Mat query1,query2;
	query1.create(1, 50, CV_32F);
	query2.create(1, 50, CV_32F);
	cv::randn(query1, cv::Mat::zeros(1, 1, CV_32F), cv::Mat::ones(1, 1, CV_32F));
	cv::randn(query2, cv::Mat::zeros(1, 1, CV_32F), cv::Mat::ones(1, 1, CV_32F));

	cv::Mat_<float> resp;
	classifier->predict(query1, resp);
	classifier->predict(query2, resp);
}

TEST(OrthogonalRF, Persistence) {
	cv::Mat_<int> labels = (cv::Mat_<int>(6, 1) << 1, 1, 1, -1, -1, -1);
	cv::Mat_<float> inp =
		(cv::Mat_<float>(6, 2) <<
		1, 2, 2, 2, 4, 6,
		102, 100, 104, 105, 99, 101);

	auto classifier = ssig::OrthogonalRF::create();

	classifier->setNTrees(3);
	classifier->setDepth(10);
	classifier->setAccuracy(0.1);
	classifier->setMTry(10);

	classifier->learn(inp, labels);

	cv::Mat_<float> query1 = (cv::Mat_<float>(1, 2) << 1, 2);
	cv::Mat_<float> query2 = (cv::Mat_<float>(1, 2) << 100, 103);

	cv::Mat_<float> resp;
	classifier->predict(query1, resp);
	auto ordering = classifier->getLabelsOrdering();
	int idx = ordering[1];
	EXPECT_GE(resp[0][idx], 0);
	classifier->predict(query2, resp);
	idx = ordering[-1];
	EXPECT_GE(resp[0][idx], 0);

	classifier->save("rf.yml", "root");

	auto loaded = ssig::OrthogonalRF::create();
	loaded->load("rf.yml", "root");
	loaded->predict(query1, resp);
	EXPECT_GE(resp[0][idx], 0);
	loaded->predict(query2, resp);
	idx = ordering[-1];
	EXPECT_GE(resp[0][idx], 0);
}
