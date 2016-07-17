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

TEST(OBLIQUENODE, BinaryClassification) {
	cv::Mat_<int> labels = (cv::Mat_<int>(10, 1) << 1, 1, 1, 1, 1,  -1, -1, -1, -1, -1);
	cv::Mat_<float> inp =
		(cv::Mat_<float>(10, 10) <<
		1, 2, 2, 2, 4, 6, 2, 9, 10, 11,
		102, 100, 104, 105, 99, 101, 99, 12, 19, 100,
		1, 2, 2, 4 , 9, 10, 8, 8, 10, 9,
		10, 22, 32, 54, 70, 10, 8, 80, 90, 9,
		35, 27, 2, 40, 69, 10, 88, 8, 10, 89,
		11, 21, 112, 34, 89, 10, 8, 48, 1, 78,
		1, 22, 2, 2, 43, 36, 2, 9, 10, 31,
		102, 100, 14, 115, 99, 101, 99, 12, 19, 200,
		12, 10, 14, 15, 79, 11, 78, 12, 19, 12,
		122, 19, 14, 15, 7, 3, 2, 1, 2, 1);
	
	auto classifierType = ssig::PLSClassifier::create();
	classifierType->setNumberOfFactors(1);

	auto classifier = ssig::ObliqueDTClassifier::create();
	classifier->setClassifier(classifierType);
	classifier->setDepth(2);
	classifier->setFSType("noPermutation");
	classifier->learn(inp, labels);
	

	cv::Mat_<float> query1 = (cv::Mat_<float>(1, 2) << 1, 2);
	cv::Mat_<float> query2 = (cv::Mat_<float>(1, 2) << 100, 103);

	cv::Mat_<float> resp;
	/*classifier->predict(query1, resp);
	auto ordering = classifier->getLabelsOrdering();
	int idx = ordering[1];
	EXPECT_GE(resp[0][idx], 0);
	classifier->predict(query2, resp);
	idx = ordering[-1];
	EXPECT_GE(resp[0][idx], 0);*/
}

