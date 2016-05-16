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

#include <random>

#include <opencv2/core.hpp>

#include <ssiglib/ml/svm_classifier.hpp>

TEST(SVMClassifier, BinaryClassification) {
  cv::Mat_<int> labels = (cv::Mat_<int>(6, 1) << 1, 1, 1, -1, -1, -1);
  cv::Mat_<float> inp =
    (cv::Mat_<float>(6, 2) << 0.8f, 0.8f, 0.7f, 0.7f, 0.9f, 0.8f,
    -0.8f, -0.9f, -0.8f, -0.7f, -0.7f, -0.7f);

  ssig::SVMClassifier classifier;

  classifier.setTermType(cv::TermCriteria::MAX_ITER);
  classifier.setEpsilon(0.01f);

  classifier.learn(inp, labels);

  cv::Mat_<float> query1 = (cv::Mat_<float>(1, 2) << 0.6f, 0.7f);
  cv::Mat_<float> query2 = (cv::Mat_<float>(1, 2) << -0.7f, -0.6f);

  cv::Mat_<float> resp;
  classifier.predict(query1, resp);
  auto ordering = classifier.getLabelsOrdering();
  int idx = ordering[1];
  EXPECT_GE(resp[0][idx], 0);
  idx = ordering[-1];
  classifier.predict(query2, resp);
  EXPECT_GE(resp[0][idx], 0);
}

// TEST(SVMClassifier, BinaryChi2Classification) {
//  cv::Mat_<float> inp;
//  cv::Mat_<int> labels = cv::Mat_<int>::zeros(6, 1);
//  inp = cv::Mat_<float>::zeros(6, 2);
//  auto rnd = std::default_random_engine();
//  for (int i = 0; i < 3; ++i) {
//    inp[i][0] = static_cast<float>(rnd() % 5);
//    inp[i][1] = static_cast<float>(rnd() % 5);
//    labels[i][0] = 1;
//    inp[3 + i][0] = static_cast<float>(100 + rnd() % 5);
//    inp[3 + i][1] = static_cast<float>(100 + rnd() % 5);
//    labels[3 + i][0] = -1;
//  }
//
//  ssig::SVMClassifier svm;
//  const int mtype = static_cast<int>(cv::ml::SVM::C_SVC);
//  svm.setModelType(mtype);
//  svm.setC(0.1f);
//  svm.setGamma(1);
//  svm.setNu(0);
//  svm.setP(0);
//  svm.setDegree(0);
//  const int ktype = static_cast<int>(cv::ml::SVM::CHI2);
//  svm.setKernelType(ktype);
//  svm.setTermType(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS);
//  svm.setEpsilon(FLT_EPSILON);
//  svm.setMaxIterations(static_cast<int>(1e8));
//
//  svm.learn(inp, labels);
//
//  cv::Mat_<float> query1 = (cv::Mat_<float>(1, 2) << 1, 2);
//  cv::Mat_<float> query2 = (cv::Mat_<float>(1, 2) << 100, 103);
//
//  cv::Mat_<float> resp;
//  svm.predict(query1, resp);
//  auto ordering = svm.getLabelsOrdering();
//  int idx = ordering[1];
//  EXPECT_GE(resp[0][idx], 0);
//  idx = ordering[-1];
//  svm.predict(query2, resp);
//  EXPECT_GE(resp[0][idx], 0);
// }
//
TEST(SVMClassifier, Persistence) {
  cv::Mat_<int> labels = (cv::Mat_<int>(6, 1) << 1, 1, 1, -1, -1, -1);
  cv::Mat_<float> inp =
    (cv::Mat_<float>(6, 2) << 0.8f, 0.8f, 0.7f, 0.7f, 0.9f, 0.8f,
    -0.8f, -0.9f, -0.8f, -0.7f, -0.7f, -0.7f);

  ssig::SVMClassifier classifier;
  classifier.setC(0.1f);
  classifier.setKernelType(ssig::SVMClassifier::LINEAR);
  classifier.setModelType(ssig::SVMClassifier::C_SVC);

  classifier.setTermType(cv::TermCriteria::MAX_ITER);
  classifier.setEpsilon(0.01f);

  classifier.learn(inp, labels);

  cv::Mat_<float> query1 = (cv::Mat_<float>(1, 2) << 0.6f, 0.7f);
  cv::Mat_<float> query2 = (cv::Mat_<float>(1, 2) << -0.7f, -0.6f);

  cv::Mat_<float> resp;
  classifier.predict(query1, resp);
  auto ordering = classifier.getLabelsOrdering();
  int idx = ordering[1];
  EXPECT_GE(resp[0][idx], 0);
  classifier.predict(query2, resp);
  idx = ordering[-1];
  EXPECT_GE(resp[0][idx], 0);

  classifier.save("svm.yml", "root");
  ssig::SVMClassifier loaded;
  loaded.load("svm.yml", "root");
  remove("svm.yml");

  ordering = loaded.getLabelsOrdering();
  idx = ordering[1];
  loaded.predict(query1, resp);
  EXPECT_GE(resp[0][idx], 0);
  loaded.predict(query2, resp);
  idx = ordering[-1];
  EXPECT_GE(resp[0][idx], 0);
}

