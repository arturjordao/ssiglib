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
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// c++
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
// ssiglib
#include "ssiglib/core/util.hpp"
#include "ssiglib/core/firefly.hpp"


struct Utility : ssig::UtilityFunctor {
  float operator()(const cv::Mat& v) const override {
    cv::Mat_<float> f;
    v.convertTo(f, CV_32FC1);
    float x = f[0][0];
    float y = f[0][1];
    auto ans = static_cast<float>(
      exp(-1 * ((x - 4) * (x - 4)) - ((y - 4) * (y - 4)))
      + exp(-((x + 4) * (x + 4)) - ((y - 4) * (y - 4))) +
        2 * exp(-(x * x) - ((y + 4) * (y + 4))) +
        2 * exp(-(x * x) - (y * y)));
    return ans;
  }
};

struct Distance : ssig::DistanceFunctor {
  virtual float operator()(const cv::Mat& x,
                           const cv::Mat& y) const override {
    return static_cast<float>(cv::norm(x - y, cv::NORM_L2));
  }
};


TEST(Utils, Reorder) {
  // TODO(Ricardo): move this test to test_util.cpp

  cv::Mat_<float> a;
  cv::Mat_<int> o;
  for (int i = 0; i < 20; ++i) {
    float rndValue = static_cast<float>(cv::theRNG().gaussian(20));
    a.push_back(rndValue);
  }

  cv::Mat gt;
  cv::sort(a, gt, cv::SORT_EVERY_COLUMN);

  cv::sortIdx(a, o, cv::SORT_EVERY_COLUMN);

  cv::Mat ordered;
  ssig::Util::reorder(a, o, ordered);
  cv::Mat ans;
  cv::compare(ordered, gt, ans, cv::CMP_EQ);
  ASSERT_EQ(20, cv::countNonZero(ans));
}

TEST(Utils, StlReorder) {
  // TODO(Ricardo): move this test to test_util.cpp

  std::vector<int> a;
  std::vector<int> o;
  for (int i = 0; i < 20; ++i) {
    auto rndValue = static_cast<int>(cv::theRNG().gaussian(20));
    a.push_back(rndValue);
  }
  auto gt = a;
  std::sort(gt.begin(), gt.end());
  auto sorted = ssig::Util::sort(a, a.size(), o);
  ASSERT_EQ(sorted, gt);
}

TEST(Firefly, Execution) {
  cv::Ptr<ssig::DistanceFunctor> dist = cv::makePtr<Distance>();
  cv::Ptr<ssig::UtilityFunctor> util = cv::makePtr<Utility>();
  auto firefly = ssig::Firefly::create(util, dist);

#if 0
  cv::Mat_<float> imgMat;
  imgMat.create(1000, 1000);
  imgMat = 0;
  float x = -6;
  const float delta = 12.0f / 1000.0f;
  for (int i = 0; i < 1000; ++i) {
    float y = 6;
    for (int j = 0; j < 1000; ++j) {
      Utility foo;
      cv::Mat_<float> v = (cv::Mat_<float>(1, 2) << x, y);
      float value = foo(v);

      imgMat[j][i] = value;

      y -= delta;
    }
    x += delta;
  }

  cv::Mat img;
  imgMat.convertTo(img, CV_8UC1, 128);
  cv::applyColorMap(img, img, cv::COLORMAP_JET);
#endif


  cv::Mat_<float> pop;
  const int popLen = 50;
  const int d = 2;
  pop = cv::Mat::zeros(popLen, d, CV_32F);
  for (int i = 0; i < popLen; ++i) {
    cv::Mat_<float> vec(1, d);
    cv::randu(vec, cv::Scalar::all(-5), cv::Scalar::all(5));
    vec.copyTo(pop.row(i));
  }

  int maxIt = 30;
  float absorption = 1.5f;
  float step = 0.02f;
  float annealling = 0.97f;

  firefly->setMaxIterations(maxIt);
  firefly->setAbsorption(absorption);
  firefly->setAnnealling(annealling);
  firefly->setStep(step);

  cv::Mat_<float> state;
  cv::Mat_<float> results;

  cv::Mat frame;
  firefly->setup(pop);
  while (!firefly->iterate()) {
#if 0
    state = firefly.getState();
    results = firefly.getResults();
    frame = img.clone();
    for (int i = 0; i < state.rows; ++i) {
      x = state[i][0];
      float y = state[i][1];

      if (x < 6 && x > -6) {
        if (y < 6 && y > -6) {
          int r = static_cast<int>(abs(x + 6) * 1000.0f / 12.0f);
          int c = static_cast<int>(abs(y - 6) * 1000.0f / 12.0f);

          cv::circle(frame, cv::Point(c, r), 2, cv::Scalar(255, 255, 255), 10);
        }
      }
    }
#endif
  }
  results = firefly->getResults();
  // Asserts that the highest utility is near 2
  ASSERT_GT(results[popLen - 1][0], 1.8f);
}
