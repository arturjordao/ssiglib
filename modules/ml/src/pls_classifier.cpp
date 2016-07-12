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

#include <ssiglib/ml/pls_classifier.hpp>

#include <cassert>
#include <string>
#include <unordered_set>

namespace ssig {

PLSClassifier::PLSClassifier() {
  // Constructor
}

PLSClassifier::~PLSClassifier() {
  // Destructor
}

PLSClassifier::PLSClassifier(const PLSClassifier& rhs) {
  // Constructor Copy
}

int PLSClassifier::predict(
  const cv::Mat_<float>& inp,
  cv::Mat_<float>& resp) const {
  if (mOpenClEnabled) {
    mClPls->predict(inp, resp);
  } else {
    mPls->predict(inp, resp);
  }
    cv::Mat_<float> r;
    r.create(inp.rows, mYColumns);

    int labelIdx = -1;
    if (!mIsMulticlass) {
      for (int row = 0; row < inp.rows; ++row) {
        r[row][0] = resp[row][0];
        r[row][1] = -1 * resp[row][0];
      }
      labelIdx = resp[0][0] > 0 ? 1 : -1;
      resp = r;
    }
  return inp.rows > 1 || mIsMulticlass ? 0 : labelIdx;
}

void PLSClassifier::addLabels(const cv::Mat& labels) {
  if (labels.cols > 1) {
    // multiclass
    mYColumns = labels.cols;
    mIsMulticlass = true;
  } else {
    std::unordered_set<int> labelsSet;
    for (int r = 0; r < labels.rows; ++r)
      labelsSet.insert(labels.at<int>(r, 0));
    if (labelsSet.size() > 2) {
      std::runtime_error(std::string("Number of Labels is greater than 2.\n") +
                         "This is a binary classifier!\n");
    }
  }
  mLabels = labels;
}

cv::Ptr<PLSClassifier> PLSClassifier::create() {
  return cv::Ptr<PLSClassifier>(new PLSClassifier());
}

void PLSClassifier::learn(
  const cv::Mat_<float>& input,
  const cv::Mat& labels) {
  // TODO(Ricardo): assert labels between -1 and 1
  addLabels(labels);
  assert(!labels.empty());

  cv::Mat_<float> l;
  mLabels.convertTo(l, CV_32F);
  auto X = input.clone();
  if (mOpenClEnabled) {
    mClPls = std::unique_ptr<OpenClPLS>(new OpenClPLS());
    mClPls->learn(X, l, mNumberOfFactors);
  } else {
    mPls = std::unique_ptr<PLS>(new PLS());
    mPls->learn(X, l, mNumberOfFactors);
  }

  X.release();
  l.release();
  mTrained = true;
}

cv::Mat PLSClassifier::getLabels() const {
  return mLabels;
}

std::unordered_map<int, int> PLSClassifier::getLabelsOrdering() const {
  if (mIsMulticlass) {
    std::unordered_map<int, int> ans;
    for (int i = 0; i < mYColumns; ++i) {
      ans[i] = i;
    }
    return ans;
  }
  return {{1, 0}, {-1, 1}};
}

bool PLSClassifier::empty() const {
  return static_cast<bool>(mPls);
}

bool PLSClassifier::isTrained() const {
  return mTrained;
}

bool PLSClassifier::isClassifier() const {
  return true;
}

void PLSClassifier::setClassWeights(const int classLabel, const float weight) {}

void PLSClassifier::read(const cv::FileNode& fn) {
  mPls = std::unique_ptr<PLS>(new PLS());
  mPls->load(fn);
}

void PLSClassifier::write(cv::FileStorage& fs) const {
  mPls->save(fs);
}

Classifier* PLSClassifier::clone() const {
  auto copy = new PLSClassifier;

  copy->setNumberOfFactors(getNumberOfFactors());

  return copy;
}

int PLSClassifier::getNumberOfFactors() const {
  return mNumberOfFactors;
}

void PLSClassifier::setNumberOfFactors(int numberOfFactors) {
  mNumberOfFactors = numberOfFactors;
}

}  // namespace ssig
