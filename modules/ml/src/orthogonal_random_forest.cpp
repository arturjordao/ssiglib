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

#include "ssiglib/ml/orthogonal_random_forest.hpp"

namespace ssig {
OrthogonalRF::OrthogonalRF() {
	// Constructor
	nTrees = 10;
	depth = 5;
	mtry = 5;
	accuracy = 0.01f;
}

OrthogonalRF::~OrthogonalRF() {
  // Destructor
}

cv::Ptr<OrthogonalRF> OrthogonalRF::create() {
	return cv::Ptr<OrthogonalRF>(new OrthogonalRF());
}

void OrthogonalRF::setNTrees(int nTrees){
	this->nTrees = nTrees;
}

int OrthogonalRF::getNTrees(){
	return this->nTrees;
}

void OrthogonalRF::setDepth(int depth){
	this->depth = depth;
}

int OrthogonalRF::getDepth(){
	return this->depth;
}

void OrthogonalRF::setMTry(int mtry){
	this->mtry = mtry;
}

int OrthogonalRF::getMTry(){
	return this->mtry;
}

void OrthogonalRF::setAccuracy(float acc){
	this->accuracy = acc;
}

float OrthogonalRF::getAccuracy(){
	return this->accuracy;
}

cv::Mat OrthogonalRF::getLabels() const {
	return mLabels;
}

std::unordered_map<int, int> OrthogonalRF::getLabelsOrdering() const {
	std::unordered_map<int, int> ordering = { { 1, 0 }, { -1, 1 } };
	return ordering;
}

void OrthogonalRF::setClassWeights(const int classLabel,
	const float weight) {
	//mMapLabel2Weight[classLabel] = weight;
}

bool OrthogonalRF::empty() const {
	return mModel == nullptr ? true : false;
}

bool OrthogonalRF::isTrained() const {
	return mModel != nullptr;
}

bool OrthogonalRF::isClassifier() const {
	return true;
}

Classifier* OrthogonalRF::clone() const {
	OrthogonalRF* ans = new OrthogonalRF();
	ans->nTrees = this->nTrees;
	ans->mtry = this->mtry;
	ans->accuracy = this->accuracy;
	ans->depth = this->depth;

	ans->mModel = cv::ml::RTrees::create();
	ans->mModel->setMaxDepth(ans->getDepth());
	ans->mModel->setRegressionAccuracy(ans->getAccuracy());
	ans->mModel->setActiveVarCount(ans->getMTry());
	return ans;
}

void OrthogonalRF::read(const cv::FileNode& fn) {
	
}

void OrthogonalRF::write(cv::FileStorage& fs) const {

}

void OrthogonalRF::learn(
	const cv::Mat_<float>& input,
	const cv::Mat& labels) {
	
	mModel = cv::ml::RTrees::create();
	mModel->setMaxDepth(getDepth());
	mModel->setRegressionAccuracy(getAccuracy());
	mModel->setActiveVarCount(getMTry());
	auto stopCriterion = cv::TermCriteria(cv::TermCriteria::MAX_ITER, getNTrees(), getAccuracy());
	mModel->setTermCriteria(stopCriterion);
	mModel->setMaxCategories(2);
	mModel->train(input, cv::ml::SampleTypes::ROW_SAMPLE, labels);
}

int OrthogonalRF::predict(
	const cv::Mat_<float>& inp,
	cv::Mat_<float>& resp,
	cv::Mat_<int>& labels) const {
	int label = -1;
	cv::Mat_<float> classLabel;
	if (!isTrained())
		return -1;
	
	mModel->predict(inp, resp, cv::ml::RTrees::Flags::PREDICT_SUM);
	resp = resp / (float)nTrees;

	resp[0][0] > 0 ? label = 1 : label = 0;;
	return label;
}

}  // namespace ssig
