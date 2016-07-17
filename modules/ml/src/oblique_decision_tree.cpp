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

#include <ssiglib/ml/oblique_decision_tree.hpp>

#include <cassert>
#include <string>
#include <unordered_set>

namespace ssig {

	ObliqueDTClassifier::ObliqueDTClassifier() {
		// Constructor
	}

	ObliqueDTClassifier::~ObliqueDTClassifier() {
		// Destructor
	}

	ObliqueDTClassifier::ObliqueDTClassifier(const ObliqueDTClassifier& rhs) {
		// Constructor Copy
	}

	int ObliqueDTClassifier::predict(
		const cv::Mat_<float>& inp,
		cv::Mat_<float>& resp) const {
		//mPls->predict(inp, resp);
		
		cv::Mat_<float> r;
		//r.create(inp.rows, mYColumns);
		return resp[0][0];/*TODO:Checar isso*/
	}

	void ObliqueDTClassifier::addLabels(const cv::Mat& labels) {
		std::unordered_set<int> labelsSet;
		for (int r = 0; r < labels.rows; ++r)
			labelsSet.insert(labels.at<int>(r, 0));
		if (labelsSet.size() > 2) {
			std::runtime_error(std::string("Number of Labels is greater than 2.\n") +
				"This is a binary classifier!\n");
		}
		mLabels = labels;
	}

	cv::Ptr<ObliqueDTClassifier> ObliqueDTClassifier::create() {
		return cv::Ptr<ObliqueDTClassifier>(new ObliqueDTClassifier());
	}

	void ObliqueDTClassifier::learn(
		const cv::Mat_<float>& input,
		const cv::Mat& labels) {
		addLabels(labels);
		assert(!labels.empty());
		std::vector<int> samplesIdx;
		cv::Mat_<float> l;
		mLabels.convertTo(l, CV_32F);
		auto X = input.clone();
		cv::Mat_<int> Y= labels.clone();

		for (int i = 0; i < input.rows; i++) {
			samplesIdx.push_back(i);
		}
		numberOfFeatures = X.cols;
		recursiveModel(root, samplesIdx, X, Y, 0);
		X.release();
		Y.release();
		mTrained = true;
	}

	void ObliqueDTClassifier::recursiveModel(cv::Ptr<ssig::ObliqueNode> &root,
		std::vector<int> samplesIdx,
		const cv::Mat_<float> &X,
		cv::Mat_<int> &responses,
		int depth){
		std::vector<int> samplesIdxLeft, samplesIdxRight;
		cv::Mat_<float> Xselect;
		cv::Mat_<int> YSelect;
		std::vector<int> feat_idx;
		std::vector<float> classResp;
		cv::Ptr<ssig::ObliqueNode> node, left, right;
		int class0 = 0, class1 = 0;
		float resp;
		//Stop criterion 1: Number of samples < 5
		if (samplesIdx.size() < 5)
			return;
		// Stop criterion 2: Depth termination condition
		if (maxDepth != -1 && depth > maxDepth)
			return;

		//Select samples according to samplesIdx
		for (int i = 0; i < samplesIdx.size(); i++) {
			Xselect.push_back(X.row(samplesIdx[i]));
			YSelect.push_back(responses.row(samplesIdx[i]));
			if (YSelect[i][0] == -1)
				class0++;
			else
				class1++;
		}

		//Stop criterion 3: Some class is below of detemined percetual
		float percentualClass0 = (float)class0 / samplesIdx.size();

		//if (percentualClass0 > classPercentage || 1 - percentualClass0 > classPercentage){
		//	//if (depth == 0)/*To avoid error(does not have col_index) in the save method*/
		//	//	ReportError("Inconsistent number of classes at the depth 0");
		//	return;
		//}


		feat_idx = nextFeatures();

		//Build model
		//node = new DT_PLS_Node(nodeIds++);
		node = ssig::ObliqueNode::create();
		node->setNSamples(class1, class0);
		node->setDepth(depth);
		node->setClassifier(this->classifier);

		node->createModel(Xselect, YSelect, feat_idx);

		//Classify samples to generate new nodes
		for (int i = 0; i < samplesIdx.size(); i++) {
			if (node->projectFeatures(X.row(samplesIdx[i])) == 0)
				samplesIdxLeft.push_back(samplesIdx[i]);
			else
				samplesIdxRight.push_back(samplesIdx[i]);
		}

		//Stop criterion 4: Does not propagates anything to some side (Avoids equal models).
		if (samplesIdxLeft.size() == 0 || samplesIdxRight.size() == 0)
			return;

		root = node;

		//Call recursively
		left = node->getChild(0);
		right = node->getChild(1);

		this->recursiveModel(left, samplesIdxLeft, X, responses, node->getDepth() + 1);
		this->recursiveModel(right, samplesIdxRight, X, responses, node->getDepth() + 1);

		//root->setChild(left, 0);
		//root->setChild(right, 1);
	}

	cv::Mat ObliqueDTClassifier::getLabels() const {
		return mLabels;
	}

	std::unordered_map<int, int> ObliqueDTClassifier::getLabelsOrdering() const {
		return{ { 1, 0 }, { -1, 1 } };
	}

	std::vector<int> ObliqueDTClassifier::nextFeatures(){
		std::vector<int> featuresIdx;
		if (fsType == "randomPermutation"){
			/*if (features.size() <= index + mtry){
				features = GenerateRandomPermutation(features.size(), features.size());
				index = 0;
			}*/

		}
		else if (fsType == "noPermutation"){
			int n(0);
			featuresIdx = std::vector<int>(numberOfFeatures);
			std::generate(featuresIdx.begin(), featuresIdx.end(), [&]{ return n++; });
		}
		
		return featuresIdx;
	}

	bool ObliqueDTClassifier::empty() const {
		return false;//return static_cast<bool>(mPls);
	}

	bool ObliqueDTClassifier::isTrained() const {
		return mTrained;
	}

	bool ObliqueDTClassifier::isClassifier() const {
		return true;
	}

	void ObliqueDTClassifier::setClassWeights(const int classLabel, const float weight) {}

	void ObliqueDTClassifier::setMTry(int mtry){
		this->mtry = mtry;
	}

	int ObliqueDTClassifier::getMTry(){
		return this->mtry;
	}

	void ObliqueDTClassifier::setFSType(std::string fsType){
		this->fsType = fsType;
	}

	void ObliqueDTClassifier::read(const cv::FileNode& fn) {
		/*mPls = std::unique_ptr<PLS>(new PLS());
		mPls->load(fn);*/
	}

	void ObliqueDTClassifier::write(cv::FileStorage& fs) const {
		//mPls->save(fs);
	}

	/*Todo:Artur poderia dar clone no classifier*/
	Classifier* ObliqueDTClassifier::clone() const {
		auto copy = new ObliqueDTClassifier;

		//copy->setNumberOfFactors(getNumberOfFactors());

		return copy;
	}

	/*int ObliqueDTClassifier::getNumberOfFactors() const {
		return mNumberOfFactors;
	}*/

	void ObliqueDTClassifier::setClassifier(ssig::Classifier *classifier) {
		this->classifier = classifier->clone();
	}

	void ObliqueDTClassifier::setDepth(int depth){
		this->maxDepth = depth;
	}

	int ObliqueDTClassifier::getDepth(){
		return maxDepth;
	}
}  // namespace ssig
