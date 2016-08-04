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
		cv::Mat_<float>& response) const {

		ObliqueNode *node = this->root;
		float resp, acc;
		bool prune = false;
		int ret, labelIdx = -1;
		int level = 1;/*Used to compute the mean*/

		acc = 0;
		while (node != NULL) {
			ret = node->projectFeatures(inp, resp);
			acc += resp;

			if (ret == 0)
				node = *node->getChild(0);
			else
				node = *node->getChild(1);

			/*Pruning*/
			if (nodePruning == true){
				if ((acc / level) < 0){
					prune = true; break;
				}
			}
			level++;
		}

		if (prune == false)
			resp = acc / (level - 1);
		else
			resp = acc / level;

		response.create(1, 1);
		response[0][0] = resp;
		labelIdx = response[0][0] > 0 ? 1 : -1;
		return labelIdx;
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

	void ObliqueDTClassifier::prepareFeatures(){
		int n(0);
		this->index = 0;
		this->featuresIdx = std::vector<int>(numberOfFeatures);
		std::generate(featuresIdx.begin(), featuresIdx.end(), [&]{ return n++; });

		if (fsType == FeatureSelectionType::RANDOM_REPOSITION)
			std::random_shuffle(this->featuresIdx.begin(), this->featuresIdx.end());

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
		this->numberOfFeatures = X.cols;
		prepareFeatures();
		recursiveModel(&root, samplesIdx, X, Y, 0);
		X.release();
		Y.release();
		mTrained = true;
	}

	void ObliqueDTClassifier::recursiveModel(ssig::ObliqueNode **root,
		std::vector<int> samplesIdx,
		const cv::Mat_<float> &X,
		cv::Mat_<int> &responses,
		int depth){

		std::vector<int> samplesIdxLeft, samplesIdxRight;
		cv::Mat_<float> Xselect;
		cv::Mat_<int> YSelect;
		std::vector<int> feat_idx;
		std::vector<float> classResp;
		ssig::ObliqueNode *node, **left, **right;
		int class0 = 0, class1 = 0;
		float resp;
		
		if (samplesIdx.size() < 5)
			return;

		if (maxDepth != -1 && depth > maxDepth)
			return;

		for (int i = 0; i < samplesIdx.size(); i++) {
			Xselect.push_back(X.row(samplesIdx[i]));
			YSelect.push_back(responses.row(samplesIdx[i]));
			if (YSelect[i][0] == -1)
				class0++;
			else
				class1++;
		}

		float percentualClass0 = (float)class0 / samplesIdx.size();
		if (percentualClass0 == 0 || 1-percentualClass0 ==0){
			return;
		}


		feat_idx = nextFeatures();

		node = new ssig::ObliqueNode();
		node->setNSamples(class1, class0);
		node->setDepth(depth);
		node->setClassifier(this->classifier);

		node->learn(Xselect, YSelect, feat_idx);

		for (int i = 0; i < samplesIdx.size(); i++) {
			if (node->projectFeatures(X.row(samplesIdx[i]), resp) == 0)
				samplesIdxLeft.push_back(samplesIdx[i]);
			else
				samplesIdxRight.push_back(samplesIdx[i]);
		}

		if (samplesIdxLeft.size() == 0 || samplesIdxRight.size() == 0)
			return;

		*root = node;
		left = node->getChild(0);
		right = node->getChild(1);

		this->recursiveModel(left, samplesIdxLeft, X, responses, node->getDepth() + 1);
		this->recursiveModel(right, samplesIdxRight, X, responses, node->getDepth() + 1);
	}

	cv::Mat ObliqueDTClassifier::getLabels() const {
		return mLabels;
	}

	std::unordered_map<int, int> ObliqueDTClassifier::getLabelsOrdering() const {
		return{ { 1, 0 }, { -1, 1 } };
	}

	std::vector<int> ObliqueDTClassifier::nextFeatures(){
		std::vector<int> subSpace;

		if (fsType == FeatureSelectionType::RANDOM_FIXED){
			auto tmp = this->featuresIdx;
			std::random_shuffle(tmp.begin(), tmp.end());
			subSpace.assign(tmp.begin(), tmp.begin() + mtry);
		}
		else{
			subSpace.assign(this->featuresIdx.begin() + index, featuresIdx.begin() + index + mtry);
			index += mtry;
		}

		if (this->featuresIdx.size() <= index + mtry)
			prepareFeatures();

		return subSpace;
	}

	bool ObliqueDTClassifier::empty() const {
		return static_cast<bool>(root);
	}

	bool ObliqueDTClassifier::isTrained() const {
		return mTrained;
	}

	bool ObliqueDTClassifier::isClassifier() const {
		return true;
	}

	void ObliqueDTClassifier::setClassWeights(const int classLabel, const float weight) {}

	int ObliqueDTClassifier::getMTry(){
		return this->mtry;
	}

	void ObliqueDTClassifier::setFSType(int fsType){
		this->fsType = fsType;
	}

	void ObliqueDTClassifier::setMTry(int mtry){
		this->mtry = mtry;
	}

	void ObliqueDTClassifier::read(const cv::FileNode& fn) {
		cv::FileNode n, n2;
		ssig::ObliqueNode *node;

		node = new ssig::ObliqueNode();

		n = fn["ObliqueDT"];
		n["mtry"] >> mtry;
		n["nodePruning"] >> nodePruning;
		n2 = n["root"];
		if (n2.begin() != n2.end()) {
			node->setClassifier(this->classifier);
			node->read(n2);
			this->root = node;

			this->recursiveLoad(n2, node);
		}
	}

	void ObliqueDTClassifier::recursiveSave(cv::FileStorage &storage,
		ssig::ObliqueNode *node) const {

		ssig::ObliqueNode *n;

		if (*node->getChild(0) != NULL) {
			storage << "left" << "{";
			n = *node->getChild(0);
			n->write(storage);
			this->recursiveSave(storage, *node->getChild(0));
			storage << "}";
		}

		if (*node->getChild(1) != NULL) {
			storage << "right" << "{";
			n = *node->getChild(1);
			n->write(storage);
			this->recursiveSave(storage, *node->getChild(1));
			storage << "}";
		}
	}

	void ObliqueDTClassifier::recursiveLoad(cv::FileNode &node,
		ssig::ObliqueNode *parent) const{

		ObliqueNode *dtNode;
		cv::FileNode n;

		if (parent == NULL)
			return;

		n = node["left"];
		if (n.begin() != n.end()) {
			dtNode = new ObliqueNode();
			dtNode->setClassifier(classifier);
			dtNode->read(n);
			parent->setChild(0,dtNode);
			recursiveLoad(n, dtNode);
		}

		n = node["right"];
		if (n.begin() != n.end()) {
			dtNode = new ObliqueNode();
			dtNode->setClassifier(classifier);
			dtNode->read(n);
			parent->setChild(1,dtNode);
			recursiveLoad(n, dtNode);
		}

	}

	void ObliqueDTClassifier::write(cv::FileStorage& fs) const {

		fs << "ObliqueDT" << "{";
		fs << "mtry" << mtry;
		fs << "nodePruning" << nodePruning;
		fs << "root" << "{";
		root->write(fs);

		recursiveSave(fs, root);

		fs << "}";

		fs << "}";
	}

	Classifier* ObliqueDTClassifier::clone() const {
		auto copy = new ObliqueDTClassifier;

		copy->setMTry(this->mtry);
		copy->setFSType(this->fsType);
		copy->setClassifier(this->classifier);
		copy->setDepth(this->maxDepth);

		return copy;
	}

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
