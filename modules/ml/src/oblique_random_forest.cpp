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

#include <ssiglib/ml/oblique_random_forest.hpp>

#include <cassert>
#include <string>
#include <unordered_set>

namespace ssig {

	ObliqueRF::ObliqueRF() {
		// Constructor
	}

	ObliqueRF::~ObliqueRF() {
		// Destructor
	}

	ObliqueRF::ObliqueRF(const ObliqueRF& rhs) {
		// Constructor Copy
	}

	int ObliqueRF::predict(
		const cv::Mat_<float>& inp,
		cv::Mat_<float>& resp) const {

		float response;
		cv::Mat_<float> treeResp;
		resp.create(inp.rows, 1);

		for (int i = 0; i < inp.rows; i++){

			response = 0.0f;
			for (int tree = 0; tree < trees.size(); tree++){

				trees[i]->predict(inp.row(i), treeResp);
				response += treeResp[0][0];
			}
			response = response / (float)trees.size();
			resp[i][0] = response;
		}
		return 1;
	}

	void ObliqueRF::addLabels(const cv::Mat& labels) {

		std::unordered_set<int> labelsSet;
		for (int r = 0; r < labels.rows; ++r)
			labelsSet.insert(labels.at<int>(r, 0));
		if (labelsSet.size() > 2) {
			std::runtime_error(std::string("Number of Labels is greater than 2.\n") +
				"This is a binary classifier!\n");
		}
		mLabels = labels;
	}

	cv::Ptr<ObliqueRF> ObliqueRF::create() {
		return cv::Ptr<ObliqueRF>(new ObliqueRF());
	}

	void ObliqueRF::learn(
		const cv::Mat_<float>& input,
		const cv::Mat& labels) {
		// TODO(Ricardo): assert labels between -1 and 1
		addLabels(labels);
		assert(!labels.empty());

		cv::Mat_<float> l;
		mLabels.convertTo(l, CV_32F);
		auto X = input.clone();

		cv::Ptr<ssig::ObliqueDTClassifier> tree;
		for (int i = 0; i < nTree; i++){

			tree = (ssig::ObliqueDTClassifier*)treeTemplate->clone();
			tree->learn(input, labels);
			trees.push_back(tree);
		}

		X.release();
		l.release();
		mTrained = true;
	}

	cv::Mat ObliqueRF::getLabels() const {
		return mLabels;
	}

	std::unordered_map<int, int> ObliqueRF::getLabelsOrdering() const {
		
		return{ { 1, 0 }, { -1, 1 } };
	}

	bool ObliqueRF::empty() const {

		return trees.size()==0 ? true:false;
	}

	bool ObliqueRF::isTrained() const {
		return mTrained;
	}

	bool ObliqueRF::isClassifier() const {
		return true;
	}

	void ObliqueRF::setClassWeights(const int classLabel, const float weight) {}

	void ObliqueRF::read(const cv::FileNode& fn) {
		/*mPls = std::unique_ptr<PLS>(new PLS());
		mPls->load(fn);*/
	}

	void ObliqueRF::write(cv::FileStorage& fs) const {
		//mPls->save(fs);
	}

	Classifier* ObliqueRF::clone() const {
		auto copy = new ObliqueRF;

		//copy->setNumberTree(getNumberTree());

		return copy;
	}

	void ObliqueRF::setObliqueTree(ssig::ObliqueDTClassifier *tree){

		this->treeTemplate = tree;
	}

	int ObliqueRF::getNumberTree(){
		return nTree;
	}

	void ObliqueRF::setNumberTree(int n){
		this->nTree = n;
	}

}  // namespace ssig
