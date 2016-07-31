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

//#include <ssiglib/ml/pls_classifier.hpp>
#include <ssiglib/ml/oblique_node.hpp>
#include <cassert>
#include <string>
#include <unordered_set>

namespace ssig {

	ObliqueNode::ObliqueNode() {
		// Constructor
		children[0] = NULL;
		children[1] = NULL;
	}

	ObliqueNode::~ObliqueNode() {
		// Destructor
	}

	ObliqueNode::ObliqueNode(const ObliqueNode& rhs) {
		// Constructor Copy
	}

	cv::Ptr<ObliqueNode> ObliqueNode::create() {
		return cv::Ptr<ObliqueNode>(new ObliqueNode());
	}

	void ObliqueNode::setClassifier(ssig::Classifier *classifier){
		this->classifier = classifier;
	}

	ObliqueNode **ObliqueNode::getChild(int id){
		return &children[id];
	}

	int ObliqueNode::getDepth(){
		return this->depth;
	}
	void ObliqueNode::setNSamples(int pos, int neg){
		this->neg = neg;
		this->pos = pos;
	}
	void ObliqueNode::setDepth(int depth){
		this->depth = depth;
	}
	void ObliqueNode::createModel(
		const cv::Mat_<float> &X,
		cv::Mat_<int> &responses,
		std::vector<int> &col_index){
		cv::Mat_<float> Xtmp, ret, pos, neg;
		std::vector<float> v0;
		std::vector<float> v1;
		int i, idxPos;

		pos.create(0, X.cols);
		neg.create(0, X.cols);
		Xtmp.create(X.rows, (int)col_index.size());
		//Select the features before running the classification
		for (i = 0; i < col_index.size(); i++)
			X.col((int)col_index[(int)i]).copyTo(Xtmp.col((int)i));

		for (i = 0; i < responses.rows; i++){

			if (responses[(int)i][0] == 1)
				pos.push_back(Xtmp.row(i));
			else
				neg.push_back(Xtmp.row(i));
		}

		/*classifier->addSamples(pos, "1");
		classifier->addSamples(neg, "0");*/
		classifier->learn(Xtmp,responses);

		this->col_index = col_index;

		idxPos = 0;//classifier->retrieveResponseClassIDPosition("1");
		// regress each sample and compute the threshold
		for (i = 0; i < X.rows; i++) {
			classifier->predict(Xtmp.row(i), ret);
			if (responses(i, 0) == -1)
				v0.push_back(ret(0, 0));
			else
				v1.push_back(ret(0, 0));
		}

		computeThreshold(v0, v1);
	}

	int ObliqueNode::projectFeatures(const cv::Mat_<float> &X,
		float &resp) {
		cv::Mat_<float> ret;
		cv::Mat_<float> Xtmp;
		int i;

		//if (classifier == NULL)//if (model == NULL)
		//	ReportError("Classifier model must be created first - call DT_PLS_Node::createModel()");

		Xtmp.create(1, (int)col_index.size());

		//Select the features of this node before running the projection
		for (i = 0; i < col_index.size(); i++)
			Xtmp[0][i] = X[0][(int)col_index[i]];

		//Project sample
		classifier->predict(Xtmp, ret);
		resp = ret(0,0);
		//According to the threshold, return 0 or 1		
		if (ret(0, 0) >= this->threshold)
			return 1;

		return 0;
	}

	void ObliqueNode::computeThreshold(std::vector<float> &v0,
		std::vector<float> &v1) {
		float c1n1, c1n2, c2n1, c2n2;
		float n, n1, n2;
		float cm;
		std::map<float, float> gini;
		size_t i, j;
		float gini1, gini2;
		std::map<float, float>::iterator it;


		n = (float)(v0.size() + v1.size());

		for (cm = 0; cm <= 1; cm += 0.05f) {
			c1n1 = c1n2 = c2n1 = c2n2 = n1 = n2 = 0;
			for (i = 0; i < v0.size(); i++) {
				if (v0[i] < cm) {
					c1n1++;
					n1++;
				}
				else {
					c1n2++;
					n2++;
				}
			}
			for (i = 0; i < v1.size(); i++) {
				if (v1[i] < cm) {
					c2n1++;
					n1++;
				}
				else {
					c2n2++;
					n2++;
				}
			}

			if (n1 == 0 || n2 == 0)
				continue;
			gini1 = 1 - (c1n1 / n1) * (c1n1 / n1) - (c2n1 / n1) * (c2n1 / n1);
			gini2 = 1 - (c1n2 / n2) * (c1n2 / n2) - (c2n2 / n2) * (c2n2 / n2);
			gini.insert(std::pair<float, float>((n1 / n) * gini1 + (n2 / n) * gini2, cm));
		}
		//TODO:Artur, fix gini bug (gini.size()==0)
		if (gini.size() == 0)
			this->threshold = 0;
		else{
			it = gini.begin();
			this->threshold = it->second;
		}
	}

	void ObliqueNode::learn(
		const cv::Mat_<float>& input,
		const cv::Mat& labels) {
		//// TODO(Ricardo): assert labels between -1 and 1
		//addLabels(labels);
		//assert(!labels.empty());

		//cv::Mat_<float> l;
		//mLabels.convertTo(l, CV_32F);
		//auto X = input.clone();
		//mPls = std::unique_ptr<PLS>(new PLS());
		//mPls->learn(X, l, mNumberOfFactors);

		//X.release();
		//l.release();
		//mTrained = true;
	}

	void ObliqueNode::read(const cv::FileNode& fn) {

		cv::Mat_<int> indices;
		cv::FileNode n;
		size_t i;
		std::string name;
		std::string type;

		n = fn["classifier"];
		n["name"] >> name;
		n["type"] >> type;
		classifier->read(fn);

		fn["col_index"] >> indices;
		fn["threshold"] >> threshold;

		for (int i = 0; i < indices.rows; i++)
			col_index.push_back(indices[(int)i][0]);
	}

	void ObliqueNode::save(cv::FileStorage& storage) const {
		classifier->write(storage);
	}
}  // namespace ssig
