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


#ifndef _SSIG_ML_OBLIQUENODE_HPP_
#define _SSIG_ML_OBLIQUENODE_HPP_
// c++
#include <memory>
#include <map>
// ssiglib
#include <ssiglib/ml/classification.hpp>
#include <ssiglib/ml/pls.hpp>

namespace ssig {

	class ObliqueNode {

	public:
		ML_EXPORT ObliqueNode(void);
		ML_EXPORT static cv::Ptr<ObliqueNode> create();
		ML_EXPORT virtual ~ObliqueNode(void);
		ML_EXPORT void read(const cv::FileNode& fn);
		ML_EXPORT void write(cv::FileStorage& storage) const;

		void setNSamples(int pos, int neg);
		int getDepth();
		void setDepth(int depth);
		void setClassifier(ssig::Classifier *classifier);
		void learn(const cv::Mat_<float> &X,
			cv::Mat_<int> &responses,
			std::vector<int> &col_index);
		int projectFeatures(const cv::Mat_<float> &X,
			float &resp);
		void computeThreshold(std::vector<float> &v0, std::vector<float> &v1);
		ObliqueNode **getChild(int child);
		void setChild(int id, ssig::ObliqueNode *node);

	protected:
		ML_EXPORT ObliqueNode(const ObliqueNode& rhs);

	private:
		int id;
		int neg, pos;
		int depth;
		float threshold;			
		std::vector<int> col_index;

		ObliqueNode *children[2];
		ssig::Classifier *classifier;
	};

}  // namespace ssig

#endif  // !_SSIG_ML_ObliqueNode_HPP_


