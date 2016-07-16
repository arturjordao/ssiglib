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
		ML_EXPORT static cv::Ptr<ObliqueNode> create();
		ML_EXPORT virtual ~ObliqueNode(void);

		ML_EXPORT void learn(
			const cv::Mat_<float>& input,
			const cv::Mat& labels);

		ML_EXPORT void read(const cv::FileNode& fn);
		//ML_EXPORT void write(cv::FileStorage& fs);

		ML_EXPORT void setNSamples(int pos, int neg);

		ML_EXPORT void setDepth(int depth);

		//Builds a classification model for thisnode
		ML_EXPORT void createModel(const cv::Mat_<float> &X, cv::Mat_<int> &responses, std::vector<size_t> &col_index);

		//Projects a feature vector onto this node (return 0 to go to the left and 1 to go to the right)
		ML_EXPORT int projectFeatures(const cv::Mat_<float> &X);

	protected:
		ML_EXPORT ObliqueNode(void);
		ML_EXPORT ObliqueNode(const ObliqueNode& rhs);

	private:
		//Id of this node
		int id;
		//Amount of positive and negative samples of this node
		int neg, pos;
		//Depth of this node
		int depth;
		//Index of features used for this node
		std::vector<size_t> col_index;	
		//Decision threshold for this node
		float threshold;			
		//Children of this node
		ObliqueNode *children[2];
		
		Classifier *classifier;
		
		void computeThreshold(std::vector<float> &v0, std::vector<float> &v1);
	};

}  // namespace ssig

#endif  // !_SSIG_ML_ObliqueNode_HPP_


