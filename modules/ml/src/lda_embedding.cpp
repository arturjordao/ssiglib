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

#include "ssiglib/ml/lda_embedding.hpp"

namespace ssig {
	cv::Ptr<LDAEmbedding> LDAEmbedding::create(
		const int dimensions,
		cv::InputArray labels) {
		auto ans = cv::Ptr<LDAEmbedding>(new LDAEmbedding());
		ans->setDimensions(dimensions);
		ans->setLabels(labels);

		return ans;
	}

	LDAEmbedding::LDAEmbedding(const LDAEmbedding& rhs) {
		// Constructor Copy
		mDimensions = rhs.getDimensions();
		mLabels = rhs.getLabels().clone();
	}

	LDAEmbedding& LDAEmbedding::operator=(const LDAEmbedding& rhs) {
		if (this != &rhs) {
			this->mDimensions = rhs.getDimensions();
			this->mLabels = rhs.getLabels().clone();
		}
		return *this;
	}

	void LDAEmbedding::learn(
		cv::InputArray input) {

		cv::Mat_<float> X = input.getMat();
		cv::Mat_<float> Xcopy = X.clone();
		cv::Mat B;
		mLDA.resize(mDimensions);

		for (int i = 0; i < mDimensions; i++){

			mLDA[i] = std::make_unique<cv::LDA>();
			mLDA[i]->compute(Xcopy, mLabels);
			B = mLDA[i]->eigenvectors();

			// Deflation of X
			B.convertTo(B, Xcopy.type());
			for (int row = 0; row < Xcopy.rows; ++row)
				Xcopy.row(row) = Xcopy.row(row) - Xcopy.row(row).mul(B.t());
		}
	}

	void LDAEmbedding::project(
		cv::InputArray sample,
		cv::OutputArray output) {
		cv::Mat_<float> X, ldaSpace, proj;

		X = sample.getMat();
		cv::Mat_<float> Xcopy = X.clone();
		proj.create(X.rows, mDimensions);
		for (int i = 0; i < mDimensions; i++){

			for (int row = 0; row < X.rows; ++row){

				ldaSpace = mLDA[i]->project(X.row(row));
				proj[row][i] = ldaSpace[0][0];
			}
		}

		proj.copyTo(output);
	}

	int LDAEmbedding::getDimensions() const {
		return mDimensions;
	}

	void LDAEmbedding::setDimensions(const int dimensions) {
		mDimensions = dimensions;
	}

	cv::Mat_<float> LDAEmbedding::getLabels() const {
		return mLabels;
	}

	void LDAEmbedding::setLabels(cv::InputArray labels) {
		cv::Mat localLabels = labels.getMat();
		localLabels.convertTo(mLabels, CV_32F);
	}

	void LDAEmbedding::read(const cv::FileNode& fn) {}

	void LDAEmbedding::write(cv::FileStorage& fs) const {
		/*fs << "LDA"
			<< "{";
		fs << "nfactors" << mDimensions;
		for (int i = 0; i < mDimensions; i++){
			fs << "Dimension_" + std::to_string(i)
			   << "{";
			mLDA[i]->save(fs);
			fs << "}";
		}
		fs << "}";*/
	
	}
}  // namespace ssig
