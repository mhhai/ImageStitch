#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <iostream>
#include <fstream> 
#include <string>
#include <iomanip> 
#include "D:\\Opencv3.4.2\\opencv\\sources\\modules\\calib3d\\include\\opencv2\\calib3d\\calib3d.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/hal/hal.hpp"
#include <algorithm>
#include "D:\\Opencv3.4.2\\opencv\\sources\\modules\\core\\include\\opencv2\\core\\opencl\\ocl_defs.hpp"
#include"D:\Opencv3.4.2\opencv\sources\modules\calib3d\src\precomp.hpp"
#include<opencv2/opencv.hpp>
#include<opencv2/opencv_modules.hpp>
#include<opencv2/core.hpp>
#include<opencv2/world.hpp>

using namespace cv;
using namespace std;
using namespace detail;
int RANSACUpdateNumIters1(double p, double ep, int modelPoints, int maxIters)
{
	if (modelPoints <= 0)
		CV_Error(Error::StsOutOfRange, "the number of model points should be positive");

	p = MAX(p, 0.);
	p = MIN(p, 1.);
	ep = MAX(ep, 0.);
	ep = MIN(ep, 1.);

	// avoid inf's & nan's
	double num = MAX(1. - p, DBL_MIN);
	double denom = 1. - std::pow(1. - ep, modelPoints);
	if (denom < DBL_MIN)
		return 0;

	num = std::log(num);
	denom = std::log(denom);

	return denom >= 0 || -num >= maxIters * (-denom) ? maxIters : cvRound(num / denom);
}
class RANSACPointSetRegistrator1 : public PointSetRegistrator
{
public:
	RANSACPointSetRegistrator1(const Ptr<PointSetRegistrator::Callback>& _cb = Ptr<PointSetRegistrator::Callback>(),
		int _modelPoints = 0, double _threshold = 0, double _confidence = 0.99, int _maxIters = 1000)
		: cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), maxIters(_maxIters) {}

	int findInliers(const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh) const
	{
		//匹配的特征点是按照索引来的
		cb->computeError(m1, m2, model, err);
		//mask值为1表示该特征点是一个好的特征点，否则是一个外点
		mask.create(err.size(), CV_8U);

		CV_Assert(err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
		const float* errptr = err.ptr<float>();
		uchar* maskptr = mask.ptr<uchar>();
		float t = (float)(thresh*thresh);
		int i, n = (int)err.total(), nz = 0;
		for (i = 0; i < n; i++)
		{
			int f = errptr[i] <= t;
			maskptr[i] = (uchar)f;
			nz += f;
		}
		return nz;
	}

	bool getSubset(const Mat& m1, const Mat& m2,
		Mat& ms1, Mat& ms2, RNG& rng,
		int maxAttempts = 1000) const
	{
		cv::AutoBuffer<int> _idx(modelPoints);
		int* idx = _idx;
		int i = 0, j, k, iters = 0;
		int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
		int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
		int esz1 = (int)m1.elemSize1()*d1, esz2 = (int)m2.elemSize1()*d2;
		int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
		const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

		ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
		ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

		int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

		CV_Assert(count >= modelPoints && count == count2);
		CV_Assert((esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0);
		esz1 /= sizeof(int);
		esz2 /= sizeof(int);

		for (; iters < maxAttempts; iters++)
		{
			for (i = 0; i < modelPoints && iters < maxAttempts; )
			{
				int idx_i = 0;
				for (;;)
				{
					idx_i = idx[i] = rng.uniform(0, count);
					for (j = 0; j < i; j++)
						if (idx_i == idx[j])
							break;
					if (j == i)
						break;
				}
				for (k = 0; k < esz1; k++)
					ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
				for (k = 0; k < esz2; k++)
					ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
				i++;
			}
			if (i == modelPoints && !cb->checkSubset(ms1, ms2, i))
				continue;
			break;
		}

		return i == modelPoints && iters < maxAttempts;
	}

	bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const
	{
		Mat a = _model.getMat();
		Mat b = _mask.getMat();
		bool result = false;
		Mat m1 = _m1.getMat(), m2 = _m2.getMat();
		Mat err, mask, model, bestModel, ms1, ms2;

		int iter, niters = MAX(maxIters, 1);

		int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
		int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
		int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

		RNG rng((uint64)-1);

		CV_Assert(cb);
		CV_Assert(confidence > 0 && confidence < 1);

		CV_Assert(count >= 0 && count2 == count);
		if (count < modelPoints)
			return false;

		Mat bestMask0, bestMask;

		if (_mask.needed())
		{
			_mask.create(count, 1, CV_8U, -1, true);
			bestMask0 = bestMask = _mask.getMat();
			CV_Assert((bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count);
		}
		else
		{
			bestMask.create(count, 1, CV_8U);
			bestMask0 = bestMask;
		}

		if (count == modelPoints)
		{
			if (cb->runKernel(m1, m2, bestModel) <= 0)
				return false;
			bestModel.copyTo(_model);
			bestMask.setTo(Scalar::all(1));
			return true;
		}
		for (iter = 0; iter < niters; iter++)
		{
			int i, nmodels;
			if (count > modelPoints)
			{
				bool found = getSubset(m1, m2, ms1, ms2, rng, 10000);
				if (!found)
				{
					if (iter == 0)
						return false;
					break;
				}
			}
			if (nmodels <= 0)
				continue;
			CV_Assert(model.rows % nmodels == 0);
			Size modelSize(model.cols, model.rows / nmodels);  //3 * 3

			for (i = 0; i < nmodels; i++)
			{
				Mat model_i = model.rowRange(i*modelSize.height, (i + 1)*modelSize.height);
				int goodCount = findInliers(m1, m2, model_i, err, mask, threshold);
				//该函得到重映射误差err，并更新了mask
				if (goodCount > MAX(maxGoodCount, modelPoints - 1))
				{
					std::swap(mask, bestMask);
					model_i.copyTo(bestModel);
					maxGoodCount = goodCount;
					niters = RANSACUpdateNumIters1(confidence, (double)(count - goodCount) / count, modelPoints, niters);
				}
			}
		}

		if (maxGoodCount > 0)
		{
			if (bestMask.data != bestMask0.data)
			{
				if (bestMask.size() == bestMask0.size())
					bestMask.copyTo(bestMask0);
				else
					transpose(bestMask, bestMask0);
			}
			bestModel.copyTo(_model);
			result = true;
		}
		else
			_model.release();

		return result;
	}

	void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) CV_OVERRIDE { cb = _cb; }

	Ptr<PointSetRegistrator::Callback> cb;
	int modelPoints;
	double threshold;
	double confidence;
	int maxIters;
};
Ptr<PointSetRegistrator> createRANSACPointSetRegistrator1(const Ptr<PointSetRegistrator::Callback>& _cb,
	int _modelPoints, double _threshold,
	double _confidence, int _maxIters)
{
	return Ptr<PointSetRegistrator>(
		new RANSACPointSetRegistrator1(_cb, _modelPoints, _threshold, _confidence, _maxIters));
}
class HomographyEstimatorCallback CV_FINAL : public PointSetRegistrator::Callback
{
public:
	bool checkSubset(InputArray _ms1, InputArray _ms2, int count) const CV_OVERRIDE
	{
		Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
		if (haveCollinearPoints(ms1, count) || haveCollinearPoints(ms2, count))
			return false;

		// We check whether the minimal set of points for the homography estimation
		// are geometrically consistent. We check if every 3 correspondences sets
		// fulfills the constraint.
		//
		// The usefullness of this constraint is explained in the paper:
		//
		// "Speeding-up homography estimation in mobile devices"
		// Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
		// Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
		if (count == 4)
		{
			static const int tt[][3] = { {0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3} };
			const Point2f* src = ms1.ptr<Point2f>();
			const Point2f* dst = ms2.ptr<Point2f>();
			int negative = 0;

			for (int i = 0; i < 4; i++)
			{
				const int* t = tt[i];
				Matx33d A(src[t[0]].x, src[t[0]].y, 1., src[t[1]].x, src[t[1]].y, 1., src[t[2]].x, src[t[2]].y, 1.);
				Matx33d B(dst[t[0]].x, dst[t[0]].y, 1., dst[t[1]].x, dst[t[1]].y, 1., dst[t[2]].x, dst[t[2]].y, 1.);

				negative += determinant(A)*determinant(B) < 0;
			}
			if (negative != 0 && negative != 4)
				return false;
		}

		return true;
	}

	/**
	 * Normalization method:
	 *  - $x$ and $y$ coordinates are normalized independently
	 *  - first the coordinates are shifted so that the average coordinate is \f$(0,0)\f$
	 *  - then the coordinates are scaled so that the average L1 norm is 1, i.e,
	 *  the average L1 norm of the \f$x\f$ coordinates is 1 and the average
	 *  L1 norm of the \f$y\f$ coordinates is also 1.
	 *
	 * @param _m1 source points containing (X,Y), depth is CV_32F with 1 column 2 channels or
	 *            2 columns 1 channel
	 * @param _m2 destination points containing (x,y), depth is CV_32F with 1 column 2 channels or
	 *            2 columns 1 channel
	 * @param _model, CV_64FC1, 3x3, normalized, i.e., the last element is 1
	 */
	int runKernel(InputArray _m1, InputArray _m2, OutputArray _model) const CV_OVERRIDE
	{

		Mat m1 = _m1.getMat(), m2 = _m2.getMat();
		int i, count = m1.checkVector(2);
		const Point2f* M = m1.ptr<Point2f>();
		const Point2f* m = m2.ptr<Point2f>();

		double LtL[9][9], W[9][1], V[9][9];
		Mat _LtL(9, 9, CV_64F, &LtL[0][0]);
		Mat matW(9, 1, CV_64F, W);
		Mat matV(9, 9, CV_64F, V);
		Mat _H0(3, 3, CV_64F, V[8]);
		Mat _Htemp(3, 3, CV_64F, V[7]);
		//cm表示位移量，sm表示尺度
		Point2d cM(0, 0), cm(0, 0), sM(0, 0), sm(0, 0);
		//计算位移量
		for (i = 0; i < count; i++)
		{
			cm.x += m[i].x; cm.y += m[i].y;
			cM.x += M[i].x; cM.y += M[i].y;
		}

		cm.x /= count;
		cm.y /= count;
		cM.x /= count;
		cM.y /= count;

		for (i = 0; i < count; i++)
		{
			sm.x += fabs(m[i].x - cm.x);
			sm.y += fabs(m[i].y - cm.y);
			sM.x += fabs(M[i].x - cM.x);
			sM.y += fabs(M[i].y - cM.y);
		}
		if (fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
			fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON)
			return 0;
		sm.x = count / sm.x; sm.y = count / sm.y;
		sM.x = count / sM.x; sM.y = count / sM.y;

		double invHnorm[9] = { 1. / sm.x, 0, cm.x, 0, 1. / sm.y, cm.y, 0, 0, 1 };
		double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };

		Mat _invHnorm(3, 3, CV_64FC1, invHnorm);
		Mat _Hnorm2(3, 3, CV_64FC1, Hnorm2);
		_LtL.setTo(Scalar::all(0));
		//LtL是_LTL的引用
		for (i = 0; i < count; i++)
		{
			double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
			double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
			double Lx[] = { X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x };
			double Ly[] = { 0, 0, 0, X, Y, 1, -y * X, -y * Y, -y };
			int j, k;
			for (j = 0; j < 9; j++)
				for (k = j; k < 9; k++)
					LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
		}
		//把ATA的右上角元素复制到右下角，构成完整的ATA
		completeSymm(_LtL);
		//获得其特征值
		//matW是特征值，9行一列
		//matV是特征向量，9 * 9
		eigen(_LtL, matW, matV);
		_Htemp = _invHnorm * _H0;
		_H0 = _Htemp * _Hnorm2;
		_H0.convertTo(_model, _H0.type(), 1. / _H0.at<double>(2, 2));
		return 1;
	}

	/**
	 * Compute the reprojection error.
	 * m2 = H*m1
	 * @param _m1 depth CV_32F, 1-channel with 2 columns or 2-channel with 1 column
	 * @param _m2 depth CV_32F, 1-channel with 2 columns or 2-channel with 1 column
	 * @param _model CV_64FC1, 3x3
	 * @param _err, output, CV_32FC1, square of the L2 norm
	 */
	void computeError(InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err) const CV_OVERRIDE
	{
		Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
		int i, count = m1.checkVector(2);
		const Point2f* M = m1.ptr<Point2f>();
		const Point2f* m = m2.ptr<Point2f>();
		const double* H = model.ptr<double>();
		float Hf[] = { (float)H[0], (float)H[1], (float)H[2], (float)H[3], (float)H[4], (float)H[5], (float)H[6], (float)H[7] };

		_err.create(count, 1, CV_32F);
		float* err = _err.getMat().ptr<float>();

		for (i = 0; i < count; i++)
		{
			float ww = 1.f / (Hf[6] * M[i].x + Hf[7] * M[i].y + 1.f);
			float dx = (Hf[0] * M[i].x + Hf[1] * M[i].y + Hf[2])*ww - m[i].x;
			float dy = (Hf[3] * M[i].x + Hf[4] * M[i].y + Hf[5])*ww - m[i].y;
			err[i] = dx * dx + dy * dy;
		}
	}
};
class HomographyRefineCallback CV_FINAL : public LMSolver::Callback
{
public:
	HomographyRefineCallback(InputArray _src, InputArray _dst)
	{
		src = _src.getMat();
		dst = _dst.getMat();
	}

	bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const CV_OVERRIDE
	{
		int i, count = src.checkVector(2);
		Mat param = _param.getMat();
		_err.create(count * 2, 1, CV_64F);
		Mat err = _err.getMat(), J;
		if (_Jac.needed())
		{
			_Jac.create(count * 2, param.rows, CV_64F);
			J = _Jac.getMat();
			CV_Assert(J.isContinuous() && J.cols == 8);
		}

		const Point2f* M = src.ptr<Point2f>();
		const Point2f* m = dst.ptr<Point2f>();
		const double* h = param.ptr<double>();
		double* errptr = err.ptr<double>();
		double* Jptr = J.data ? J.ptr<double>() : 0;

		for (i = 0; i < count; i++)
		{
			double Mx = M[i].x, My = M[i].y;
			double ww = h[6] * Mx + h[7] * My + 1.;
			ww = fabs(ww) > DBL_EPSILON ? 1. / ww : 0;
			double xi = (h[0] * Mx + h[1] * My + h[2])*ww;
			double yi = (h[3] * Mx + h[4] * My + h[5])*ww;
			errptr[i * 2] = xi - m[i].x;
			errptr[i * 2 + 1] = yi - m[i].y;

			if (Jptr)
			{
				Jptr[0] = Mx * ww; Jptr[1] = My * ww; Jptr[2] = ww;
				Jptr[3] = Jptr[4] = Jptr[5] = 0.;
				Jptr[6] = -Mx * ww*xi; Jptr[7] = -My * ww*xi;
				Jptr[8] = Jptr[9] = Jptr[10] = 0.;
				Jptr[11] = Mx * ww; Jptr[12] = My * ww; Jptr[13] = ww;
				Jptr[14] = -Mx * ww*yi; Jptr[15] = -My * ww*yi;

				Jptr += 16;
			}
		}

		return true;
	}

	Mat src, dst;
};

class LMSolverImpl1 CV_FINAL : public LMSolver
{
public:
	LMSolverImpl1() : maxIters(100) { init(); }
	LMSolverImpl1(const Ptr<LMSolver::Callback>& _cb, int _maxIters) : cb(_cb), maxIters(_maxIters) { init(); }

	void init()
	{
		epsx = epsf = FLT_EPSILON;
		printInterval = 0;
	}

	int run(InputOutputArray _param0) const CV_OVERRIDE
	{

		Mat param0 = _param0.getMat(), x, xd, r, rd, J, A, Ap, v, temp_d, d;
		//maxIters = 10；
		int ptype = param0.type();

		CV_Assert((param0.cols == 1 || param0.rows == 1) && (ptype == CV_32F || ptype == CV_64F));
		CV_Assert(cb);

		int lx = param0.rows + param0.cols - 1;
		param0.convertTo(x, CV_64F);
		if (x.cols != 1)
			transpose(x, x);

		if (!cb->compute(x, r, J))
			return -1;
		double S = norm(r, NORM_L2SQR);
		int nfJ = 2;

		mulTransposed(J, A, true);
		gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);

		Mat D = A.diag().clone();

		const double Rlo = 0.25, Rhi = 0.75;
		double lambda = 1, lc = 0.75;
		int i, iter = 0;

		if (printInterval != 0)
		{
			printf("************************************************************************************\n");
			printf("\titr\tnfJ\t\tSUM(r^2)\t\tx\t\tdx\t\tl\t\tlc\n");
			printf("************************************************************************************\n");
		}

		for (;; )
		{
			CV_Assert(A.type() == CV_64F && A.rows == lx);
			A.copyTo(Ap);
			for (i = 0; i < lx; i++)
				Ap.at<double>(i, i) += lambda * D.at<double>(i);
			solve(Ap, v, d, DECOMP_EIG);
			subtract(x, d, xd);
			if (!cb->compute(xd, rd, noArray()))
				return -1;
			nfJ++;
			double Sd = norm(rd, NORM_L2SQR);
			gemm(A, d, -1, v, 2, temp_d);
			double dS = d.dot(temp_d);
			double R = (S - Sd) / (fabs(dS) > DBL_EPSILON ? dS : 1);

			if (R > Rhi)
			{
				lambda *= 0.5;
				if (lambda < lc)
					lambda = 0;
			}
			else if (R < Rlo)
			{
				// find new nu if R too low
				double t = d.dot(v);
				double nu = (Sd - S) / (fabs(t) > DBL_EPSILON ? t : 1) + 2;
				nu = std::min(std::max(nu, 2.), 10.);
				if (lambda == 0)
				{
					invert(A, Ap, DECOMP_EIG);
					double maxval = DBL_EPSILON;
					for (i = 0; i < lx; i++)
						maxval = std::max(maxval, std::abs(Ap.at<double>(i, i)));
					lambda = lc = 1. / maxval;
					nu *= 0.5;
				}
				lambda *= nu;
			}

			if (Sd < S)
			{
				nfJ++;
				S = Sd;
				std::swap(x, xd);
				if (!cb->compute(x, r, J))
					return -1;
				mulTransposed(J, A, true);
				gemm(J, r, 1, noArray(), 0, v, GEMM_1_T);
			}

			iter++;
			bool proceed = iter < maxIters && norm(d, NORM_INF) >= epsx && norm(r, NORM_INF) >= epsf;

			if (printInterval != 0 && (iter % printInterval == 0 || iter == 1 || !proceed))
			{
				printf("%c%10d %10d %15.4e %16.4e %17.4e %16.4e %17.4e\n",
					(proceed ? ' ' : '*'), iter, nfJ, S, x.at<double>(0), d.at<double>(0), lambda, lc);
			}

			if (!proceed)
				break;
		}

		if (param0.size != x.size)
			transpose(x, x);

		x.convertTo(param0, ptype);
		if (iter == maxIters)
			iter = -iter;

		return iter;
	}

	void setCallback(const Ptr<LMSolver::Callback>& _cb) CV_OVERRIDE { cb = _cb; }

	Ptr<LMSolver::Callback> cb;

	double epsx;
	double epsf;
	int maxIters;
	int printInterval;
};


Ptr<LMSolver> createLMSolver1(const Ptr<LMSolver::Callback>& cb, int maxIters)
{
	return makePtr<LMSolverImpl1>(cb, maxIters);
}



typedef std::set<std::pair<int, int> > MatchesSet;
Mat findHomography2(InputArray _points1, InputArray _points2,
	int method, double ransacReprojThreshold, OutputArray _mask,
	const int maxIters, const double confidence)
{
	const double defaultRANSACReprojThreshold = 3;
	bool result = false;

	Mat points1 = _points1.getMat(), points2 = _points2.getMat();
	Mat src, dst, H, tempMask;
	int npoints = -1;

	for (int i = 1; i <= 2; i++)
	{
		Mat& p = i == 1 ? points1 : points2;
		Mat& m = i == 1 ? src : dst;
		npoints = p.checkVector(2, -1, false);
		if (npoints < 0)
		{
			npoints = p.checkVector(3, -1, false);
			if (npoints < 0)
				CV_Error(Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
			if (npoints == 0)
				return Mat();
			convertPointsFromHomogeneous(p, p);
		}
		p.reshape(2, npoints).convertTo(m, CV_32F);
	}
	cout << src.cols << " " << src.rows << endl;
	//for (int i = 0; i < src.rows; ++i)
	//{
		//cout << src.at<Vec2f>(i) << " " << dst.at<Vec2f>(i) << endl;
	//}
	CV_Assert(src.checkVector(2) == dst.checkVector(2));

	if (ransacReprojThreshold <= 0)
		ransacReprojThreshold = defaultRANSACReprojThreshold;

	Ptr<PointSetRegistrator::Callback> cb = makePtr<HomographyEstimatorCallback>();

	if (method == 0 || npoints == 4)
	{
		tempMask = Mat::ones(npoints, 1, CV_8U);
		result = cb->runKernel(src, dst, H) > 0;
	}
	else if (method == RANSAC)
		result = createRANSACPointSetRegistrator1(cb, 4, ransacReprojThreshold, confidence, maxIters)->run(src, dst, H, tempMask);
	else if (method == LMEDS)
		result = createLMeDSPointSetRegistrator(cb, 4, confidence, maxIters)->run(src, dst, H, tempMask);
	else if (method == RHO)
		;//result = createAndRunRHORegistrator(confidence, maxIters, ransacReprojThreshold, npoints, src, dst, H, tempMask);
	else
		CV_Error(Error::StsBadArg, "Unknown estimation method");
	cout << src.size() << endl;
	cout << dst.size() << endl;
	cout << "H: " << H << endl;
	if (result && npoints > 4 && method != RHO)
	{
		compressElems(src.ptr<Point2f>(), tempMask.ptr<uchar>(), 1, npoints);
		npoints = compressElems(dst.ptr<Point2f>(), tempMask.ptr<uchar>(), 1, npoints);
		if (npoints > 0)
		{
			Mat src1 = src.rowRange(0, npoints);
			Mat dst1 = dst.rowRange(0, npoints);
			src = src1;
			dst = dst1;
			if (method == RANSAC || method == LMEDS)
				cb->runKernel(src, dst, H);
			Mat H8(8, 1, CV_64F, H.ptr<double>());
			//H8是一个八行一列的值
			//创建了一个临时对象，并且进行了初始化
			createLMSolver1(makePtr<HomographyRefineCallback>(src, dst), 10)->run(H8);
		}
	}
	cout << src.size() << endl;
	cout << dst.size() << endl;

	if (result)
	{
		if (_mask.needed())
			tempMask.copyTo(_mask);
	}
	else
	{
		H.release();
		if (_mask.needed()) {
			tempMask = Mat::zeros(npoints >= 0 ? npoints : 0, 1, CV_8U);
			tempMask.copyTo(_mask);
		}
	}
	return H;

}

Mat findHomography1(InputArray _points1, InputArray _points2,
	OutputArray _mask, int method, double ransacReprojThreshold)
{
	return  findHomography2(_points1, _points2, method, ransacReprojThreshold, _mask, 2000, 0.995);
}

class FeaturesMatcher1
{
public:
	virtual ~FeaturesMatcher1() {}
	void operator ()(const ImageFeatures &features1, const ImageFeatures &features2,
		MatchesInfo& matches_info) {
		match(features1, features2, matches_info);
	}
	void operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches,
		const cv::UMat &mask = cv::UMat());
	bool isThreadSafe() const { return is_thread_safe_; }
	virtual void collectGarbage() {}

protected:
	FeaturesMatcher1(bool is_thread_safe = false) : is_thread_safe_(is_thread_safe) {}
	virtual void match(const ImageFeatures &features1, const ImageFeatures &features2,
		MatchesInfo& matches_info) = 0;
	bool is_thread_safe_;
};
class CpuMatcher1 : public FeaturesMatcher1
{
public:
	CpuMatcher1(float match_conf) : FeaturesMatcher1(true), match_conf_(match_conf) {}
	void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info);

private:
	float match_conf_;
};
struct MatchPairsBody1 : ParallelLoopBody
{
	MatchPairsBody1(FeaturesMatcher1 &_matcher, const std::vector<ImageFeatures> &_features,
		std::vector<MatchesInfo> &_pairwise_matches, std::vector<std::pair<int, int> > &_near_pairs)
		: matcher(_matcher), features(_features),
		pairwise_matches(_pairwise_matches), near_pairs(_near_pairs) {}

	void operator ()(const Range &r) const
	{
		cv::RNG rng = cv::theRNG(); // save entry rng state
		const int num_images = static_cast<int>(features.size());
		for (int i = r.start; i < r.end; ++i)
		{
			cv::theRNG() = cv::RNG(rng.state + i); // force "stable" RNG seed for each processed pair

			int from = near_pairs[i].first;
			int to = near_pairs[i].second;
			int pair_idx = from * num_images + to;

			matcher(features[from], features[to], pairwise_matches[pair_idx]);
			pairwise_matches[pair_idx].src_img_idx = from;
			pairwise_matches[pair_idx].dst_img_idx = to;

			size_t dual_pair_idx = to * num_images + from;

			pairwise_matches[dual_pair_idx] = pairwise_matches[pair_idx];
			pairwise_matches[dual_pair_idx].src_img_idx = to;
			pairwise_matches[dual_pair_idx].dst_img_idx = from;

			if (!pairwise_matches[pair_idx].H.empty())
				pairwise_matches[dual_pair_idx].H = pairwise_matches[pair_idx].H.inv();

			for (size_t j = 0; j < pairwise_matches[dual_pair_idx].matches.size(); ++j)
				std::swap(pairwise_matches[dual_pair_idx].matches[j].queryIdx,
					pairwise_matches[dual_pair_idx].matches[j].trainIdx);
		}
	}

	FeaturesMatcher1 &matcher;
	const std::vector<ImageFeatures> &features;
	std::vector<MatchesInfo> &pairwise_matches;
	std::vector<std::pair<int, int> > &near_pairs;

private:
	//void operator =(const MatchPairsBody1&);
};



class BestOf2NearestMatcher1 : public FeaturesMatcher1
{
public:
	BestOf2NearestMatcher1(bool try_use_gpu = false, float match_conf = 0.3f, int num_matches_thresh1 = 6,
		int num_matches_thresh2 = 6);

	//void collectGarbage();

protected:
	void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

	int num_matches_thresh1_;
	int num_matches_thresh2_;
	Ptr<FeaturesMatcher1> impl_;
};

void FeaturesMatcher1::operator ()(const std::vector<ImageFeatures> &features, std::vector<MatchesInfo> &pairwise_matches,
	const UMat &mask)
{
	const int num_images = static_cast<int>(features.size());

	CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.cols == num_images && mask.rows));
	Mat_<uchar> mask_(mask.getMat(ACCESS_READ));
	if (mask_.empty())
		mask_ = Mat::ones(num_images, num_images, CV_8U);

	std::vector<std::pair<int, int> > near_pairs;
	for (int i = 0; i < num_images - 1; ++i)
		for (int j = i + 1; j < num_images; ++j)
			if (features[i].keypoints.size() > 0 && features[j].keypoints.size() > 0 && mask_(i, j))
				near_pairs.push_back(std::make_pair(i, j));

	pairwise_matches.resize(num_images * num_images);
	MatchPairsBody1 body(*this, features, pairwise_matches, near_pairs);
	if (is_thread_safe_)
		parallel_for_(Range(0, static_cast<int>(near_pairs.size())), body);
	else
		body(Range(0, static_cast<int>(near_pairs.size())));

}

BestOf2NearestMatcher1::BestOf2NearestMatcher1(bool try_use_gpu, float match_conf, int num_matches_thresh1, int num_matches_thresh2)
{


	impl_ = makePtr<CpuMatcher1>(match_conf);

	is_thread_safe_ = impl_->isThreadSafe();
	num_matches_thresh1_ = num_matches_thresh1;
	num_matches_thresh2_ = num_matches_thresh2;
}

void BestOf2NearestMatcher1::match(const ImageFeatures &features1, const ImageFeatures &features2,
	MatchesInfo &matches_info)
{
	//CV_INSTRUMENT_REGION();
	(*impl_)(features1, features2, matches_info);
	// Check if it makes sense to find homography
	if (matches_info.matches.size() < static_cast<size_t>(num_matches_thresh1_)) //6
		return;
	cout << matches_info.matches.size() << endl;  //252
	// Construct point-point correspondences for homography estimation
	Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
	Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
	for (size_t i = 0; i < matches_info.matches.size(); ++i)
	{
		const DMatch& m = matches_info.matches[i];

		Point2f p = features1.keypoints[m.queryIdx].pt;
		p.x -= features1.img_size.width * 0.5f;
		p.y -= features1.img_size.height * 0.5f;
		src_points.at<Point2f>(0, static_cast<int>(i)) = p;

		p = features2.keypoints[m.trainIdx].pt;
		p.x -= features2.img_size.width * 0.5f;
		p.y -= features2.img_size.height * 0.5f;
		dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
	}
	//把这些点列出来
	for (int i = 0; i < src_points.cols; ++i)
	{
		cout << src_points.at<Vec2f>(i) << " " << dst_points.at<Vec2f>(i) << endl;
	}
	// Find pair-wise motion
	matches_info.H = findHomography1(src_points, dst_points, matches_info.inliers_mask, RANSAC, 3);
	if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
		return;

	// Find number of inliers
	matches_info.num_inliers = 0;
	for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
		if (matches_info.inliers_mask[i])
			matches_info.num_inliers++;

	// These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
	// using Invariant Features"
	matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

	// Set zero confidence to remove matches between too close images, as they don't provide
	// additional information anyway. The threshold was set experimentally.
	matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

	// Check if we should try to refine motion
	if (matches_info.num_inliers < num_matches_thresh2_)
		return;

	// Construct point-point correspondences for inliers only
	src_points.create(1, matches_info.num_inliers, CV_32FC2);
	dst_points.create(1, matches_info.num_inliers, CV_32FC2);
	int inlier_idx = 0;
	for (size_t i = 0; i < matches_info.matches.size(); ++i)
	{
		if (!matches_info.inliers_mask[i])
			continue;

		const DMatch& m = matches_info.matches[i];

		Point2f p = features1.keypoints[m.queryIdx].pt;
		p.x -= features1.img_size.width * 0.5f;
		p.y -= features1.img_size.height * 0.5f;
		src_points.at<Point2f>(0, inlier_idx) = p;

		p = features2.keypoints[m.trainIdx].pt;
		p.x -= features2.img_size.width * 0.5f;
		p.y -= features2.img_size.height * 0.5f;
		dst_points.at<Point2f>(0, inlier_idx) = p;

		inlier_idx++;
	}

	// Rerun motion estimation on inliers only
	matches_info.H = findHomography2(src_points, dst_points, RANSAC, 3, noArray(), 2000, 0.995);
	cout << matches_info.H << endl;
}

void CpuMatcher1::match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo& matches_info)
{
	CV_Assert(features1.descriptors.type() == features2.descriptors.type());
	CV_Assert(features2.descriptors.depth() == CV_8U || features2.descriptors.depth() == CV_32F);
	matches_info.matches.clear();
	Ptr<cv::DescriptorMatcher> matcher;
	{
		Ptr<flann::IndexParams> indexParams = makePtr<flann::KDTreeIndexParams>();
		Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>();

		if (features2.descriptors.depth() == CV_8U)
		{
			indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
			searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
		}

		matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);
	}
	std::vector< std::vector<DMatch> > pair_matches;
	MatchesSet matches;

	// Find 1->2 matches
	matcher->knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);
	//cout << "pair_matches.size(): " << pair_matches.size() << endl; //1530
	int count = 0;
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		//cout << "queryIdx: " << m0.queryIdx << "  trainIdx: " << m1.trainIdx << endl;// << " distance0: " << m0.distance << " distance1: " << m1.distance
			//<< "m0.distance < (1.f - match_conf_) * m1.distance: " << (m0.distance < (1.f - match_conf_) * m1.distance) << endl;

		if (m0.distance < (1.f - match_conf_) * m1.distance)
		{
			count++;
			matches_info.matches.push_back(m0);
			matches.insert(std::make_pair(m0.queryIdx, m0.trainIdx));
		}

	}
	//cout << "初步找到的匹配点对数目: " << count << endl; //231

	// Find 2->1 matches
	pair_matches.clear();
	matcher->knnMatch(features2.descriptors, features1.descriptors, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - match_conf_) * m1.distance)
			if (matches.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
				matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
	}

}


int main(int argc, char** argv)
{
	int num_images = 2;
	vector<Mat> imgs;    //输入图像
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\00.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\11.bmp");
	imgs.push_back(img);
	cout << imgs[0].type() << endl;
	Ptr<OrbFeaturesFinder> finder;    //定义特征寻找器
	finder = new  OrbFeaturesFinder();    //应用ORB方法寻找特征
	vector<ImageFeatures> features(num_images);    //表示图像特征
	for (int i = 0; i < num_images; i++)
		(*finder)(imgs[i], features[i]);    //特征检测
	vector<MatchesInfo> pairwise_matches;    //表示特征匹配信息变量
	BestOf2NearestMatcher1 matcher(false, 0.3f, 6, 6);    //定义特征匹配器，2NN方法
	matcher(features, pairwise_matches);    //进行特征匹配
	cout << "pairwise_matches0: " << pairwise_matches[0].H << endl;
	cout << "pairwise_matches1: " << pairwise_matches[1].H << endl;
	cout << "pairwise_matches1: " << pairwise_matches[2].H << endl;
	cout << "pairwise_matches1: " << pairwise_matches[3].H << endl;

	//cout << "diff1=" << t2 - t1 << endl;
	HomographyBasedEstimator estimator;    //定义参数评估器
	vector<CameraParams> cameras;    //表示相机参数
	estimator(features, pairwise_matches, cameras);    //进行相机参数评估

	for (size_t i = 0; i < cameras.size(); ++i)    //转换相机旋转参数的数据类型
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	cout << cameras.size() << endl; //7,不应该是6个吗
	cout << cameras[0].R << endl;

	Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
	//adjuster = new detail::BundleAdjusterReproj();    //重映射误差方法
	adjuster = new detail::BundleAdjusterRay();    //射线发散误差方法

	adjuster->setConfThresh(1);    //设置匹配置信度，该值设为1
	(*adjuster)(features, pairwise_matches, cameras);    //精确评估相机参数

	//vector<Mat> rmats;
	//for (size_t i = 0; i < cameras.size(); ++i)    //复制相机的旋转参数
		//rmats.push_back(cameras[i].R.clone());
	//waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //进行波形校正
	//for (size_t i = 0; i < cameras.size(); ++i)    //相机参数赋值
	//	cameras[i].R = rmats[i];
	//rmats.clear();    //清变量

	vector<Point> corners(num_images);    //表示映射变换后图像的左上角坐标
	vector<UMat> masks_warped(num_images);    //表示映射变换后的图像掩码
	vector<UMat> images_warped(num_images);    //表示映射变换后的图像
	vector<Size> sizes(num_images);    //表示映射变换后的图像尺寸
	vector<Mat> masks(num_images);    //表示源图的掩码

	for (int i = 0; i < num_images; ++i)    //初始化源图的掩码
	{
		masks[i].create(imgs[i].size(), CV_8U);    //定义尺寸大小
		masks[i].setTo(Scalar::all(255));    //全部赋值为255，表示源图的所有区域都使用
	}

	Ptr<WarperCreator> warper_creator;    //定义图像映射变换创造器
	//warper_creator = new cv::PlaneWarper();    //平面投影
	warper_creator = new cv::CylindricalWarper();    //柱面投影
	//warper_creator = new cv::SphericalWarper();    //球面投影
	//warper_creator = new cv::FisheyeWarper();    //鱼眼投影
	//warper_creator = new cv::StereographicWarper();    //立方体投影

	//定义图像映射变换器，设置映射的尺度为相机的焦距，所有相机的焦距都相同
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);    //转换相机内参数的数据类型
		//对当前图像镜像投影变换，得到变换后的图像以及该图像的左上角坐标
		corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		cout << corners[i] << endl;
		sizes[i] = images_warped[i].size();    //得到尺寸
		//得到变换后的图像掩码
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	imgs.clear();    //清变量
	masks.clear();

	//创建曝光补偿器，应用增益补偿方法
	Ptr<ExposureCompensator> compensator =
		ExposureCompensator::createDefault(ExposureCompensator::GAIN);
	compensator->feed(corners, images_warped, masks_warped);    //得到曝光补偿器
	for (int i = 0; i < num_images; ++i)    //应用曝光补偿器，对图像进行曝光补偿
	{
		compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
	}

	//在后面，我们还需要用到映射变换图的掩码masks_warped，因此这里为该变量添加一个副本masks_seam
	vector<UMat> masks_seam(num_images);
	for (int i = 0; i < num_images; i++)
		masks_warped[i].copyTo(masks_seam[i]);
	Ptr<SeamFinder> seam_finder;    //定义接缝线寻找器
   //seam_finder = new NoSeamFinder();    //无需寻找接缝线
	//seam_finder = new VoronoiSeamFinder();    //逐点法
	//seam_finder是一个类
	//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //动态规范法
	//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	//图割法
	seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
	//seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);
	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)    //图像数据类型转换
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	//得到接缝线的掩码图像masks_warped
	seam_finder->find(images_warped_f, corners, masks_warped);
	vector<Mat> images_warped_s(num_images);
	Ptr<Blender> blender;    //定义图像融合器
	//blender = Blender::createDefault(Blender::MULTI_BAND, false);    //多频段融合
	//MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));

	blender = Blender::createDefault(Blender::NO, false);    //简单融合方法
	//羽化融合方法
	blender = Blender::createDefault(Blender::FEATHER, false);
	FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
	fb->setSharpness(5);    //设置羽化锐度
	//cout << sizes[0] << endl;
	//cout << sizes[1] << endl;
	blender->prepare(corners, sizes);    //生成全景图像区域

	//在融合的时候，最重要的是在接缝线两侧进行处理，而上一步在寻找接缝线后得到的掩码的边界就是接缝线处，因此我们还需要在接缝线两侧开辟一块区域用于融合处理，这一处理过程对羽化方法尤为关键
	//应用膨胀算法缩小掩码面积
	vector<Mat> dilate_img(num_images);
	Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));    //定义结构元素
	vector<Mat> a(num_images);
	vector<Mat> b(num_images);
	vector<Mat> c(num_images);
	for (int k = 0; k < num_images; k++)
	{
		images_warped_f[k].convertTo(images_warped_s[k], CV_16S);    //改变数据类型
		dilate(masks_seam[k], masks_seam[k], element);    //膨胀运算
		//映射变换图的掩码和膨胀后的掩码相“与”，从而使扩展的区域仅仅限于接缝线两侧，其他边界处不受影响
		masks_seam[k].copyTo(a[k]);
		masks_warped[k].copyTo(b[k]);
		//masks_seam[k] = masks_seam[k] & masks_warped[k];
		c[k] = a[k] & b[k];
		c[k].copyTo(masks_seam[k]);
		blender->feed(images_warped_s[k], masks_seam[k], corners[k]);    //初始化数据
	}


	masks_seam.clear();    //清内存
	images_warped_s.clear();
	masks_warped.clear();
	images_warped_f.clear();

	Mat result, result_mask;
	//完成融合操作，得到全景图像result和它的掩码result_mask
	blender->blend(result, result_mask);

	imwrite("pano.jpg", result);    //存储全景图像
	system("pause");
	return 0;
}