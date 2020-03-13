#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include<opencv2/imgcodecs.hpp>
#include<opencv2/features2d.hpp>
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
//#include<opencv2/core/opencl/ocl_defs.hpp>
#include "D:\\Opencv3.4.2\\opencv\\sources\\modules\\core\\include\\opencv2\\core\\opencl\\ocl_defs.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/hal/hal.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/hal/hal.hpp"
#include <algorithm>
using namespace cv;
using namespace std;
using namespace detail;
static void
HarrisResponses(const Mat& img, const std::vector<Rect>& layerinfo,
	std::vector<KeyPoint>& pts, int blockSize, float harris_k);
static void ICAngles(const Mat& img, const std::vector<Rect>& layerinfo,
	std::vector<KeyPoint>& pts, const std::vector<int> & u_max, int half_k);
enum { kBytes = 32, HARRIS_SCORE = 0, FAST_SCORE = 1 };
int nfeatures = 510;  //保留的最大特征点数量，这个数值的设定在matchers.cpp文件中
//OrbFeaturesFinder::OrbFeaturesFinder(Size _grid_size, int n_features, float scaleFactor, int nlevels)
//{
	//grid_size = _grid_size;
	//orb = ORB::create(n_features * (99 + grid_size.area()) / 100 / grid_size.area(), scaleFactor, nlevels);
//}   //n_features * (99 + grid_size.area()) / 100 / grid_size.area()这个参数的值是510
double scaleFactor = 1.3f;  //尺度金字塔之间各层的取样率
int nlevels = 5;  //金字塔层数
int edgeThreshold = 31;  //图像边界尺寸
int firstLevel = 0;  //原图为金字塔最底层
int wta_k = 2;   //在生成BRIEF描述符时，需要比较多少个像素点才能产生一个结果
int scoreType = HARRIS_SCORE;  //使用Harris算法计算角点的响应值
int patchSize = 31;   //生成BRIEF大小时使用patch区域的大小
int fastThreshold = 20;   //Fast算法的阈值，可以自己设定，20 is good
const float HARRIS_K = 0.04f;
Size grid_size = Size(3, 1);  //这个很重要, 该参数将待检测图像分成3个部分，每个部分检测出的特征点数目是510，
							  //一张图片检测出总的特征点数目为1500
static void computeKeyPoints(const Mat& imagePyramid,
	const UMat& uimagePyramid,
	const Mat& maskPyramid,
	const std::vector<Rect>& layerInfo,
	const UMat& ulayerInfo,
	const std::vector<float>& layerScale,
	std::vector<KeyPoint>& allKeypoints,
	int nfeatures, double scaleFactor,
	int edgeThreshold, int patchSize, int scoreType,
	bool useOCL, int fastThreshold)
{

	int i, nkeypoints, level, nlevels = (int)layerInfo.size();
	std::vector<int> nfeaturesPerLevel(nlevels);

	// fill the extractors and descriptors for the corresponding scales
	float factor = (float)(1.0 / scaleFactor);
	float ndesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)std::pow((double)factor, (double)nlevels));

	int sumFeatures = 0;
	for (level = 0; level < nlevels - 1; level++)
	{
		nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
		sumFeatures += nfeaturesPerLevel[level];
		ndesiredFeaturesPerScale *= factor;
	}
	nfeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

	// Make sure we forget about what is too close to the boundary
	//edge_threshold_ = std::max(edge_threshold_, patch_size_/2 + kKernelWidth / 2 + 2);

	// pre-compute the end of a row in a circular patch
	int halfPatchSize = patchSize / 2;
	std::vector<int> umax(halfPatchSize + 2);

	int v, v0, vmax = cvFloor(halfPatchSize * std::sqrt(2.f) / 2 + 1);
	int vmin = cvCeil(halfPatchSize * std::sqrt(2.f) / 2);
	for (v = 0; v <= vmax; ++v)
		umax[v] = cvRound(std::sqrt((double)halfPatchSize * halfPatchSize - v * v));

	// Make sure we are symmetric
	for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
	{
		while (umax[v0] == umax[v0 + 1])
			++v0;
		umax[v] = v0;
		++v0;
	}

	allKeypoints.clear();
	std::vector<KeyPoint> keypoints;
	std::vector<int> counters(nlevels);
	keypoints.reserve(nfeaturesPerLevel[0] * 2);

	for (level = 0; level < nlevels; level++)
	{
		int featuresNum = nfeaturesPerLevel[level];
		Mat img = imagePyramid(layerInfo[level]);
		Mat mask = maskPyramid.empty() ? Mat() : maskPyramid(layerInfo[level]);

		// Detect FAST features, 20 is a good threshold
		{
			Ptr<FastFeatureDetector> fd = FastFeatureDetector::create(fastThreshold, true);
			fd->detect(img, keypoints, mask);
		}

		// Remove keypoints very close to the border
		KeyPointsFilter::runByImageBorder(keypoints, img.size(), edgeThreshold);

		// Keep more points than necessary as FAST does not give amazing corners
		KeyPointsFilter::retainBest(keypoints, scoreType == HARRIS_SCORE ? 2 * featuresNum : featuresNum);

		nkeypoints = (int)keypoints.size();
		counters[level] = nkeypoints;

		float sf = layerScale[level];
		for (i = 0; i < nkeypoints; i++)
		{
			keypoints[i].octave = level;
			keypoints[i].size = patchSize * sf;
		}

		std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(allKeypoints));
	}

	std::vector<Vec3i> ukeypoints_buf;

	nkeypoints = (int)allKeypoints.size();
	if (nkeypoints == 0)
	{
		return;
	}
	Mat responses;
	UMat ukeypoints, uresponses(1, nkeypoints, CV_32F);

	// Select best features using the Harris cornerness (better scoring than FAST)
	if (scoreType == HARRIS_SCORE)
	{

		HarrisResponses(imagePyramid, layerInfo, allKeypoints, 7, HARRIS_K);

		std::vector<KeyPoint> newAllKeypoints;
		newAllKeypoints.reserve(nfeaturesPerLevel[0] * nlevels);

		int offset = 0;
		for (level = 0; level < nlevels; level++)
		{
			int featuresNum = nfeaturesPerLevel[level];
			nkeypoints = counters[level];
			keypoints.resize(nkeypoints);
			std::copy(allKeypoints.begin() + offset,
				allKeypoints.begin() + offset + nkeypoints,
				keypoints.begin());
			offset += nkeypoints;

			//cull to the final desired level, using the new Harris scores.
			KeyPointsFilter::retainBest(keypoints, featuresNum);

			std::copy(keypoints.begin(), keypoints.end(), std::back_inserter(newAllKeypoints));
		}
		std::swap(allKeypoints, newAllKeypoints);
	}

	nkeypoints = (int)allKeypoints.size();


	{
		ICAngles(imagePyramid, layerInfo, allKeypoints, umax, halfPatchSize);
	}

	for (i = 0; i < nkeypoints; i++)
	{
		float scale = layerScale[allKeypoints[i].octave];
		allKeypoints[i].pt *= scale;
	}
}
template<typename _Tp> inline void copyVectorToUMat(const std::vector<_Tp>& v, OutputArray um)
{
	if (v.empty())
		um.release();
	else
		Mat(1, (int)(v.size() * sizeof(v[0])), CV_8U, (void*)&v[0]).copyTo(um);
}

/**
 * Function that computes the Harris responses in a
 * blockSize x blockSize patch at given points in the image
 */
static void
HarrisResponses(const Mat& img, const std::vector<Rect>& layerinfo,
	std::vector<KeyPoint>& pts, int blockSize, float harris_k)
{
	CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);

	size_t ptidx, ptsize = pts.size();

	const uchar* ptr00 = img.ptr<uchar>();
	int step = (int)(img.step / img.elemSize1());
	int r = blockSize / 2;

	float scale = 1.f / ((1 << 2) * blockSize * 255.f);
	float scale_sq_sq = scale * scale * scale * scale;

	AutoBuffer<int> ofsbuf(blockSize*blockSize);
	int* ofs = ofsbuf;
	for (int i = 0; i < blockSize; i++)
		for (int j = 0; j < blockSize; j++)
			ofs[i*blockSize + j] = (int)(i*step + j);

	for (ptidx = 0; ptidx < ptsize; ptidx++)
	{
		int x0 = cvRound(pts[ptidx].pt.x);
		int y0 = cvRound(pts[ptidx].pt.y);
		int z = pts[ptidx].octave;

		const uchar* ptr0 = ptr00 + (y0 - r + layerinfo[z].y)*step + x0 - r + layerinfo[z].x;
		int a = 0, b = 0, c = 0;

		for (int k = 0; k < blockSize*blockSize; k++)
		{
			const uchar* ptr = ptr0 + ofs[k];
			int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]);
			int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]);
			a += Ix * Ix;
			b += Iy * Iy;
			c += Ix * Iy;
		}
		pts[ptidx].response = ((float)a * b - (float)c * c -
			harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void ICAngles(const Mat& img, const std::vector<Rect>& layerinfo,
	std::vector<KeyPoint>& pts, const std::vector<int> & u_max, int half_k)
{
	int step = (int)img.step1();
	size_t ptidx, ptsize = pts.size();

	for (ptidx = 0; ptidx < ptsize; ptidx++)
	{
		const Rect& layer = layerinfo[pts[ptidx].octave];
		const uchar* center = &img.at<uchar>(cvRound(pts[ptidx].pt.y) + layer.y, cvRound(pts[ptidx].pt.x) + layer.x);

		int m_01 = 0, m_10 = 0;

		// Treat the center line differently, v=0
		for (int u = -half_k; u <= half_k; ++u)
			m_10 += u * center[u];

		// Go line by line in the circular patch
		for (int v = 1; v <= half_k; ++v)
		{
			// Proceed over the two lines
			int v_sum = 0;
			int d = u_max[v];
			for (int u = -d; u <= d; ++u)
			{
				int val_plus = center[u + v * step], val_minus = center[u - v * step];
				v_sum += (val_plus - val_minus);
				m_10 += u * (val_plus + val_minus);
			}
			m_01 += v * v_sum;
		}
		pts[ptidx].angle = fastAtan2((float)m_01, (float)m_10);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void
computeOrbDescriptors(const Mat& imagePyramid, const std::vector<Rect>& layerInfo,
	const std::vector<float>& layerScale, std::vector<KeyPoint>& keypoints,
	Mat& descriptors, const std::vector<Point>& _pattern, int dsize, int wta_k)
{
	int step = (int)imagePyramid.step;
	int j, i, nkeypoints = (int)keypoints.size();

	for (j = 0; j < nkeypoints; j++)
	{
		const KeyPoint& kpt = keypoints[j];
		const Rect& layer = layerInfo[kpt.octave];
		float scale = 1.f / layerScale[kpt.octave];
		float angle = kpt.angle;

		angle *= (float)(CV_PI / 180.f);
		float a = (float)cos(angle), b = (float)sin(angle);

		const uchar* center = &imagePyramid.at<uchar>(cvRound(kpt.pt.y*scale) + layer.y,
			cvRound(kpt.pt.x*scale) + layer.x);
		float x, y;
		int ix, iy;
		const Point* pattern = &_pattern[0];
		uchar* desc = descriptors.ptr<uchar>(j);

#if 1
#define GET_VALUE(idx) \
               (x = pattern[idx].x*a - pattern[idx].y*b, \
                y = pattern[idx].x*b + pattern[idx].y*a, \
                ix = cvRound(x), \
                iy = cvRound(y), \
                *(center + iy*step + ix) )
#else
#define GET_VALUE(idx) \
            (x = pattern[idx].x*a - pattern[idx].y*b, \
            y = pattern[idx].x*b + pattern[idx].y*a, \
            ix = cvFloor(x), iy = cvFloor(y), \
            x -= ix, y -= iy, \
            cvRound(center[iy*step + ix]*(1-x)*(1-y) + center[(iy+1)*step + ix]*(1-x)*y + \
                    center[iy*step + ix+1]*x*(1-y) + center[(iy+1)*step + ix+1]*x*y))
#endif

		if (wta_k == 2)
		{
			for (i = 0; i < dsize; ++i, pattern += 16)
			{
				int t0, t1, val;
				t0 = GET_VALUE(0); t1 = GET_VALUE(1);
				val = t0 < t1;
				t0 = GET_VALUE(2); t1 = GET_VALUE(3);
				val |= (t0 < t1) << 1;
				t0 = GET_VALUE(4); t1 = GET_VALUE(5);
				val |= (t0 < t1) << 2;
				t0 = GET_VALUE(6); t1 = GET_VALUE(7);
				val |= (t0 < t1) << 3;
				t0 = GET_VALUE(8); t1 = GET_VALUE(9);
				val |= (t0 < t1) << 4;
				t0 = GET_VALUE(10); t1 = GET_VALUE(11);
				val |= (t0 < t1) << 5;
				t0 = GET_VALUE(12); t1 = GET_VALUE(13);
				val |= (t0 < t1) << 6;
				t0 = GET_VALUE(14); t1 = GET_VALUE(15);
				val |= (t0 < t1) << 7;

				desc[i] = (uchar)val;
			}
		}
		else if (wta_k == 3)
		{
			for (i = 0; i < dsize; ++i, pattern += 12)
			{
				int t0, t1, t2, val;
				t0 = GET_VALUE(0); t1 = GET_VALUE(1); t2 = GET_VALUE(2);
				val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

				t0 = GET_VALUE(3); t1 = GET_VALUE(4); t2 = GET_VALUE(5);
				val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

				t0 = GET_VALUE(6); t1 = GET_VALUE(7); t2 = GET_VALUE(8);
				val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

				t0 = GET_VALUE(9); t1 = GET_VALUE(10); t2 = GET_VALUE(11);
				val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;

				desc[i] = (uchar)val;
			}
		}
		else if (wta_k == 4)
		{
			for (i = 0; i < dsize; ++i, pattern += 16)
			{
				int t0, t1, t2, t3, u, v, k, val;
				t0 = GET_VALUE(0); t1 = GET_VALUE(1);
				t2 = GET_VALUE(2); t3 = GET_VALUE(3);
				u = 0, v = 2;
				if (t1 > t0) t0 = t1, u = 1;
				if (t3 > t2) t2 = t3, v = 3;
				k = t0 > t2 ? u : v;
				val = k;

				t0 = GET_VALUE(4); t1 = GET_VALUE(5);
				t2 = GET_VALUE(6); t3 = GET_VALUE(7);
				u = 0, v = 2;
				if (t1 > t0) t0 = t1, u = 1;
				if (t3 > t2) t2 = t3, v = 3;
				k = t0 > t2 ? u : v;
				val |= k << 2;

				t0 = GET_VALUE(8); t1 = GET_VALUE(9);
				t2 = GET_VALUE(10); t3 = GET_VALUE(11);
				u = 0, v = 2;
				if (t1 > t0) t0 = t1, u = 1;
				if (t3 > t2) t2 = t3, v = 3;
				k = t0 > t2 ? u : v;
				val |= k << 4;

				t0 = GET_VALUE(12); t1 = GET_VALUE(13);
				t2 = GET_VALUE(14); t3 = GET_VALUE(15);
				u = 0, v = 2;
				if (t1 > t0) t0 = t1, u = 1;
				if (t3 > t2) t2 = t3, v = 3;
				k = t0 > t2 ? u : v;
				val |= k << 6;

				desc[i] = (uchar)val;
			}
		}
		else
			CV_Error(Error::StsBadSize, "Wrong wta_k. It can be only 2, 3 or 4.");
#undef GET_VALUE
	}
}


static void initializeOrbPattern(const Point* pattern0, std::vector<Point>& pattern, int ntuples, int tupleSize, int poolSize)
{
	RNG rng(0x12345678);
	int i, k, k1;
	pattern.resize(ntuples*tupleSize);

	for (i = 0; i < ntuples; i++)
	{
		for (k = 0; k < tupleSize; k++)
		{
			for (;;)
			{
				int idx = rng.uniform(0, poolSize);
				Point pt = pattern0[idx];
				for (k1 = 0; k1 < k; k1++)
					if (pattern[tupleSize*i + k1] == pt)
						break;
				if (k1 == k)
				{
					pattern[tupleSize*i + k] = pt;
					break;
				}
			}
		}
	}
}

static int bit_pattern_31_[256 * 4] =
{
	8,-3, 9,5/*mean (0), correlation (0)*/,
	4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
	-11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
	7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
	2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
	1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
	-2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
	-13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
	-13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
	10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
	-13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
	-11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
	7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
	-4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
	-13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
	-9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
	12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
	-3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
	-6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
	11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
	4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
	5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
	3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
	-8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
	-2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
	-13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
	-7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
	-4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
	-10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
	5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
	5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
	1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
	9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
	4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
	2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
	-4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
	-8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
	4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
	0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
	-13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
	-3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
	-6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
	8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
	0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
	7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
	-13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
	10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
	-6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
	10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
	-13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
	-13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
	3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
	5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
	-1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
	3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
	2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
	-13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
	-13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
	-13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
	-7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
	6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
	-9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
	-2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
	-12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
	3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
	-7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
	-3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
	2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
	-11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
	-1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
	5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
	-4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
	-9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
	-12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
	10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
	7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
	-7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
	-4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
	7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
	-7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
	-13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
	-3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
	7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
	-13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
	1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
	2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
	-4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
	-1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
	7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
	1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
	9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
	-1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
	-13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
	7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
	12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
	6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
	5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
	2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
	3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
	2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
	9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
	-8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
	-11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
	1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
	6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
	2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
	6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
	3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
	7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
	-11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
	-10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
	-5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
	-10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
	8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
	4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
	-10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
	4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
	-2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
	-5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
	7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
	-9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
	-5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
	8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
	-9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
	1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
	7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
	-2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
	11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
	-12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
	3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
	5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
	0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
	-9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
	0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
	-1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
	5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
	3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
	-13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
	-5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
	-4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
	6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
	-7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
	-13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
	1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
	4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
	-2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
	2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
	-2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
	4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
	-6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
	-3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
	7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
	4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
	-13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
	7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
	7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
	-7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
	-8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
	-13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
	2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
	10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
	-6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
	8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
	2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
	-11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
	-12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
	-11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
	5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
	-2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
	-1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
	-13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
	-10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
	-3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
	2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
	-9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
	-4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
	-4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
	-6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
	6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
	-13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
	11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
	7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
	-1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
	-4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
	-7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
	-13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
	-7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
	-8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
	-5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
	-13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
	1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
	1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
	9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
	5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
	-1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
	-9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
	-1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
	-13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
	8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
	2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
	7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
	-10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
	-10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
	4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
	3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
	-4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
	5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
	4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
	-9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
	0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
	-12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
	3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
	-10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
	8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
	-8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
	2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
	10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
	6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
	-7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
	-3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
	-1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
	-3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
	-8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
	4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
	2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
	6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
	3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
	11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
	-3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
	4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
	2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
	-10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
	-13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
	-13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
	6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
	0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
	-13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
	-9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
	-13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
	5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
	2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
	-1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
	9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
	11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
	3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
	-1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
	3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
	-13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
	5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
	8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
	7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
	-10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
	7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
	9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
	7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
	-1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


static void makeRandomPattern(int patchSize, Point* pattern, int npoints)
{
	RNG rng(0x34985739); // we always start with a fixed seed,
						 // to make patterns the same on each run
	for (int i = 0; i < npoints; i++)
	{
		pattern[i].x = rng.uniform(-patchSize / 2, patchSize / 2 + 1);
		pattern[i].y = rng.uniform(-patchSize / 2, patchSize / 2 + 1);
	}
}


static inline float getScale(int level, int firstLevel, double scaleFactor)
{
	return (float)std::pow(scaleFactor, (double)(level - firstLevel));
}


void detectAndCompute(InputArray _image, InputArray _mask,
	std::vector<KeyPoint>& keypoints,
	OutputArray _descriptors, bool useProvidedKeypoints)
{


	CV_Assert(patchSize >= 2);  //做了小改动

	bool do_keypoints = !useProvidedKeypoints;
	bool do_descriptors = _descriptors.needed();

	if ((!do_keypoints && !do_descriptors) || _image.empty())
		return;

	//ROI handling
	const int HARRIS_BLOCK_SIZE = 9;
	int halfPatchSize = patchSize / 2;
	// sqrt(2.0) is for handling patch rotation
	int descPatchSize = cvCeil(halfPatchSize*sqrt(2.0));
	int border = std::max(edgeThreshold, std::max(descPatchSize, HARRIS_BLOCK_SIZE / 2)) + 1;

	bool useOCL = ocl::isOpenCLActivated();// && OCL_FORCE_CHECK(_image.isUMat() || _descriptors.isUMat());  //小改动

	Mat image = _image.getMat(), mask = _mask.getMat();
	if (image.type() != CV_8UC1)
		cvtColor(_image, image, COLOR_BGR2GRAY);

	int i, level, nLevels = nlevels, nkeypoints = (int)keypoints.size();
	bool sortedByLevel = true;

	if (!do_keypoints)
	{
		// if we have pre-computed keypoints, they may use more levels than it is set in parameters
		// !!!TODO!!! implement more correct method, independent from the used keypoint detector.
		// Namely, the detector should provide correct size of each keypoint. Based on the keypoint size
		// and the algorithm used (i.e. BRIEF, running on 31x31 patches) we should compute the approximate
		// scale-factor that we need to apply. Then we should cluster all the computed scale-factors and
		// for each cluster compute the corresponding image.
		//
		// In short, ultimately the descriptor should
		// ignore octave parameter and deal only with the keypoint size.
		nLevels = 0;
		for (i = 0; i < nkeypoints; i++)
		{
			level = keypoints[i].octave;
			CV_Assert(level >= 0);
			if (i > 0 && level < keypoints[i - 1].octave)
				sortedByLevel = false;
			nLevels = std::max(nLevels, level);
		}
		nLevels++;
	}

	std::vector<Rect> layerInfo(nLevels);
	std::vector<int> layerOfs(nLevels);
	std::vector<float> layerScale(nLevels);
	Mat imagePyramid, maskPyramid;
	UMat uimagePyramid, ulayerInfo;

	int level_dy = image.rows + border * 2;
	Point level_ofs(0, 0);
	Size bufSize((cvRound(image.cols / getScale(0, firstLevel, scaleFactor)) + border * 2 + 15) & -16, 0);

	for (level = 0; level < nLevels; level++)
	{
		float scale = getScale(level, firstLevel, scaleFactor);
		layerScale[level] = scale;
		Size sz(cvRound(image.cols / scale), cvRound(image.rows / scale));
		Size wholeSize(sz.width + border * 2, sz.height + border * 2);
		if (level_ofs.x + wholeSize.width > bufSize.width)
		{
			level_ofs = Point(0, level_ofs.y + level_dy);
			level_dy = wholeSize.height;
		}

		Rect linfo(level_ofs.x + border, level_ofs.y + border, sz.width, sz.height);
		layerInfo[level] = linfo;
		layerOfs[level] = linfo.y*bufSize.width + linfo.x;
		level_ofs.x += wholeSize.width;
	}
	bufSize.height = level_ofs.y + level_dy;

	imagePyramid.create(bufSize, CV_8U);
	if (!mask.empty())
		maskPyramid.create(bufSize, CV_8U);

	Mat prevImg = image, prevMask = mask;

	// Pre-compute the scale pyramids
	for (level = 0; level < nLevels; ++level)
	{
		Rect linfo = layerInfo[level];
		Size sz(linfo.width, linfo.height);
		Size wholeSize(sz.width + border * 2, sz.height + border * 2);
		Rect wholeLinfo = Rect(linfo.x - border, linfo.y - border, wholeSize.width, wholeSize.height);
		Mat extImg = imagePyramid(wholeLinfo), extMask;
		Mat currImg = extImg(Rect(border, border, sz.width, sz.height)), currMask;

		if (!mask.empty())
		{
			extMask = maskPyramid(wholeLinfo);
			currMask = extMask(Rect(border, border, sz.width, sz.height));
		}

		// Compute the resized image
		if (level != firstLevel)
		{
			resize(prevImg, currImg, sz, 0, 0, INTER_LINEAR_EXACT);
			if (!mask.empty())
			{
				resize(prevMask, currMask, sz, 0, 0, INTER_LINEAR_EXACT);
				if (level > firstLevel)
					threshold(currMask, currMask, 254, 0, THRESH_TOZERO);
			}

			copyMakeBorder(currImg, extImg, border, border, border, border,
				BORDER_REFLECT_101 + BORDER_ISOLATED);
			if (!mask.empty())
				copyMakeBorder(currMask, extMask, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
		}
		else
		{
			copyMakeBorder(image, extImg, border, border, border, border,
				BORDER_REFLECT_101);
			if (!mask.empty())
				copyMakeBorder(mask, extMask, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
		}
		if (level > firstLevel)
		{
			prevImg = currImg;
			prevMask = currMask;
		}
	}

	if (useOCL)
		copyVectorToUMat(layerOfs, ulayerInfo);

	if (do_keypoints)
	{
		if (useOCL)
			imagePyramid.copyTo(uimagePyramid);

		// Get keypoints, those will be far enough from the border that no check will be required for the descriptor
		computeKeyPoints(imagePyramid, uimagePyramid, maskPyramid,
			layerInfo, ulayerInfo, layerScale, keypoints,
			nfeatures, scaleFactor, edgeThreshold, patchSize, scoreType, useOCL, fastThreshold);
	}
	else
	{
		KeyPointsFilter::runByImageBorder(keypoints, image.size(), edgeThreshold);

		if (!sortedByLevel)
		{
			std::vector<std::vector<KeyPoint> > allKeypoints(nLevels);
			nkeypoints = (int)keypoints.size();
			for (i = 0; i < nkeypoints; i++)
			{
				level = keypoints[i].octave;
				CV_Assert(0 <= level);
				allKeypoints[level].push_back(keypoints[i]);
			}
			keypoints.clear();
			for (level = 0; level < nLevels; level++)
				std::copy(allKeypoints[level].begin(), allKeypoints[level].end(), std::back_inserter(keypoints));
		}
	}

	if (do_descriptors)
	{
		int dsize = 32;  //自己设置成了32

		nkeypoints = (int)keypoints.size();
		if (nkeypoints == 0)
		{
			_descriptors.release();
			return;
		}

		_descriptors.create(nkeypoints, dsize, CV_8U);
		std::vector<Point> pattern;

		const int npoints = 512;
		Point patternbuf[npoints];
		const Point* pattern0 = (const Point*)bit_pattern_31_;

		if (patchSize != 31)
		{
			pattern0 = patternbuf;
			makeRandomPattern(patchSize, patternbuf, npoints);
		}

		CV_Assert(wta_k == 2 || wta_k == 3 || wta_k == 4);

		if (wta_k == 2)
			std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));
		else
		{
			int ntuples = 32 * 4;  //自己设置的32
			initializeOrbPattern(pattern0, pattern, ntuples, wta_k, npoints);
		}

		for (level = 0; level < nLevels; level++)
		{
			// preprocess the resized image
			Mat workingMat = imagePyramid(layerInfo[level]);

			//boxFilter(working_mat, working_mat, working_mat.depth(), Size(5,5), Point(-1,-1), true, BORDER_REFLECT_101);
			GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);
		}


		{
			Mat descriptors = _descriptors.getMat();
			computeOrbDescriptors(imagePyramid, layerInfo, layerScale,
				keypoints, descriptors, pattern, dsize, wta_k);
		}
	}
}

void find(InputArray image, ImageFeatures &features)
{

	//cout << 1500 * (99 + grid_size.area()) / 100 / grid_size.area() << endl; //510
	UMat gray_image;

	CV_Assert((image.type() == CV_8UC3) || (image.type() == CV_8UC4) || (image.type() == CV_8UC1));

	if (image.type() == CV_8UC3) {
		cvtColor(image, gray_image, COLOR_BGR2GRAY);
	}
	else if (image.type() == CV_8UC4) {
		cvtColor(image, gray_image, COLOR_BGRA2GRAY);
	}
	else if (image.type() == CV_8UC1) {
		gray_image = image.getUMat();
	}
	else {
		CV_Error(Error::StsUnsupportedFormat, "");
	}

	if (grid_size.area() == 1)
		detectAndCompute(gray_image, Mat(), features.keypoints, features.descriptors, false);
	else
	{
		features.keypoints.clear();
		features.descriptors.release();

		std::vector<KeyPoint> points;
		Mat _descriptors;
		UMat descriptors;
		//下面这段代码将图像分成1 * 3的网格，对于每个网格部分进行特征点检测，
		//这样做的好处是，你可以选择只在重叠区域进行检测
		for (int r = 0; r < grid_size.height; ++r)
			for (int c = 0; c < grid_size.width; ++c)
			{
				int xl = c * gray_image.cols / grid_size.width;
				int yl = r * gray_image.rows / grid_size.height;
				int xr = (c + 1) * gray_image.cols / grid_size.width;
				int yr = (r + 1) * gray_image.rows / grid_size.height;

				// LOGLN("OrbFeaturesFinder::find: gray_image.empty=" << (gray_image.empty()?"true":"false") << ", "
				//     << " gray_image.size()=(" << gray_image.size().width << "x" << gray_image.size().height << "), "
				//     << " yl=" << yl << ", yr=" << yr << ", "
				//     << " xl=" << xl << ", xr=" << xr << ", gray_image.data=" << ((size_t)gray_image.data) << ", "
				//     << "gray_image.dims=" << gray_image.dims << "\n");
				//网格区域
				UMat gray_image_part = gray_image(Range(yl, yr), Range(xl, xr));
				// LOGLN("OrbFeaturesFinder::find: gray_image_part.empty=" << (gray_image_part.empty()?"true":"false") << ", "
				//     << " gray_image_part.size()=(" << gray_image_part.size().width << "x" << gray_image_part.size().height << "), "
				//     << " gray_image_part.dims=" << gray_image_part.dims << ", "
				//     << " gray_image_part.data=" << ((size_t)gray_image_part.data) << "\n");
				//特征点检测
				detectAndCompute(gray_image_part, UMat(), points, descriptors, false);

				features.keypoints.reserve(features.keypoints.size() + points.size());
				for (std::vector<KeyPoint>::iterator kp = points.begin(); kp != points.end(); ++kp)
				{
					kp->pt.x += xl;
					kp->pt.y += yl;
					features.keypoints.push_back(*kp);
				}
				_descriptors.push_back(descriptors.getMat(ACCESS_READ));
			}

		// TODO optimize copyTo()
		//features.descriptors = _descriptors.getUMat(ACCESS_READ);
		_descriptors.copyTo(features.descriptors);
	}
}

using namespace cv;
using namespace std;
using namespace detail;

int main(int argc, char** argv)
{
	int num_images = 2;
	vector<Mat> imgs;    //输入图像
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\00.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\11.bmp");
	imgs.push_back(img);
	imwrite("src1.bmp", imgs[0]);
	imwrite("src2.bmp", imgs[1]);

	Ptr<OrbFeaturesFinder> finder;    //定义特征寻找器
	finder = new  OrbFeaturesFinder();    //应用ORB方法寻找特征
	vector<ImageFeatures> features(num_images);    //表示图像特征
	for (int i = 0; i < num_images; i++)
		find(imgs[i], features[i]);    //特征检测
	//for (int i = 0; i < features[0].keypoints.size(); ++i)
	//{
		//cout << "检测到的特征点:" << i << " " << features[0].keypoints[i].pt << endl;
	//}
	//drawKeypoints(imgs[0], features[0].keypoints, imgs[0], Scalar(255, 0, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
	//imwrite("原始特征点.bmp", imgs[0]);
	//drawKeypoints(imgs[1], features[1].keypoints, imgs[1], Scalar(255, 0, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
	//imwrite("原始特征点2.bmp", imgs[1]);
	drawKeypoints(imgs[0], features[0].keypoints, imgs[0], Scalar(255, 0, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);
	imwrite("重构的特征点.bmp", imgs[0]);
	vector<MatchesInfo> pairwise_matches;    //表示特征匹配信息变量
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //定义特征匹配器，2NN方法
	matcher(features, pairwise_matches);    //进行特征匹配
	HomographyBasedEstimator estimator;    //定义参数评估器
	vector<CameraParams> cameras;    //表示相机参数
	estimator(features, pairwise_matches, cameras);    //进行相机参数评估

	for (size_t i = 0; i < cameras.size(); ++i)    //转换相机旋转参数的数据类型
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
	//adjuster = new detail::BundleAdjusterReproj();    //重映射误差方法
	adjuster = new detail::BundleAdjusterRay();    //射线发散误差方法

	adjuster->setConfThresh(1);    //设置匹配置信度，该值设为1
	(*adjuster)(features, pairwise_matches, cameras);    //精确评估相机参数
	//如果不用光束平差法，效果很差
	//精确相机参数和原始参数差别还是很大的

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