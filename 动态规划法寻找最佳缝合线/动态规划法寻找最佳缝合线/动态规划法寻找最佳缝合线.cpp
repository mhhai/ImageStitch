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
#include<map>
#include <iostream>
#include <fstream> 
#include <string>
#include <iomanip> 
//#include"E:\\Study Software\\opencv_3.4.4\\opencv_3.4.4_bulid\\install\\include\\opencv2\\stitching\\detail\\seam_finders.hpp"
#include"D:\Opencv3.4.2\opencv\build\include\opencv2\stitching\\detail\\seam_finders.hpp"

using namespace cv;
using namespace std;
using namespace detail;

bool getSeamTips(int comp1, int comp2, Point &p1, Point &p2);
void findComponents();
void findEdges();
void resolveConflicts(
	const Mat &image1, const Mat &image2, Point tl1, Point tl2, Mat &mask1, Mat &mask2);
void computeGradients(const Mat &image1, const Mat &image2);
bool hasOnlyOneNeighbor(int comp);
bool closeToContour(int y, int x, const Mat_<uchar> &contourMask);
void computeCosts(
	const Mat &image1, const Mat &image2, Point tl1, Point tl2,
	int comp, Mat_<float> &costV, Mat_<float> &costH);
void updateLabelsUsingSeam(
	int comp1, int comp2, const std::vector<Point> &seam, bool isHorizontalSeam);
void updateLabelsUsingSeam(
	int comp1, int comp2, const std::vector<Point> &seam, bool isHorizontalSeam);
void process(
	const Mat &image1, const Mat &image2, Point tl1, Point tl2,
	Mat &mask1, Mat &mask2);
bool estimateSeam(
	const Mat &image1, const Mat &image2, Point tl1, Point tl2, int comp,
	Point p1, Point p2, std::vector<Point> &seam, bool &isHorizontal);
class ClosePoints
{
public:
	ClosePoints(int minDist) : minDist_(minDist) {}

	bool operator() (const Point &p1, const Point &p2) const
	{
		int dist2 = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
		return dist2 < minDist_ * minDist_;
	}

private:
	int minDist_;
};

Point unionTl_, unionBr_;
Size unionSize_;
Mat_<uchar> mask1_, mask2_;
Mat_<uchar> contour1mask_, contour2mask_;
Mat_<float> gradx1_, grady1_;
Mat_<float> gradx2_, grady2_;
enum CostFunction { COLOR, COLOR_GRAD };
CostFunction costFunc_;
enum ComponentState
{
	FIRST = 1, SECOND = 2, INTERS = 4,
	INTERS_FIRST = INTERS | FIRST,
	INTERS_SECOND = INTERS | SECOND
};

int ncomps_;
Mat_<int> labels_;
std::vector<ComponentState> states_;
std::vector<Point> tls_, brs_;
std::vector<std::vector<Point> > contours_;
std::set<std::pair<int, int> > edges_;

void find(const std::vector<UMat> &src, const std::vector<Point> &corners, std::vector<UMat> &masks)
{
	//LOGLN("Finding seams...");
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	if (src.size() == 0)
		return;

	std::vector<std::pair<size_t, size_t> > pairs;
	//src.size() = 2
	//只得到一个图像对 pairs[0] = pair<0,1>;
	for (size_t i = 0; i + 1 < src.size(); ++i)
		for (size_t j = i + 1; j < src.size(); ++j)
			pairs.push_back(std::make_pair(i, j));

	{
		//把UMat格式的src转换成Mat格式的_src
		std::vector<Mat> _src(src.size());
		for (size_t i = 0; i < src.size(); ++i) _src[i] = src[i].getMat(ACCESS_READ);

		//sort(pairs.begin(), pairs.end(), ImagePairLess(_src, corners));
	}
	std::reverse(pairs.begin(), pairs.end());

	//由于只有一个pair对，所以排序算法不关键

	for (size_t i = 0; i < pairs.size(); ++i)
	{
		size_t i0 = pairs[i].first, i1 = pairs[i].second;
		//i0 = 0,  i1=1
		Mat mask0 = masks[i0].getMat(ACCESS_RW), mask1 = masks[i1].getMat(ACCESS_RW);
		process(src[i0].getMat(ACCESS_READ), src[i1].getMat(ACCESS_READ), corners[i0], corners[i1], mask0, mask1);
	}

	//LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
}


void process(
	const Mat &image1, const Mat &image2, Point tl1, Point tl2,
	Mat &mask1, Mat &mask2)
{
	//CV_INSTRUMENT_REGION();
	//CV_Assert()若括号中的表达式值为false，则返回一个错误信息。
	CV_Assert(image1.size() == mask1.size());
	CV_Assert(image2.size() == mask2.size());

	//这个只是看有没有重叠区域，用完就没用了，可以不看
	Point intersectTl(std::max(tl1.x, tl2.x), std::max(tl1.y, tl2.y));

	Point intersectBr(std::min(tl1.x + image1.cols, tl2.x + image2.cols),
		std::min(tl1.y + image1.rows, tl2.y + image2.rows));

	if (intersectTl.x >= intersectBr.x || intersectTl.y >= intersectBr.y)
		return; // there are no conflicts
	//unionT1_表示并集区域的左下角坐标(-541,-551)，unionBr_表示并集区域的右上角坐标(1345,550)
	//得到的全局图像刚好是1886*1101
	unionTl_ = Point(std::min(tl1.x, tl2.x), std::min(tl1.y, tl2.y));

	unionBr_ = Point(std::max(tl1.x + image1.cols, tl2.x + image2.cols),
		std::max(tl1.y + image1.rows, tl2.y + image2.rows));

	unionSize_ = Size(unionBr_.x - unionTl_.x, unionBr_.y - unionTl_.y);

	mask1_ = Mat::zeros(unionSize_, CV_8U);
	mask2_ = Mat::zeros(unionSize_, CV_8U);
	//mask1 = mask2 = 1886*1101
	//mask1_(0, 1, 1082, 1100)
	Mat tmp = mask1_(Rect(tl1.x - unionTl_.x, tl1.y - unionTl_.y, mask1.cols, mask1.rows));
	mask1.copyTo(tmp);
	//mask2_(800, 0, 1086, 1100)
	tmp = mask2_(Rect(tl2.x - unionTl_.x, tl2.y - unionTl_.y, mask2.cols, mask2.rows));
	mask2.copyTo(tmp);

	// find both images contour masks

	contour1mask_ = Mat::zeros(unionSize_, CV_8U);
	contour2mask_ = Mat::zeros(unionSize_, CV_8U);

	for (int y = 0; y < unionSize_.height; ++y)
	{
		for (int x = 0; x < unionSize_.width; ++x)
		{
			if (mask1_(y, x) &&
				((x == 0 || !mask1_(y, x - 1)) || (x == unionSize_.width - 1 || !mask1_(y, x + 1)) ||
				(y == 0 || !mask1_(y - 1, x)) || (y == unionSize_.height - 1 || !mask1_(y + 1, x))))
			{
				contour1mask_(y, x) = 255;
			}

			if (mask2_(y, x) &&
				((x == 0 || !mask2_(y, x - 1)) || (x == unionSize_.width - 1 || !mask2_(y, x + 1)) ||
				(y == 0 || !mask2_(y - 1, x)) || (y == unionSize_.height - 1 || !mask2_(y + 1, x))))
			{
				contour2mask_(y, x) = 255;
			}
		}
	}

	findComponents();

	findEdges();

	resolveConflicts(image1, image2, tl1, tl2, mask1, mask2);
}


void findComponents()
{
	// label all connected components and get information about them
	ncomps_ = 0;
	labels_.create(unionSize_);
	states_.clear();
	tls_.clear();
	brs_.clear();
	contours_.clear();
	for (int y = 0; y < unionSize_.height; ++y)
	{
		for (int x = 0; x < unionSize_.width; ++x)
		{
			if (mask1_(y, x) && mask2_(y, x))
				labels_(y, x) = std::numeric_limits<int>::max(); //2147483647
			else if (mask1_(y, x))
				labels_(y, x) = std::numeric_limits<int>::max() - 1;
			else if (mask2_(y, x))
				labels_(y, x) = std::numeric_limits<int>::max() - 2;
			else
				labels_(y, x) = 0;
		}
	}
	//cout << ncomps_ << endl;
	Mat b;
	labels_.copyTo(b);
	for (int y = 0; y < unionSize_.height; ++y)
	{
		for (int x = 0; x < unionSize_.width; ++x)
		{
			if (labels_(y, x) >= std::numeric_limits<int>::max() - 2)
			{
				if (labels_(y, x) == std::numeric_limits<int>::max())
					states_.push_back(INTERS);
				else if (labels_(y, x) == std::numeric_limits<int>::max() - 1)
					states_.push_back(FIRST);
				else if (labels_(y, x) == std::numeric_limits<int>::max() - 2)
					states_.push_back(SECOND);
				floodFill(labels_, Point(x, y), ++ncomps_);
				tls_.push_back(Point(x, y));
				brs_.push_back(Point(x + 1, y + 1));
				contours_.push_back(std::vector<Point>());
			}
			if (labels_(y, x))
			{
				int l = labels_(y, x);
				int ci = l - 1;

				tls_[ci].x = std::min(tls_[ci].x, x);
				tls_[ci].y = std::min(tls_[ci].y, y);
				brs_[ci].x = std::max(brs_[ci].x, x + 1);
				brs_[ci].y = std::max(brs_[ci].y, y + 1);

				if ((x == 0 || labels_(y, x - 1) != l) || (x == unionSize_.width - 1 || labels_(y, x + 1) != l) ||
					(y == 0 || labels_(y - 1, x) != l) || (y == unionSize_.height - 1 || labels_(y + 1, x) != l))
				{
					contours_[ci].push_back(Point(x, y));
				}
			}

		}

		//
	}

	//cout << tls_.size() << endl;
	//cout << brs_.size() << endl;
	//cout << tls_[0] << endl;
	//cout << tls_[1] << endl;
	//cout << tls_[2] << endl;
	//cout << brs_[0] << endl;
	//cout << brs_[1] << endl;
	//cout << brs_[2] << endl;
	//cout << "************" << endl;
	//cout << contours_[0].size() << endl;
	//cout << contours_[1].size() << endl;
	//cout << contours_[2].size() << endl;
	//放的是labels_为1的值
	/*
	cout << "contours_[0]: " << endl;

	for (int i = 0; i < contours_[0].size(); i++)
	{
		cout << contours_[0][i].x << ' '<<contours_[0][i].y<<endl;
	}
	*/
	/*
	cout << "contours_[1]: " << endl;
	for (int i = 0; i < contours_[1].size(); i++)
	{
		cout << contours_[1][i].x << ' ' << contours_[1][i].y << endl;
	}
	cout << "contours_[2]: " << endl;
	*/
	/*
	for (int i = 0; i < contours_[2].size(); i++)
	{
		cout << contours_[2][i].x << ' ' << contours_[2][i].y << endl;
	}
	*/
	//cout << contours_[0][0].x << endl;
	//cout << contours_[0][0].y << endl;
	//cout << tls_.size() << endl; //3
	//cout << brs_.size() << endl; //3
	//cout << contours_.size() << endl; //3
	//其实labels_有四个值，0，1，2，3

	//imwrite("labels.bmp", labels_);
	//Mat a;
	//labels_.copyTo(a);
	//imwrite("a.bmp", a);
	//cout << ncomps_ << endl;
}


void findEdges()
{
	// find edges between components

	std::map<std::pair<int, int>, int> wedges; // weighted edges

	for (int ci = 0; ci < ncomps_ - 1; ++ci)
	{
		for (int cj = ci + 1; cj < ncomps_; ++cj)
		{
			wedges[std::make_pair(ci, cj)] = 0;
			wedges[std::make_pair(cj, ci)] = 0;
		}
	}

	for (int ci = 0; ci < ncomps_; ++ci)
	{
		for (size_t i = 0; i < contours_[ci].size(); ++i)
		{
			int x = contours_[ci][i].x;
			int y = contours_[ci][i].y;
			int l = ci + 1;

			if (x > 0 && labels_(y, x - 1) && labels_(y, x - 1) != l)
			{
				wedges[std::make_pair(ci, labels_(y, x - 1) - 1)]++;
				wedges[std::make_pair(labels_(y, x - 1) - 1, ci)]++;
			}

			if (y > 0 && labels_(y - 1, x) && labels_(y - 1, x) != l)
			{
				wedges[std::make_pair(ci, labels_(y - 1, x) - 1)]++;
				wedges[std::make_pair(labels_(y - 1, x) - 1, ci)]++;
			}

			if (x < unionSize_.width - 1 && labels_(y, x + 1) && labels_(y, x + 1) != l)
			{
				wedges[std::make_pair(ci, labels_(y, x + 1) - 1)]++;
				wedges[std::make_pair(labels_(y, x + 1) - 1, ci)]++;
			}

			if (y < unionSize_.height - 1 && labels_(y + 1, x) && labels_(y + 1, x) != l)
			{
				wedges[std::make_pair(ci, labels_(y + 1, x) - 1)]++;
				wedges[std::make_pair(labels_(y + 1, x) - 1, ci)]++;
			}
		}
	}
	//map<pair<int, int>, int> coll;
	map<pair<int, int>, int>::iterator pos;
	//for (pos = wedges.begin(); pos != wedges.end(); ++pos)
	//{
		//cout<<pos->first.first<<' '<<pos->first.second<<' '<<
		//pos->second << endl;

	//}
	//for (auto wedges: coll) {
		//cout << wedges.second << endl;
	//}
	//std::set<std::pair<int, int> > edges_;

	edges_.clear();

	for (int ci = 0; ci < ncomps_ - 1; ++ci)
	{
		for (int cj = ci + 1; cj < ncomps_; ++cj)
		{
			std::map<std::pair<int, int>, int>::iterator itr = wedges.find(std::make_pair(ci, cj));
			if (itr != wedges.end() && itr->second > 0)
				edges_.insert(itr->first);

			itr = wedges.find(std::make_pair(cj, ci));
			if (itr != wedges.end() && itr->second > 0)
				edges_.insert(itr->first);
		}
	}
	//cout << edges_.size() << endl;  //4
	set<pair<int, int>>::iterator it;
	//for (it = edges_.begin(); it != edges_.end(); it++)
		///cout << it->first << ' ' << it->second << endl;

}


void resolveConflicts(
	const Mat &image1, const Mat &image2, Point tl1, Point tl2, Mat &mask1, Mat &mask2)
{
	if (costFunc_ == COLOR_GRAD)
		computeGradients(image1, image2);

	// resolve conflicts between components
	/*
	enum ComponentState
	{
		FIRST = 1, SECOND = 2, INTERS = 4,
		INTERS_FIRST = INTERS | FIRST,
		INTERS_SECOND = INTERS | SECOND
	};

	int ncomps_;
	Mat_<int> labels_;
	std::vector<ComponentState> states_;
	*/

	bool hasConflict = true;

	//cout <<hex << ~INTERS << endl;
	while (hasConflict)
	{
		int c1 = 0, c2 = 0;
		hasConflict = false;

		for (std::set<std::pair<int, int> >::iterator itr = edges_.begin(); itr != edges_.end(); ++itr)
		{
			c1 = itr->first;
			c2 = itr->second;
			if ((states_[c1] & INTERS) && (states_[c1] & (~INTERS)) != states_[c2])
			{

				hasConflict = true;
				break;
			}
		}
		//cout << hasConflict << endl;
		//cout << clock() << endl;
		if (hasConflict)
		{
			int l1 = c1 + 1, l2 = c2 + 1;
			//l1 = 3,l2 = 1;
			if (hasOnlyOneNeighbor(c1))
			{
				// if the first components has only one adjacent component

				for (int y = tls_[c1].y; y < brs_[c1].y; ++y)
					for (int x = tls_[c1].x; x < brs_[c1].x; ++x)
						if (labels_(y, x) == l1)
							labels_(y, x) = l2;

				states_[c1] = states_[c2] == FIRST ? SECOND : FIRST;
			}
			else
			{
				// if the first component has more than one adjacent component

				Point p1, p2;
				//cout << getSeamTips(c1, c2, p1, p2) << endl;
				if (getSeamTips(c1, c2, p1, p2))
				{
					//cout << p1.x << " " << p1.y << endl;
					//cout << p2.x << " " << p2.y << endl;
					//cout << c1 << ' ' << c2 << endl;
					std::vector<Point> seam;
					bool isHorizontalSeam;
					//cout << "eeee" << endl;
					//cout << clock() << endl;
					//cout << estimateSeam(image1, image2, tl1, tl2, c1, p1, p2, seam, isHorizontalSeam) << endl;
					//cout << tl1.x << ' ' << tl1.y << endl;
					//cout << tl2.x << ' ' << tl2.y << endl;
					if (estimateSeam(image1, image2, tl1, tl2, c1, p1, p2, seam, isHorizontalSeam))
						updateLabelsUsingSeam(c1, c2, seam, isHorizontalSeam);
					//cout << seam.size() << endl;
					//cout << clock() << endl;
					//for (int i = 0; i < seam.size(); i++)
						//cout << seam[i].x << ' ' << seam[i].y << endl;
				}

				states_[c1] = states_[c2] == FIRST ? INTERS_SECOND : INTERS_FIRST;
			}

			const int c[] = { c1, c2 };
			const int l[] = { l1, l2 };

			for (int i = 0; i < 2; ++i)
			{
				// update information about the (i+1)-th component

				int x0 = tls_[c[i]].x, x1 = brs_[c[i]].x;
				int y0 = tls_[c[i]].y, y1 = brs_[c[i]].y;

				tls_[c[i]] = Point(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
				brs_[c[i]] = Point(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());
				contours_[c[i]].clear();

				for (int y = y0; y < y1; ++y)
				{
					for (int x = x0; x < x1; ++x)
					{
						if (labels_(y, x) == l[i])
						{
							tls_[c[i]].x = std::min(tls_[c[i]].x, x);
							tls_[c[i]].y = std::min(tls_[c[i]].y, y);
							brs_[c[i]].x = std::max(brs_[c[i]].x, x + 1);
							brs_[c[i]].y = std::max(brs_[c[i]].y, y + 1);

							if ((x == 0 || labels_(y, x - 1) != l[i]) || (x == unionSize_.width - 1 || labels_(y, x + 1) != l[i]) ||
								(y == 0 || labels_(y - 1, x) != l[i]) || (y == unionSize_.height - 1 || labels_(y + 1, x) != l[i]))
							{
								contours_[c[i]].push_back(Point(x, y));
							}
						}
					}
				}
			}

			// remove edges

			edges_.erase(std::make_pair(c1, c2));
			edges_.erase(std::make_pair(c2, c1));
		}
	}

	// update masks

	int dx1 = unionTl_.x - tl1.x, dy1 = unionTl_.y - tl1.y;
	int dx2 = unionTl_.x - tl2.x, dy2 = unionTl_.y - tl2.y;

	for (int y = 0; y < mask2.rows; ++y)
	{
		for (int x = 0; x < mask2.cols; ++x)
		{
			int l = labels_(y - dy2, x - dx2);
			if (l > 0 && (states_[l - 1] & FIRST) && mask1.at<uchar>(y - dy2 + dy1, x - dx2 + dx1))
				mask2.at<uchar>(y, x) = 0;
		}
	}

	for (int y = 0; y < mask1.rows; ++y)
	{
		for (int x = 0; x < mask1.cols; ++x)
		{
			int l = labels_(y - dy1, x - dx1);
			if (l > 0 && (states_[l - 1] & SECOND) && mask2.at<uchar>(y - dy1 + dy2, x - dx1 + dx2))
				mask1.at<uchar>(y, x) = 0;
		}
	}
}


void computeGradients(const Mat &image1, const Mat &image2)
{
	CV_Assert(image1.channels() == 3 || image1.channels() == 4);
	CV_Assert(image2.channels() == 3 || image2.channels() == 4);
	//CV_Assert(costFunction() == COLOR_GRAD);

	Mat gray;

	if (image1.channels() == 3)
		cvtColor(image1, gray, COLOR_BGR2GRAY);
	else if (image1.channels() == 4)
		cvtColor(image1, gray, COLOR_BGRA2GRAY);

	Sobel(gray, gradx1_, CV_32F, 1, 0);
	Sobel(gray, grady1_, CV_32F, 0, 1);

	if (image2.channels() == 3)
		cvtColor(image2, gray, COLOR_BGR2GRAY);
	else if (image2.channels() == 4)
		cvtColor(image2, gray, COLOR_BGRA2GRAY);

	Sobel(gray, gradx2_, CV_32F, 1, 0);
	Sobel(gray, grady2_, CV_32F, 0, 1);
}


bool hasOnlyOneNeighbor(int comp)
{
	std::set<std::pair<int, int> >::iterator begin, end;
	begin = lower_bound(edges_.begin(), edges_.end(), std::make_pair(comp, std::numeric_limits<int>::min()));
	end = upper_bound(edges_.begin(), edges_.end(), std::make_pair(comp, std::numeric_limits<int>::max()));
	return ++begin == end;
}


bool closeToContour(int y, int x, const Mat_<uchar> &contourMask)
{
	const int rad = 2;

	for (int dy = -rad; dy <= rad; ++dy)
	{
		if (y + dy >= 0 && y + dy < unionSize_.height)
		{
			for (int dx = -rad; dx <= rad; ++dx)
			{
				if (x + dx >= 0 && x + dx < unionSize_.width &&
					contourMask(y + dy, x + dx))
				{
					return true;
				}
			}
		}
	}

	return false;
}


bool getSeamTips(int comp1, int comp2, Point &p1, Point &p2)
{
	CV_Assert(states_[comp1] & INTERS);

	// find special points

	std::vector<Point> specialPoints;
	int l2 = comp2 + 1;

	for (size_t i = 0; i < contours_[comp1].size(); ++i)
	{
		int x = contours_[comp1][i].x;
		int y = contours_[comp1][i].y;

		if (closeToContour(y, x, contour1mask_) &&
			closeToContour(y, x, contour2mask_) &&
			((x > 0 && labels_(y, x - 1) == l2) ||
			(y > 0 && labels_(y - 1, x) == l2) ||
				(x < unionSize_.width - 1 && labels_(y, x + 1) == l2) ||
				(y < unionSize_.height - 1 && labels_(y + 1, x) == l2)))
		{
			specialPoints.push_back(Point(x, y));
		}
	}

	if (specialPoints.size() < 2)
		return false;

	// find clusters

	std::vector<int> labels;
	cv::partition(specialPoints, labels, ClosePoints(10));

	int nlabels = *std::max_element(labels.begin(), labels.end()) + 1;
	if (nlabels < 2)
		return false;

	std::vector<Point> sum(nlabels);
	std::vector<std::vector<Point> > points(nlabels);

	for (size_t i = 0; i < specialPoints.size(); ++i)
	{
		sum[labels[i]] += specialPoints[i];
		points[labels[i]].push_back(specialPoints[i]);
	}

	// select two most distant clusters

	int idx[2] = { -1,-1 };
	double maxDist = -std::numeric_limits<double>::max();

	for (int i = 0; i < nlabels - 1; ++i)
	{
		for (int j = i + 1; j < nlabels; ++j)
		{
			double size1 = static_cast<double>(points[i].size()), size2 = static_cast<double>(points[j].size());
			double cx1 = cvRound(sum[i].x / size1), cy1 = cvRound(sum[i].y / size1);
			double cx2 = cvRound(sum[j].x / size2), cy2 = cvRound(sum[j].y / size2);

			double dist = (cx1 - cx2) * (cx1 - cx2) + (cy1 - cy2) * (cy1 - cy2);
			if (dist > maxDist)
			{
				maxDist = dist;
				idx[0] = i;
				idx[1] = j;
			}
		}
	}

	// select two points closest to the clusters' centers

	Point p[2];

	for (int i = 0; i < 2; ++i)
	{
		double size = static_cast<double>(points[idx[i]].size());
		double cx = cvRound(sum[idx[i]].x / size);
		double cy = cvRound(sum[idx[i]].y / size);

		size_t closest = points[idx[i]].size();
		double minDist = std::numeric_limits<double>::max();

		for (size_t j = 0; j < points[idx[i]].size(); ++j)
		{
			double dist = (points[idx[i]][j].x - cx) * (points[idx[i]][j].x - cx) +
				(points[idx[i]][j].y - cy) * (points[idx[i]][j].y - cy);
			if (dist < minDist)
			{
				minDist = dist;
				closest = j;
			}
		}

		p[i] = points[idx[i]][closest];
	}

	p1 = p[0];
	p2 = p[1];
	return true;
}


namespace
{

	template <typename T>
	float diffL2Square3(const Mat &image1, int y1, int x1, const Mat &image2, int y2, int x2)
	{
		const T *r1 = image1.ptr<T>(y1);
		const T *r2 = image2.ptr<T>(y2);
		return static_cast<float>(sqr(r1[3 * x1] - r2[3 * x2]) + sqr(r1[3 * x1 + 1] - r2[3 * x2 + 1]) + sqr(r1[3 * x1 + 2] - r2[3 * x2 + 2]));
	}


	template <typename T>
	float diffL2Square4(const Mat &image1, int y1, int x1, const Mat &image2, int y2, int x2)
	{
		const T *r1 = image1.ptr<T>(y1);
		const T *r2 = image2.ptr<T>(y2);
		return static_cast<float>(sqr(r1[4 * x1] - r2[4 * x2]) + sqr(r1[4 * x1 + 1] - r2[4 * x2 + 1]) +
			sqr(r1[4 * x1 + 2] - r2[4 * x2 + 2]));
	}

} // namespace


void computeCosts(
	const Mat &image1, const Mat &image2, Point tl1, Point tl2,
	int comp, Mat_<float> &costV, Mat_<float> &costH)
{
	CV_Assert(states_[comp] & INTERS);

	float(*diff)(const Mat&, int, int, const Mat&, int, int) = 0;
	if (image1.type() == CV_32FC3 && image2.type() == CV_32FC3)
		diff = diffL2Square3<float>;
	else if (image1.type() == CV_8UC3 && image2.type() == CV_8UC3)
		diff = diffL2Square3<uchar>;
	else if (image1.type() == CV_32FC4 && image2.type() == CV_32FC4)
		diff = diffL2Square4<float>;
	else if (image1.type() == CV_8UC4 && image2.type() == CV_8UC4)
		diff = diffL2Square4<uchar>;
	else
		CV_Error(Error::StsBadArg, "both images must have CV_32FC3(4) or CV_8UC3(4) type");
	int l = comp + 1;
	Rect roi(tls_[comp], brs_[comp]);
	int dx1 = unionTl_.x - tl1.x, dy1 = unionTl_.y - tl1.y;
	int dx2 = unionTl_.x - tl2.x, dy2 = unionTl_.y - tl2.y;
	const float badRegionCost = normL2(Point3f(255.f, 255.f, 255.f),
		Point3f(0.f, 0.f, 0.f));
	costV.create(roi.height, roi.width + 1);
	for (int y = roi.y; y < roi.br().y; ++y)
	{
		for (int x = roi.x; x < roi.br().x + 1; ++x)
		{
			if (labels_(y, x) == l && x > 0 && labels_(y, x - 1) == l)
			{
				float costColor = (diff(image1, y + dy1, x + dx1 - 1, image2, y + dy2, x + dx2) +
					diff(image1, y + dy1, x + dx1, image2, y + dy2, x + dx2 - 1)) / 2;
				if (costFunc_ == COLOR)
					costV(y - roi.y, x - roi.x) = costColor;
				else if (costFunc_ == COLOR_GRAD)
				{
					float costGrad = std::abs(gradx1_(y + dy1, x + dx1)) + std::abs(gradx1_(y + dy1, x + dx1 - 1)) +
						std::abs(gradx2_(y + dy2, x + dx2)) + std::abs(gradx2_(y + dy2, x + dx2 - 1)) + 1.f;
					costV(y - roi.y, x - roi.x) = costColor / costGrad;
				}
			}
			else
				costV(y - roi.y, x - roi.x) = badRegionCost;
		}
	}
	//cout << (double)costV(700, 204) << endl;

	costH.create(roi.height + 1, roi.width);

	for (int y = roi.y; y < roi.br().y + 1; ++y)
	{
		for (int x = roi.x; x < roi.br().x; ++x)
		{
			if (labels_(y, x) == l && y > 0 && labels_(y - 1, x) == l)
			{
				float costColor = (diff(image1, y + dy1 - 1, x + dx1, image2, y + dy2, x + dx2) +
					diff(image1, y + dy1, x + dx1, image2, y + dy2 - 1, x + dx2)) / 2;
				if (costFunc_ == COLOR)
					costH(y - roi.y, x - roi.x) = costColor;
				else if (costFunc_ == COLOR_GRAD)
				{
					float costGrad = std::abs(grady1_(y + dy1, x + dx1)) + std::abs(grady1_(y + dy1 - 1, x + dx1)) +
						std::abs(grady2_(y + dy2, x + dx2)) + std::abs(grady2_(y + dy2 - 1, x + dx2)) + 1.f;
					costH(y - roi.y, x - roi.x) = costColor / costGrad;
				}
			}
			else
				costH(y - roi.y, x - roi.x) = badRegionCost;
		}
	}
}


bool estimateSeam(
	const Mat &image1, const Mat &image2, Point tl1, Point tl2, int comp,
	Point p1, Point p2, std::vector<Point> &seam, bool &isHorizontal)
{
	CV_Assert(states_[comp] & INTERS);

	Mat_<float> costV, costH;
	computeCosts(image1, image2, tl1, tl2, comp, costV, costH);
	//imwrite("costV.bmp", costV);
	//imwrite("costH.bmp", costH);

	Rect roi(tls_[comp], brs_[comp]);
	Point src = p1 - roi.tl();
	Point dst = p2 - roi.tl();
	//cout << "#################"<<endl;
	//cout << src.x << ' ' << src.y << endl;
	//cout << dst.x << ' ' << dst.y << endl;
	int l = comp + 1;

	// estimate seam direction

	bool swapped = false;
	isHorizontal = std::abs(dst.x - src.x) > std::abs(dst.y - src.y);

	if (isHorizontal)
	{
		if (src.x > dst.x)
		{
			std::swap(src, dst);
			swapped = true;
		}
	}
	else if (src.y > dst.y)
	{
		swapped = true;
		std::swap(src, dst);
	}
	//不用交换
	// find optimal control

	Mat_<uchar> control = Mat::zeros(roi.size(), CV_8U);
	Mat_<uchar> reachable = Mat::zeros(roi.size(), CV_8U);
	Mat_<float> cost = Mat::zeros(roi.size(), CV_32F);

	reachable(src) = 1;
	cost(src) = 0.f;

	int nsteps;
	std::pair<float, int> steps[3];
	//cout << src.y + 1 << ' ' << roi.width << endl;
	//cout << roi.x << ' ' << roi.y << endl;
	if (isHorizontal)
	{
		for (int x = src.x + 1; x <= dst.x; ++x)
		{
			for (int y = 0; y < roi.height; ++y)
			{
				// seam follows along upper side of pixels

				nsteps = 0;

				if (labels_(y + roi.y, x + roi.x) == l)
				{
					if (reachable(y, x - 1))
						steps[nsteps++] = std::make_pair(cost(y, x - 1) + costH(y, x - 1), 1);
					if (y > 0 && reachable(y - 1, x - 1))
						steps[nsteps++] = std::make_pair(cost(y - 1, x - 1) + costH(y - 1, x - 1) + costV(y - 1, x), 2);
					if (y < roi.height - 1 && reachable(y + 1, x - 1))
						steps[nsteps++] = std::make_pair(cost(y + 1, x - 1) + costH(y + 1, x - 1) + costV(y, x), 3);
				}

				if (nsteps)
				{
					std::pair<float, int> opt = *min_element(steps, steps + nsteps);
					cost(y, x) = opt.first;
					control(y, x) = (uchar)opt.second;
					reachable(y, x) = 255;
				}
			}
		}
	}
	else
	{
		for (int y = src.y + 1; y <= dst.y; ++y)
		{
			for (int x = 0; x < roi.width; ++x)
			{
				// seam follows along left side of pixels

				nsteps = 0;

				if (labels_(y + roi.y, x + roi.x) == l)
				{
					if (reachable(y - 1, x))
						steps[nsteps++] = std::make_pair(cost(y - 1, x) + costV(y - 1, x), 1);
					if (x > 0 && reachable(y - 1, x - 1))
						steps[nsteps++] = std::make_pair(cost(y - 1, x - 1) + costV(y - 1, x - 1) + costH(y, x - 1), 2);
					if (x < roi.width - 1 && reachable(y - 1, x + 1))
						steps[nsteps++] = std::make_pair(cost(y - 1, x + 1) + costV(y - 1, x + 1) + costH(y, x), 3);
				}

				if (nsteps)
				{
					std::pair<float, int> opt = *min_element(steps, steps + nsteps);
					cost(y, x) = opt.first;
					control(y, x) = (uchar)opt.second;
					reachable(y, x) = 255;
				}
			}
		}
	}

	if (!reachable(dst))
		return false;

	// restore seam

	Point p = dst;
	//cout << "$$$$$$$$$$$$$$$$$" << endl;
	//cout << p.x << ' ' << p.y << endl;
	//cout << roi.tl() << endl;
	seam.clear();
	seam.push_back(p + roi.tl());

	if (isHorizontal)
	{
		for (; p.x != src.x; seam.push_back(p + roi.tl()))
		{
			if (control(p) == 2) p.y--;
			else if (control(p) == 3) p.y++;
			p.x--;
		}
	}
	else
	{
		for (; p.y != src.y; seam.push_back(p + roi.tl()))
		{
			if (control(p) == 2) p.x--;
			else if (control(p) == 3) p.x++;
			p.y--;
		}
	}

	if (!swapped)
		std::reverse(seam.begin(), seam.end());
	//for (int i = 0; i < seam.size(); i++)
		//cout << seam[i] << endl;
	CV_Assert(seam.front() == p1);
	CV_Assert(seam.back() == p2);
	//cout << (double)clock() << endl;
	return true;
}


void updateLabelsUsingSeam(
	int comp1, int comp2, const std::vector<Point> &seam, bool isHorizontalSeam)
{
	Mat_<int> mask = Mat::zeros(brs_[comp1].y - tls_[comp1].y,
		brs_[comp1].x - tls_[comp1].x, CV_32S);

	for (size_t i = 0; i < contours_[comp1].size(); ++i)
		mask(contours_[comp1][i] - tls_[comp1]) = 255;

	for (size_t i = 0; i < seam.size(); ++i)
		mask(seam[i] - tls_[comp1]) = 255;

	// find connected components after seam carving

	int l1 = comp1 + 1, l2 = comp2 + 1;

	int ncomps = 0;

	for (int y = 0; y < mask.rows; ++y)
		for (int x = 0; x < mask.cols; ++x)
			if (!mask(y, x) && labels_(y + tls_[comp1].y, x + tls_[comp1].x) == l1)
				floodFill(mask, Point(x, y), ++ncomps);

	for (size_t i = 0; i < contours_[comp1].size(); ++i)
	{
		int x = contours_[comp1][i].x - tls_[comp1].x;
		int y = contours_[comp1][i].y - tls_[comp1].y;

		bool ok = false;
		static const int dx[] = { -1, +1, 0, 0, -1, +1, -1, +1 };
		static const int dy[] = { 0, 0, -1, +1, -1, -1, +1, +1 };

		for (int j = 0; j < 8; ++j)
		{
			int c = x + dx[j];
			int r = y + dy[j];

			if (c >= 0 && c < mask.cols && r >= 0 && r < mask.rows &&
				mask(r, c) && mask(r, c) != 255)
			{
				ok = true;
				mask(y, x) = mask(r, c);
			}
		}

		if (!ok)
			mask(y, x) = 0;
	}

	if (isHorizontalSeam)
	{
		for (size_t i = 0; i < seam.size(); ++i)
		{
			int x = seam[i].x - tls_[comp1].x;
			int y = seam[i].y - tls_[comp1].y;

			if (y < mask.rows - 1 && mask(y + 1, x) && mask(y + 1, x) != 255)
				mask(y, x) = mask(y + 1, x);
			else
				mask(y, x) = 0;
		}
	}
	else
	{
		for (size_t i = 0; i < seam.size(); ++i)
		{
			int x = seam[i].x - tls_[comp1].x;
			int y = seam[i].y - tls_[comp1].y;

			if (x < mask.cols - 1 && mask(y, x + 1) && mask(y, x + 1) != 255)
				mask(y, x) = mask(y, x + 1);
			else
				mask(y, x) = 0;
		}
	}

	// find new components connected with the second component and
	// with other components except the ones we are working with

	std::map<int, int> connect2;
	std::map<int, int> connectOther;

	for (int i = 1; i <= ncomps; ++i)
	{
		connect2.insert(std::make_pair(i, 0));
		connectOther.insert(std::make_pair(i, 0));
	}

	for (size_t i = 0; i < contours_[comp1].size(); ++i)
	{
		int x = contours_[comp1][i].x;
		int y = contours_[comp1][i].y;

		if ((x > 0 && labels_(y, x - 1) == l2) ||
			(y > 0 && labels_(y - 1, x) == l2) ||
			(x < unionSize_.width - 1 && labels_(y, x + 1) == l2) ||
			(y < unionSize_.height - 1 && labels_(y + 1, x) == l2))
		{
			connect2[mask(y - tls_[comp1].y, x - tls_[comp1].x)]++;
		}

		if ((x > 0 && labels_(y, x - 1) != l1 && labels_(y, x - 1) != l2) ||
			(y > 0 && labels_(y - 1, x) != l1 && labels_(y - 1, x) != l2) ||
			(x < unionSize_.width - 1 && labels_(y, x + 1) != l1 && labels_(y, x + 1) != l2) ||
			(y < unionSize_.height - 1 && labels_(y + 1, x) != l1 && labels_(y + 1, x) != l2))
		{
			connectOther[mask(y - tls_[comp1].y, x - tls_[comp1].x)]++;
		}
	}

	std::vector<int> isAdjComp(ncomps + 1, 0);

	for (std::map<int, int>::iterator itr = connect2.begin(); itr != connect2.end(); ++itr)
	{
		double len = static_cast<double>(contours_[comp1].size());
		int res = 0;
		if (itr->second / len > 0.05)
		{
			std::map<int, int>::const_iterator sub = connectOther.find(itr->first);
			if (sub != connectOther.end() && (sub->second / len < 0.1))
			{
				res = 1;
			}
		}
		isAdjComp[itr->first] = res;
	}

	// update labels

	for (int y = 0; y < mask.rows; ++y)
		for (int x = 0; x < mask.cols; ++x)
			if (mask(y, x) && isAdjComp[mask(y, x)])
				labels_(y + tls_[comp1].y, x + tls_[comp1].x) = l2;
}
int main(int argc, char** argv)
{
	int num_images = 2;
	vector<Mat> imgs;    //输入图像
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\11.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\22.bmp");
	imgs.push_back(img);

	Ptr<FeaturesFinder> finder = new  OrbFeaturesFinder();    //定义特征寻找器
	//finder = new SurfFeaturesFinder();    //应用SURF方法寻找特征
	//finder = new  OrbFeaturesFinder();    //应用ORB方法寻找特征
	vector<ImageFeatures> features(num_images);    //表示图像特征
	for (int i = 0; i < num_images; i++)
		(*finder)(imgs[i], features[i]);    //特征检测
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

	warper_creator = new cv::CylindricalWarper();    //柱面投影
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
	seam_finder = new VoronoiSeamFinder();    //逐点法
	//seam_finder是一个类
	//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //动态规范法
	//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	//图割法
	//seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
	// seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);
	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)    //图像数据类型转换
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	//seam_finder->find(images_warped_f1, corners, masks_seam1);

	find(images_warped_f, corners, masks_seam);


	imwrite("images_warped_f[0].bmp", images_warped_f[0]);
	imwrite("images_warped_f[1].bmp", images_warped_f[1]);
	imwrite("mask_seam[0].bmp", masks_seam[0]);
	imwrite("mask_seam[1].bmp", masks_seam[1]);

	/*
	cout << corners[0] << endl;
	cout << corners[1] << endl;
	imwrite("images_warped_f[0].bmp", images_warped_f[0]);
	imwrite("images_warped_f[1].bmp", images_warped_f[1]);
	imwrite("mask_warped[0].bmp", masks_warped[0]);
	imwrite("mask_warped[1].bmp", masks_warped[1]);

	//通过canny边缘检测，得到掩码边界，其中有一条边界就是接缝线
	for (int k = 0; k < 2; k++)
		Canny(masks_warped[k], masks_warped[k], 3, 9, 3);

	//为了使接缝线看得更清楚，这里使用了膨胀运算来加粗边界线
	vector<Mat> dilate_img(2);
	Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));    //定义结构元素
	vector<Mat> a(2);
	for (int k = 0; k < 2; k++)    //遍历两幅图像
	{
		images_warped[k].copyTo(a[k]);
		dilate(masks_warped[k], dilate_img[k], element);    //膨胀运算
		//在映射变换图上画出接缝线，在这里只是为了呈现出的一种效果，所以并没有区分接缝线和其他掩码边界
		for (int y = 0; y < images_warped[k].rows; y++)
		{
			for (int x = 0; x < images_warped[k].cols; x++)
			{
				if (dilate_img[k].at<uchar>(y, x) == 255)    //掩码边界
				{
					//images_warped[k].at<Vec3b>(y, x)[0] = 255;
					//images_warped[k].at<Vec3b>(y, x)[1] = 0;
					//images_warped[k].at<Vec3b>(y, x)[2] = 255;
					a[k].at<Vec3b>(y, x)[0] = 255;
					a[k].at<Vec3b>(y, x)[1] = 0;
					a[k].at<Vec3b>(y, x)[2] = 255;
				}
			}
		}
	}

	imwrite("seam1.jpg", a[0]);    //存储图像
	imwrite("seam2.jpg", a[1]);
	*/
	vector<Mat> images_warped_s(num_images);
	Ptr<Blender> blender;    //定义图像融合器

	//blender = Blender::createDefault(Blender::MULTI_BAND, false);    //多频段融合
	//MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
	//mb->setNumBands(4);   //设置频段数，即金字塔层数
	//blender = Blender::createDefault(Blender::NO, false);    //简单融合方法
	//羽化融合方法
	blender = Blender::createDefault(Blender::FEATHER, false);
	FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
	fb->setSharpness(0.1);    //设置羽化锐度
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
