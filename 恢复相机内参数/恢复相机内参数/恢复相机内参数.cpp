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
//#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <iostream>
#include <fstream> 
#include <string>
#include <iomanip> 
using namespace cv;
using namespace std;
using namespace detail;

void focalsFromHomography1(const Mat& H, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
{
	CV_Assert(H.type() == CV_64F && H.size() == Size(3, 3));

	const double* h = H.ptr<double>();

	double d1, d2; // Denominators 分母
	double v1, v2; // Focal squares value candidates 焦点平方值候选

	f1_ok = true;
	d1 = h[6] * h[7];
	d2 = (h[7] - h[6]) * (h[7] + h[6]);
	v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
	v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f1 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f1 = std::sqrt(v1);
	else f1_ok = false;

	f0_ok = true;
	d1 = h[0] * h[3] + h[1] * h[4];
	d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
	v1 = -h[2] * h[5] / d1;
	v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f0 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f0 = std::sqrt(v1);
	else f0_ok = false;
}
void estimateFocal1(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches,
	std::vector<double> &focals)
{
	const int num_images = static_cast<int>(features.size());
	focals.resize(num_images);

	std::vector<double> all_focals;

	for (int i = 0; i < num_images; ++i)
	{
		for (int j = 0; j < num_images; ++j)
		{
			cout << "i*num_images + j: " << (i * num_images + j) << endl;
			const MatchesInfo &m = pairwise_matches[i*num_images + j];
			if (m.H.empty())
				continue;
			cout << "m.H: " << m.H << endl;
			double f0, f1;
			bool f0ok, f1ok;
			focalsFromHomography1(m.H, f0, f1, f0ok, f1ok);
			if (f0ok && f1ok)
				all_focals.push_back(std::sqrt(f0 * f1));
		}
	}
	cout << pairwise_matches[0].H << endl;
	cout << pairwise_matches[1].H << endl;
	cout << pairwise_matches[2].H << endl;
	cout << pairwise_matches[3].H << endl;

	if (static_cast<int>(all_focals.size()) >= num_images - 1)
	{
		double median;

		std::sort(all_focals.begin(), all_focals.end());
		if (all_focals.size() % 2 == 1)
			median = all_focals[all_focals.size() / 2];
		else
			median = (all_focals[all_focals.size() / 2 - 1] + all_focals[all_focals.size() / 2]) * 0.5;

		for (int i = 0; i < num_images; ++i)
			focals[i] = median;
	}
	else
	{
		double focals_sum = 0;
		for (int i = 0; i < num_images; ++i)
			focals_sum += features[i].img_size.width + features[i].img_size.height;
		for (int i = 0; i < num_images; ++i)
			focals[i] = focals_sum / num_images;
	}
}


struct IncDistance
{
	IncDistance(std::vector<int> &vdists) : dists(&vdists[0]) {}
	void operator ()(const GraphEdge &edge) { dists[edge.to] = dists[edge.from] + 1; }
	int* dists;
};
class  Estimator
{
public:
	virtual ~Estimator() {}
	bool operator ()(const std::vector<ImageFeatures> &features,
		const std::vector<MatchesInfo> &pairwise_matches,
		std::vector<CameraParams> &cameras)
	{
		return estimate(features, pairwise_matches, cameras);
	}

protected:
	virtual bool estimate(const std::vector<ImageFeatures> &features,
		const std::vector<MatchesInfo> &pairwise_matches,
		std::vector<CameraParams> &cameras) = 0;
};


class CV_EXPORTS HomographyBasedEstimator1 : public Estimator
{
public:
	HomographyBasedEstimator1(bool is_focals_estimated = false)
		: is_focals_estimated_(is_focals_estimated) {}

private:
	virtual bool estimate(const std::vector<ImageFeatures> &features,
		const std::vector<MatchesInfo> &pairwise_matches,
		std::vector<CameraParams> &cameras);

	bool is_focals_estimated_;
};
void findMaxSpanningTree(int num_images, const std::vector<MatchesInfo> &pairwise_matches,
	Graph &span_tree, std::vector<int> &centers)
{
	Graph graph(num_images);
	std::vector<GraphEdge> edges;

	// Construct images graph and remember its edges
	for (int i = 0; i < num_images; ++i)
	{
		for (int j = 0; j < num_images; ++j)
		{
			if (pairwise_matches[i * num_images + j].H.empty())
				continue;
			float conf = static_cast<float>(pairwise_matches[i * num_images + j].num_inliers);
			graph.addEdge(i, j, conf);
			edges.push_back(GraphEdge(i, j, conf));
		}
	}

	DisjointSets comps(num_images);
	span_tree.create(num_images);
	std::vector<int> span_tree_powers(num_images, 0);

	// Find maximum spanning tree
	sort(edges.begin(), edges.end(), std::greater<GraphEdge>());
	for (size_t i = 0; i < edges.size(); ++i)
	{
		int comp1 = comps.findSetByElem(edges[i].from);
		int comp2 = comps.findSetByElem(edges[i].to);
		if (comp1 != comp2)
		{
			comps.mergeSets(comp1, comp2);
			span_tree.addEdge(edges[i].from, edges[i].to, edges[i].weight);
			span_tree.addEdge(edges[i].to, edges[i].from, edges[i].weight);
			span_tree_powers[edges[i].from]++;
			span_tree_powers[edges[i].to]++;
		}
	}

	// Find spanning tree leafs
	std::vector<int> span_tree_leafs;
	for (int i = 0; i < num_images; ++i)
		if (span_tree_powers[i] == 1)
			span_tree_leafs.push_back(i);

	// Find maximum distance from each spanning tree vertex
	std::vector<int> max_dists(num_images, 0);
	std::vector<int> cur_dists;
	for (size_t i = 0; i < span_tree_leafs.size(); ++i)
	{
		cur_dists.assign(num_images, 0);
		span_tree.walkBreadthFirst(span_tree_leafs[i], IncDistance(cur_dists));
		for (int j = 0; j < num_images; ++j)
			max_dists[j] = std::max(max_dists[j], cur_dists[j]);
	}

	// Find min-max distance
	int min_max_dist = max_dists[0];
	for (int i = 1; i < num_images; ++i)
		if (min_max_dist > max_dists[i])
			min_max_dist = max_dists[i];

	// Find spanning tree centers
	centers.clear();
	for (int i = 0; i < num_images; ++i)
		if (max_dists[i] == min_max_dist)
			centers.push_back(i);
	CV_Assert(centers.size() > 0 && centers.size() <= 2);
}

struct CalcRotation
{
	CalcRotation(int _num_images, const std::vector<MatchesInfo> &_pairwise_matches, std::vector<CameraParams> &_cameras)
		: num_images(_num_images), pairwise_matches(&_pairwise_matches[0]), cameras(&_cameras[0]) {}

	void operator ()(const GraphEdge &edge)
	{
		int pair_idx = edge.from * num_images + edge.to;

		Mat_<double> K_from = Mat::eye(3, 3, CV_64F);
		K_from(0, 0) = cameras[edge.from].focal;
		K_from(1, 1) = cameras[edge.from].focal * cameras[edge.from].aspect;
		K_from(0, 2) = cameras[edge.from].ppx;
		K_from(1, 2) = cameras[edge.from].ppy;

		Mat_<double> K_to = Mat::eye(3, 3, CV_64F);
		K_to(0, 0) = cameras[edge.to].focal;
		K_to(1, 1) = cameras[edge.to].focal * cameras[edge.to].aspect;
		K_to(0, 2) = cameras[edge.to].ppx;
		K_to(1, 2) = cameras[edge.to].ppy;

		Mat R = K_from.inv() * pairwise_matches[pair_idx].H.inv() * K_to;
		cameras[edge.to].R = cameras[edge.from].R * R;
	}

	int num_images;
	const MatchesInfo* pairwise_matches;
	CameraParams* cameras;
};


bool HomographyBasedEstimator1::estimate(
	const std::vector<ImageFeatures> &features,
	const std::vector<MatchesInfo> &pairwise_matches,
	std::vector<CameraParams> &cameras)
{
	const int num_images = static_cast<int>(features.size());
	//焦距没有被评估
	if (!is_focals_estimated_)
	{
		// Estimate focal length and set it for all cameras
		std::vector<double> focals;
		estimateFocal1(features, pairwise_matches, focals);
		cameras.assign(num_images, CameraParams());
		for (int i = 0; i < num_images; ++i)
			cameras[i].focal = focals[i];
	}
	else
	{
		for (int i = 0; i < num_images; ++i)
		{
			cameras[i].ppx -= 0.5 * features[i].img_size.width;
			cameras[i].ppy -= 0.5 * features[i].img_size.height;
		}
	}

	// Restore global motion
	Graph span_tree;
	std::vector<int> span_tree_centers;
	findMaxSpanningTree(num_images, pairwise_matches, span_tree, span_tree_centers);
	span_tree.walkBreadthFirst(span_tree_centers[0], CalcRotation(num_images, pairwise_matches, cameras));

	// As calculations were performed under assumption that p.p. is in image center
	for (int i = 0; i < num_images; ++i)
	{
		cameras[i].ppx += 0.5 * features[i].img_size.width;
		cameras[i].ppy += 0.5 * features[i].img_size.height;
	}
	return true;
}

int main(int argc, char** argv)
{
	int num_images = 2;
	vector<Mat> imgs;    //输入图像
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\11.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\22.bmp");
	imgs.push_back(img);

	Ptr<FeaturesFinder> finder;    //定义特征寻找器
	//finder = new SurfFeaturesFinder();    //应用SURF方法寻找特征
	finder = new  OrbFeaturesFinder();    //应用ORB方法寻找特征
	vector<ImageFeatures> features(num_images);    //表示图像特征
	cout << "a" << endl;
	for (int i = 0; i < num_images; i++)
		(*finder)(imgs[i], features[i]);    //特征检测
	cout << "c" << endl;

	vector<MatchesInfo> pairwise_matches;    //表示特征匹配信息变量
	cout << "a: " << pairwise_matches.size() << endl;
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //定义特征匹配器，2NN方法
	matcher(features, pairwise_matches);    //进行特征匹配
	cout << "b: " << pairwise_matches.size() << endl;

	double t2 = clock();
	cout << "t2=" << t2 << endl;
	//cout << "diff1=" << t2 - t1 << endl;
	HomographyBasedEstimator1 estimator;    //定义参数评估器
	vector<CameraParams> cameras(2);    //表示相机参数
	cout << "cameras[0].R: " << cameras[0].R << endl;
	cout << "cameras[1].R: " << cameras[1].R << endl;
	cout << "cameras[0].focal: " << cameras[0].focal << endl;
	cout << "cameras[1].focal: " << cameras[1].focal << endl;
	cout << "cameras[0].t: " << cameras[0].t << endl;
	cout << "cameras[1].t: " << cameras[1].t << endl;
	estimator(features, pairwise_matches, cameras);    //进行相机参数评估
	cout << "cameras[0].R: " << cameras[0].R << endl;
	cout << "cameras[1].R: " << cameras[1].R << endl;
	cout << "cameras[0].focal: " << cameras[0].focal << endl;
	cout << "cameras[1].focal: " << cameras[1].focal << endl;
	cout << "cameras[0].t: " << cameras[0].t << endl;
	cout << "cameras[1].t: " << cameras[1].t << endl;

	for (size_t i = 0; i < cameras.size(); ++i)    //转换相机旋转参数的数据类型
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	cout << cameras.size() << endl; //7,不应该是6个吗
	cout << cameras[0].R << endl;
	double t3 = clock();
	cout << "t3=" << t3 << endl;
	//Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
	//adjuster = new detail::BundleAdjusterReproj();    //重映射误差方法
	//adjuster = new detail::BundleAdjusterRay();    //射线发散误差方法

	//adjuster->setConfThresh(1);    //设置匹配置信度，该值设为1
	//(*adjuster)(features, pairwise_matches, cameras);    //精确评估相机参数
	//cout << "0: " << cameras[0].R << endl;
	//cout << "1: " << cameras[1].R << endl;
	//cout << "2: " << cameras[2].R << endl;
	//cout << "3: " << cameras[3].R << endl;
	//cout << "4: " << cameras[4].R << endl;
	//cout << "5: " << cameras[5].R << endl;
	//cout << "6: " << cameras[6].R << endl;
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
	//mb->setNumBands(4);   //设置频段数，即金字塔层数


	blender = Blender::createDefault(Blender::NO, false);    //简单融合方法
	//羽化融合方法
	blender = Blender::createDefault(Blender::FEATHER, false);
	FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
	fb->setSharpness(0.1);    //设置羽化锐度
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
	double t7 = clock();
	cout << "t7=" << t7 << endl;
	system("pause");
	return 0;
}
