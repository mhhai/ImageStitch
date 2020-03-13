#include "opencv2/core/core.hpp"
//#include "highgui.h"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
//#include "opencv2/legacy/legacy.hpp"
#include<opencv2/opencv.hpp>
#include<opencv2/opencv_modules.hpp>
//#include<opencv2/imgcodecs.hpp>
//#include<opencv2/features2d.hpp>
//#include "opencv2/stitching/detail/autocalib.hpp"
//#include "opencv2/stitching/detail/blenders.hpp"
//#include "opencv2/stitching/detail/camera.hpp"
//#include "opencv2/stitching/detail/exposure_compensate.hpp"
//#include "opencv2/stitching/detail/matchers.hpp"
//#include "opencv2/stitching/detail/motion_estimators.hpp"
//#include "opencv2/stitching/detail/seam_finders.hpp"
//#include "opencv2/stitching/detail/util.hpp"
//#include "opencv2/stitching/detail/warpers.hpp"
//#include "opencv2/stitching/warpers.hpp"

#include <iostream>
#include <fstream> 
#include <string>
#include <iomanip> 
using namespace cv;
using namespace std;
using namespace detail;

float scale = 2707.47f;
float k[9];
float rinv[9];
float r_kinv[9];
float k_rinv[9];
float t[3];
inline
void mapForward(float x, float y, float &u, float &v)
{
	float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
	float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
	float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

	u = scale * atan2f(x_, z_);
	v = scale * y_ / sqrtf(x_ * x_ + z_ * z_);
}
inline
void mapBackward(float u, float v, float &x, float &y)
{
	u /= scale;
	v /= scale;

	float x_ = sinf(u);
	float y_ = v;
	float z_ = cosf(u);

	float z;
	x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
	y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
	z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;

	if (z > 0) { x /= z; y /= z; }
	else x = y = -1;
}
void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
	float tl_uf = (std::numeric_limits<float>::max)();
	float tl_vf = (std::numeric_limits<float>::max)();
	float br_uf = -(std::numeric_limits<float>::max)();
	float br_vf = -(std::numeric_limits<float>::max)();

	float u, v;
	for (int y = 0; y < src_size.height; ++y)
	{
		for (int x = 0; x < src_size.width; ++x)
		{

			mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
			tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
			br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);
		}
	}

	dst_tl.x = static_cast<int>(tl_uf);
	dst_tl.y = static_cast<int>(tl_vf);
	dst_br.x = static_cast<int>(br_uf);
	dst_br.y = static_cast<int>(br_vf);

}

void setCameraParams(InputArray _K, InputArray _R, InputArray _T = Mat::zeros(3, 1, CV_32F))
{
	Mat K = _K.getMat(), R = _R.getMat(), T = _T.getMat();

	CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
	CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);
	CV_Assert((T.size() == Size(1, 3) || T.size() == Size(3, 1)) && T.type() == CV_32F);

	Mat_<float> K_(K);
	k[0] = K_(0, 0); k[1] = K_(0, 1); k[2] = K_(0, 2);
	k[3] = K_(1, 0); k[4] = K_(1, 1); k[5] = K_(1, 2);
	k[6] = K_(2, 0); k[7] = K_(2, 1); k[8] = K_(2, 2);

	Mat_<float> Rinv = R.t();
	rinv[0] = Rinv(0, 0); rinv[1] = Rinv(0, 1); rinv[2] = Rinv(0, 2);
	rinv[3] = Rinv(1, 0); rinv[4] = Rinv(1, 1); rinv[5] = Rinv(1, 2);
	rinv[6] = Rinv(2, 0); rinv[7] = Rinv(2, 1); rinv[8] = Rinv(2, 2);

	Mat_<float> R_Kinv = R * K.inv();
	r_kinv[0] = R_Kinv(0, 0); r_kinv[1] = R_Kinv(0, 1); r_kinv[2] = R_Kinv(0, 2);
	r_kinv[3] = R_Kinv(1, 0); r_kinv[4] = R_Kinv(1, 1); r_kinv[5] = R_Kinv(1, 2);
	r_kinv[6] = R_Kinv(2, 0); r_kinv[7] = R_Kinv(2, 1); r_kinv[8] = R_Kinv(2, 2);

	Mat_<float> K_Rinv = K * Rinv;
	k_rinv[0] = K_Rinv(0, 0); k_rinv[1] = K_Rinv(0, 1); k_rinv[2] = K_Rinv(0, 2);
	k_rinv[3] = K_Rinv(1, 0); k_rinv[4] = K_Rinv(1, 1); k_rinv[5] = K_Rinv(1, 2);
	k_rinv[6] = K_Rinv(2, 0); k_rinv[7] = K_Rinv(2, 1); k_rinv[8] = K_Rinv(2, 2);

	Mat_<float> T_(T.reshape(0, 3));
	t[0] = T_(0, 0); t[1] = T_(1, 0); t[2] = T_(2, 0);
}

Rect buildMaps(Size src_size, InputArray K, InputArray R, OutputArray _xmap, OutputArray _ymap)
{
	setCameraParams(K, R);
	Point dst_tl, dst_br;
	detectResultRoi(src_size, dst_tl, dst_br);

	_xmap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);
	_ymap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);

	Mat xmap = _xmap.getMat(), ymap = _ymap.getMat();
	float x, y;
	for (int v = dst_tl.y; v <= dst_br.y; ++v)
	{
		for (int u = dst_tl.x; u <= dst_br.x; ++u)
		{
			mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
			xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;
			ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;
		}
	}

	return Rect(dst_tl, dst_br);
}
Point warp(InputArray src, InputArray K, InputArray R, int interp_mode, int border_mode,
	OutputArray dst)
{
	UMat xmap, ymap;
	Rect dst_roi = buildMaps(src.size(), K, R, xmap, ymap);
	dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
	//测试添加
	Mat a, b, c, d;
	xmap.copyTo(a);
	ymap.copyTo(b);
	imwrite("xmap.bmp", xmap);
	imwrite("ymap.bmp", ymap);
	remap(src, dst, xmap, ymap, interp_mode, border_mode);
	src.copyTo(c);
	dst.copyTo(d);
	return dst_roi.tl();
}
int main()
{
	int num_images = 2;
	vector<Mat> imgs;    //输入图像
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\11.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\22.bmp");
	imgs.push_back(img);

	Ptr<FeaturesFinder> finder = new  OrbFeaturesFinder();    //定义特征寻找器
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
	//cout << "2: " << cameras[2].R << endl;
	//cout << "3: " << cameras[3].R << endl;
	//cout << "4: " << cameras[4].R << endl;
	//cout << "5: " << cameras[5].R << endl;
	//cout << "6: " << cameras[6].R << endl;
	//如果不用光束平差法，效果很差
	//精确相机参数和原始参数差别还是很大的



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
		//从这里开始
		corners[i] = warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();    //得到尺寸
		//得到变换后的图像掩码
		warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
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
