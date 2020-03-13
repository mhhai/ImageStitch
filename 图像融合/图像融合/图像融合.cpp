#include "opencv2/core/core.hpp"
#include "highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include<opencv2/imgcodecs.hpp>
#include<opencv2/features2d.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
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
using namespace cv;
using namespace std;
using namespace detail;

int main(int argc, char** argv)
{
	double t1 = clock();
	cout << "t1 = " << t1 << endl;
	int num_images = 2;
	vector<Mat> imgs;    //输入图像
	Mat img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\-22.bmp");
	imgs.push_back(img);
	img = imread("C:\\Users\\mhhai\\Desktop\\mh\\src_image\\IFOV\\-11.bmp");
	imgs.push_back(img);

	Ptr<FeaturesFinder> finder;    //定义特征寻找器
	//finder = new SurfFeaturesFinder();    //应用SURF方法寻找特征
	finder = new  OrbFeaturesFinder();    //应用ORB方法寻找特征
	vector<ImageFeatures> features(num_images);    //表示图像特征
	for (int i = 0; i < num_images; i++)
		(*finder)(imgs[i], features[i]);    //特征检测
	vector<MatchesInfo> pairwise_matches;    //表示特征匹配信息变量
	BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //定义特征匹配器，2NN方法
	matcher(features, pairwise_matches);    //进行特征匹配
	double t2 = clock();
	cout << "t2=" << t2 << endl;
	HomographyBasedEstimator estimator;    //定义参数评估器
	vector<CameraParams> cameras;    //表示相机参数
	estimator(features, pairwise_matches, cameras);    //进行相机参数评估

	for (size_t i = 0; i < cameras.size(); ++i)    //转换相机旋转参数的数据类型
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
	}
	double t3 = clock();
	cout << "t3=" << t3 << endl;
	Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
	//adjuster = new detail::BundleAdjusterReproj();    //重映射误差方法
	adjuster = new detail::BundleAdjusterRay();    //射线发散误差方法

	adjuster->setConfThresh(1);    //设置匹配置信度，该值设为1
	(*adjuster)(features, pairwise_matches, cameras);    //精确评估相机参数

	//如果不用光束平差法，效果很差
	//精确相机参数和原始参数差别还是很大的
	double t4 = clock();
	cout << "t4=" << t4 << endl;
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
	double t5 = clock();
	cout << "t5=" << t5 << endl;
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
	double t6 = clock();
	cout << "t6=" << t6 << endl;
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
	//	Ptr<SeamFinder> seam_finder;    //定义接缝线寻找器
	   //seam_finder = new NoSeamFinder();    //无需寻找接缝线
		//seam_finder = new VoronoiSeamFinder();    //逐点法
		//seam_finder是一个类
		//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //动态规范法
		//seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
		//图割法
		//seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
		//seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);
	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)    //图像数据类型转换
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	double t7 = getTickCount();
	//需要的一些参数
	Mat images1, images2;
	images_warped_f[0].copyTo(images1);
	images_warped_f[1].copyTo(images2);
	cout << corners[0] << endl;
	cout << corners[1] << endl;
	Point tl1 = corners[0];
	Point tl2 = corners[1];
	//计算全景图像的大小
	int panoBr_, panoHe_;
	panoBr_ = tl2.x - tl1.x + images_warped[1].cols;
	panoHe_ = max(tl1.y + images_warped[0].rows, tl2.y + images_warped[1].rows) - min(tl1.y, tl2.y);
	//计算偏移量,以图像1在全景图像左上角dx = 0；
	//偏移量用来喂数据，注意是反的像。
	int dx1, dx2;
	dx1 = 0;
	dx2 = tl2.x - tl1.x;
	int dy, dy1, dy2;
	dy1 = 0;
	dy2 = 0;
	dy = tl2.y - tl1.y;
	if (dy > 0)
	{
		dy2 = dy;
		dy1 = 0;
	}
	if (dy < 0)
	{
		dy1 = -dy;
		dy2 = 0;
	}
	Point unionTl_, unionBr_;
	//计算重叠区域的宽和高
	Point intersectTl(std::max(tl1.x, tl2.x), std::max(tl1.y, tl2.y));

	Point intersectBr(std::min(tl1.x + images1.cols, tl2.x + images2.cols),
		std::min(tl1.y + images1.rows, tl2.y + images2.rows));
	//如果if条件成立，则说明image1和image2没有重叠区域，因此退出该函数
	cout << "intersectTl: " << intersectTl << endl;
	cout << "intersectBr: " << intersectBr << endl;
	if (intersectTl.x >= intersectBr.x || intersectTl.y >= intersectBr.y)
		return 0; // there are no conflicts
	int height, width;
	height = intersectBr.y - intersectTl.y;
	width = intersectBr.x - intersectTl.x;
	cout << "height: " << height << endl;
	cout << "width: " << width << endl;
	//计算全景重叠区域的宽和高
	//计算全景重叠区域内水平和竖直方向上的梯度
	int interSectBr_ = images_warped[0].cols - dx2;
	int interSectHe_ = panoHe_;
	cout << "图像0高度: " << images_warped[0].rows << endl;
	cout << "图像0宽度: " << images_warped[0].cols << endl;
	cout << "图像1高度: " << images_warped[1].rows << endl;
	cout << "图像1宽度: " << images_warped[1].cols << endl;
	cout << "panoBr_: " << panoBr_ << endl;
	cout << "panoHe_: " << panoHe_ << endl;
	cout << "interSecBr_: " << interSectBr_ << endl;
	cout << "interSecHe_: " << interSectHe_ << endl;
	cout << "dy: " << dy << endl;
	cout << "dx1: " << dx1 << endl;
	cout << "dx2: " << dx2 << endl;
	cout << "dy1: " << dy1 << endl;
	cout << "dy2: " << dy2 << endl;
	//计算能量函数
	Mat_<float> costV;
	costV.create(interSectHe_, interSectBr_ + 2);
	costV.setTo(0);
	//使用指针计算costV
	//一种是创建掩码的方案。一种是不创建掩码的方案
	//不创建掩码
	double t11 = getTickCount();
	if (dy > 0)
	{
		for (int y = dy2; y < interSectHe_ - dy2; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y - dy2);
			float* p3 = costV.ptr<float>(y);
			for (int x = 1; x < interSectBr_ - 1; ++x)
			{
				p3[x] = ((sqr(p1[(x + dx2) * 3] - p2[x * 3]) + sqr(p1[(x + dx2) * 3 + 1] - p2[x * 3 + 1]) + sqr(p1[(x + dx2) * 3 + 2] - p2[x * 3 + 2])) +
					(sqr(p1[(x + dx2 + 1) * 3] - p2[(x - 1) * 3]) + sqr(p1[(x + dx2 + 1) * 3 + 1] - p2[(x - 1) * 3 + 1]) + sqr(p1[(x + dx2 + 1) * 3 + 2] - p2[(x - 1) * 3 + 2]))) / 2;

			}
		}
	}
	else if (dy < 0)
	{
		//cout << "a" << endl;
		for (int y = dy1; y < interSectHe_ - dy1; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y - dy1);
			float* p3 = costV.ptr<float>(y);
			for (int x = 1; x < interSectBr_ - 1; ++x)
			{
				p3[x] = ((sqr(p1[(x + dx2) * 3] - p2[x * 3]) + sqr(p1[(x + dx2) * 3 + 1] - p2[x * 3 + 1]) + sqr(p1[(x + dx2) * 3 + 2] - p2[x * 3 + 2])) +
					(sqr(p1[(x + dx2 + 1) * 3] - p2[(x - 1) * 3]) + sqr(p1[(x + dx2 + 1) * 3 + 1] - p2[(x - 1) * 3 + 1]) + sqr(p1[(x + dx2 + 1) * 3 + 2] - p2[(x - 1) * 3 + 2]))) / 2;

			}
		}
	}
	else
	{
		//没有重叠的部分做0处理
		int row = min(images1.rows, images2.rows);
		for (int y = 0; y < row; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y);
			float* p3 = costV.ptr<float>(y);
			for (int x = 1; x < interSectBr_ - 1; ++x)
			{
				p3[x] = ((sqr(p1[(x + dx2) * 3] - p2[x * 3]) + sqr(p1[(x + dx2) * 3 + 1] - p2[x * 3 + 1]) + sqr(p1[(x + dx2) * 3 + 2] - p2[x * 3 + 2])) +
					(sqr(p1[(x + dx2 + 1) * 3] - p2[(x - 1) * 3]) + sqr(p1[(x + dx2 + 1) * 3 + 1] - p2[(x - 1) * 3 + 1]) + sqr(p1[(x + dx2 + 1) * 3 + 2] - p2[(x - 1) * 3 + 2]))) / 2;

			}
		}
	}

	double t12 = getTickCount();
	cout << (t12 - t11) / (getTickFrequency()) << endl;
	imwrite("costV.bmp", costV);


	//寻找最佳缝合线
	vector<Point> seam;
	//第一条缝，起始点为Point(142，1098)
	Point p1(interSectBr_ / 2, 0);
	seam.push_back(p1);
	while (p1.y < interSectHe_ - 1)
	{
		float* p = costV.ptr<float>(p1.y + 1);
		float a = p[p1.x - 1];
		float b = p[p1.x];
		float c = p[p1.x + 1];

		if (a == b && a == c)
		{
			p1 = Point(p1.x, p1.y + 1);
			seam.push_back(p1);
			continue;

		}
		if (a <= b && a <= c)
		{
			p1 = Point(p1.x - 1, p1.y + 1);
			seam.push_back(p1);
			continue;

		}
		if (b <= a && b <= c)
		{
			p1 = Point(p1.x, p1.y + 1);
			seam.push_back(p1);
			continue;

		}
		if (c <= a && c <= b)
		{
			p1 = Point(p1.x + 1, p1.y + 1);
			seam.push_back(p1);
			continue;
		}
	}
	//for (int i = 0; i < seam.size(); i++)
		//cout << seam[i] << endl;

	Mat mask0, mask1;
	Mat a, b;
	cvtColor(images1, a, CV_RGB2GRAY);
	cvtColor(images2, b, CV_RGB2GRAY);
	mask0.create(a.rows, a.cols, CV_32FC1);
	mask1.create(b.rows, b.cols, CV_32FC1);
	mask0.setTo(1);
	mask1.setTo(1);
	//创建mask0和mask1，以粗略找出重叠部分
	//创建mask0的重叠部分
	cout << "bbbbbbbbbbb" << endl;
	double t13 = getTickCount();
	//对图像1重叠部分赋予权重
	Mat aaa;
	mask0.copyTo(aaa);
	//int count = 2;
	//int left = 0, right = 0;
	//设置两张重叠区域的掩码
	Mat mask_r1, mask_r2;
	mask_r1.create(height, width + 2, CV_32FC1);
	mask_r2.create(height, width + 2, CV_32FC1);
	if (dy > 0)
	{
		//设置边界区域
		for (int y = 0; y < height; ++y)
		{
			float* p1 = mask_r1.ptr<float>(y);
			float* p2 = mask_r2.ptr<float>(y);

			p1[0] = 128;
			p2[0] = 128;
			p1[width + 1] = 128;
			p2[width + 1] = 128;
		}
		//找重叠部分
		for (int y = 0; y < height; ++y)
		{
			float* p1 = a.ptr<float>(y + dy2);
			float* p2 = b.ptr<float>(y);
			float* p3 = mask_r1.ptr<float>(y);
			float* p4 = mask_r2.ptr<float>(y);

			for (int x = 1; x < width + 1; ++x)
			{
				if (p1[x + dx2 - 1] >= 20 && p2[x - 1] >= 20)
				{
					p3[x] = 255;
					p4[x] = 255;
				}
				if (p1[x + dx2 - 1] >= 20 && p2[x - 1] < 20)
				{
					p3[x] = 1;
					p4[x] = 0;
				}
				if (p1[x + dx2 - 1] < 20 && p2[x - 1] >= 20)
				{
					p3[x] = 0;
					p4[x] = 1;
				}
				if (p1[x + dx2 - 1] < 20 && p2[x - 1] < 20)
				{
					p3[x] = 1;
					p4[x] = 1;
				}
			}
		}

	}
	if (dy < 0)
	{
		//设置边界区域
		for (int y = 0; y < height; ++y)
		{
			float* p1 = mask_r1.ptr<float>(y);
			float* p2 = mask_r2.ptr<float>(y);

			p1[0] = 128;
			p2[0] = 128;
			p1[width + 1] = 128;
			p2[width + 1] = 128;
		}
		//找重叠部分
		for (int y = 0; y < height; ++y)
		{
			float* p1 = a.ptr<float>(y);
			float* p2 = b.ptr<float>(y + dy1);
			float* p3 = mask_r1.ptr<float>(y);
			float* p4 = mask_r2.ptr<float>(y);

			for (int x = 1; x < width + 1; ++x)
			{
				if (p1[x + dx2 - 1] >= 20 && p2[x - 1] >= 20)
				{
					p3[x] = 255;
					p4[x] = 255;
				}
				if (p1[x + dx2 - 1] >= 20 && p2[x - 1] < 20)
				{
					p3[x] = 1;
					p4[x] = 0;
				}
				if (p1[x + dx2 - 1] < 20 && p2[x - 1] >= 20)
				{
					p3[x] = 0;
					p4[x] = 1;
				}
				if (p1[x + dx2 - 1] < 20 && p2[x - 1] < 20)
				{
					p3[x] = 1;
					p4[x] = 1;
				}
			}
		}
	}
	if (dy == 0)
	{
		//设置边界区域
		for (int y = 0; y < height; ++y)
		{
			float* p1 = mask_r1.ptr<float>(y);
			float* p2 = mask_r2.ptr<float>(y);

			p1[0] = 128;
			p2[0] = 128;
			p1[width + 1] = 128;
			p2[width + 1] = 128;
		}
		//找重叠部分
		for (int y = 0; y < height; ++y)
		{
			float* p1 = a.ptr<float>(y);
			float* p2 = b.ptr<float>(y);
			float* p3 = mask_r1.ptr<float>(y);
			float* p4 = mask_r2.ptr<float>(y);

			for (int x = 1; x < width + 1; ++x)
			{
				if (p1[x + dx2 - 1] >= 10 && p2[x - 1] >= 10)
				{
					p3[x] = 255;
					p4[x] = 255;
				}
				if (p1[x + dx2 - 1] >= 10 && p2[x - 1] < 10)
				{
					p3[x] = 1;
					p4[x] = 0;
				}
				if (p1[x + dx2 - 1] < 10 && p2[x - 1] >= 10)
				{
					p3[x] = 0;
					p4[x] = 1;
				}
				if (p1[x + dx2 - 1] < 10 && p2[x - 1] < 10)
				{
					p3[x] = 1;
					p4[x] = 1;
				}
			}
		}
	}
	Mat c, d;
	mask_r1.copyTo(c);
	mask_r2.copyTo(d);

	//动态分配权重
	//对mask_r2进行一次遍历，两次分配
	int count = 3;
	int left = 0;
	int right = 0;
	int count1 = 0;
	//if (dy > 0)
	//{
	for (int y = 0; y < height; ++y)
	{
		count = 3;
		left = 0;
		right = 0;
		while (count > 0)
		{
			float* p1 = mask_r1.ptr<float>(y);
			float* p2 = mask_r2.ptr<float>(y);
			if (count == 3)
			{
				for (int x = 1; x < width + 1; ++x)
				{
					if (p2[x] == 255 && p2[x - 1] == 0 && p2[x + 1] == 1)
					{
						left = x;
						//continue;
					}
					if ((p2[x] == 255 && p2[x - 1] == 0 && p2[x + 1] == 255) || (p2[x - 1] == 128 && p2[x] == 255 && p2[x + 1] == 255 && p2[x + 2] == 255 && p2[x + 3] == 255))
					{
						left = x;
						//continue;
					}
				}
				count--;
			}
			if (count == 2)
			{
				for (int x = 1; x < width + 1; ++x)
				{
					if (p2[x - 1] == 0 && p2[x] == 255 && p2[x + 1] == 1)
					{
						right = x;
						//continue;
					}
					if (p2[x - 1] == 255 && p2[x] == 255 && (p2[x + 1] == 1 || p2[x + 1] == 128))
					{
						right = x;
						//continue;
					}
				}
				count--;
			}

			//两次分配权重
			//根据最佳缝合线分配权重
			if (count == 1)
			{
				for (int x = 1; x < width + 1; ++x)
				{
					if (p2[x] == 255)
					{
						if (left && left == right)
						{
							p1[x] = 1;
							p2[x] = 0;
						}
						else if (x <= (seam[y + dy2 + dy1].x + 1))
						{
							p1[x] = 1 - 0.5 * (x - left) / (seam[y + dy2 + dy1].x + 1 - left);
							p2[x] = 1 - p1[x];
						}
						else if (x > (seam[y + dy2 + dy1].x + 1) && x <= right)
						{
							//cout << "a" << endl;
							p1[x] = 0.5 * (right - x) / (right - seam[y + dy2 + dy1].x - 1);
							p2[x] = 1 - p1[x];
						}
					}
				}
				count--;
			}
		}
		count1++;
		//cout << count1 <<": "<<left << ' ' << right << endl;
	}
	//再对一些特殊部分进行处理
	for (int y = 0; y < height; ++y)
	{
		float*p1 = mask_r1.ptr<float>(y);
		float*p2 = mask_r2.ptr<float>(y);
		for (int x = 0; x < width + 1; ++x)
		{
			if (p1[x] == 255)
			{
				p1[x] = 1;
				p2[x] = 0;
			}
		}
	}
	//}
	
	double t14 = getTickCount();
	cout << (t14 - t13) / (getTickFrequency()) << endl;
	cout << "ddddddddddd" << endl;

	//填充数据
	Mat pano;
	pano.create(panoHe_, panoBr_, CV_32FC3);
	pano.setTo(0);
	if (dy > 0)
	{
		//喂图像1的数据
		for (int y = 0; y < images1.rows; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = pano.ptr<float>(y);
			for (int x = 0; x < dx2; ++x)
			{
				p2[3 * x] = p1[3 * x];
				p2[3 * x + 1] = p1[3 * x + 1];
				p2[3 * x + 2] = p1[3 * x + 2];
			}
		}
		//喂图像2的数据
		for (int y = dy2; y < images2.rows; ++y)
		{
			float* p1 = images2.ptr<float>(y - dy2);
			float* p2 = pano.ptr<float>(y);
			for (int x = images1.cols; x < pano.cols; ++x)
			{
				p2[3 * x] = p1[3 * (x - dx2)];
				p2[3 * x + 1] = p1[3 * (x - dx2) + 1];
				p2[3 * x + 2] = p1[3 * (x - dx2) + 2];
			}
		}
		//喂重叠部分数据
		for (int y = 0; y < height; ++y)
		{
			float* p1 = images1.ptr<float>(y + dy2);
			float* p2 = images2.ptr<float>(y);
			float* m1 = mask_r1.ptr<float>(y);
			float* m2 = mask_r2.ptr<float>(y);
			float* p3 = pano.ptr<float>(y + dy2);
			for (int x = dx2; x < dx2 + width; ++x)
			{
				p3[3 * x] = p1[3 * x] * m1[x - dx2 + 1] + p2[3 * (x - dx2)] * m2[(x - dx2 + 1)];
				p3[3 * x + 1] = p1[3 * x + 1] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 1] * m2[(x - dx2 + 1)];
				p3[3 * x + 2] = p1[3 * x + 2] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 2] * m2[(x - dx2 + 1)];

			}
		}
	}
	if (dy < 0)
	{
		//喂图像1的数据
		for (int y = 0; y < images1.rows; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = pano.ptr<float>(y + dy1);
			for (int x = 0; x < dx2; ++x)
			{
				p2[3 * x] = p1[3 * x];
				p2[3 * x + 1] = p1[3 * x + 1];
				p2[3 * x + 2] = p1[3 * x + 2];
			}
		}
		//喂图像2的数据
		for (int y = 0; y < images2.rows; ++y)
		{
			float* p1 = images2.ptr<float>(y);
			float* p2 = pano.ptr<float>(y);
			for (int x = images1.cols; x < pano.cols; ++x)
			{
				p2[3 * x] = p1[3 * (x - dx2)];
				p2[3 * x + 1] = p1[3 * (x - dx2) + 1];
				p2[3 * x + 2] = p1[3 * (x - dx2) + 2];
			}
		}
		//喂重叠部分数据
		for (int y = 0; y < height; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y + dy1);
			float* m1 = mask_r1.ptr<float>(y);
			float* m2 = mask_r2.ptr<float>(y);
			float* p3 = pano.ptr<float>(y + dy1);
			for (int x = dx2; x < dx2 + width; ++x)
			{
				p3[3 * x] = p1[3 * x] * m1[x - dx2 + 1] + p2[3 * (x - dx2)] * m2[x - dx2 + 1];
				p3[3 * x + 1] = p1[3 * x + 1] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 1] * m2[x - dx2 + 1];
				p3[3 * x + 2] = p1[3 * x + 2] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 2] * m2[x - dx2 + 1];

			}
		}
	}
	if (dy == 0)
	{
		//喂图像1的数据
		for (int y = 0; y < images1.rows; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = pano.ptr<float>(y + dy1);
			for (int x = 0; x < dx2; ++x)
			{
				p2[3 * x] = p1[3 * x];
				p2[3 * x + 1] = p1[3 * x + 1];
				p2[3 * x + 2] = p1[3 * x + 2];
			}
		}
		//喂图像2的数据
		for (int y = 0; y < images2.rows; ++y)
		{
			float* p1 = images2.ptr<float>(y);
			float* p2 = pano.ptr<float>(y);
			for (int x = images1.cols; x < pano.cols; ++x)
			{
				p2[3 * x] = p1[3 * (x - dx2)];
				p2[3 * x + 1] = p1[3 * (x - dx2) + 1];
				p2[3 * x + 2] = p1[3 * (x - dx2) + 2];
			}
		}
		//喂重叠部分数据
		for (int y = 0; y < height; ++y)
		{
			float* p1 = images1.ptr<float>(y);
			float* p2 = images2.ptr<float>(y);
			float* m1 = mask_r1.ptr<float>(y);
			float* m2 = mask_r2.ptr<float>(y);
			float* p3 = pano.ptr<float>(y);
			for (int x = dx2; x < dx2 + width; ++x)
			{
				p3[3 * x] = p1[3 * x] * m1[x - dx2 + 1] + p2[3 * (x - dx2)] * m2[x - dx2 + 1];
				p3[3 * x + 1] = p1[3 * x + 1] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 1] * m2[x - dx2 + 1];
				p3[3 * x + 2] = p1[3 * x + 2] * m1[x - dx2 + 1] + p2[3 * (x - dx2) + 2] * m2[x - dx2 + 1];

			}
		}
	}
	int aa;
	aa = 0;
	cout << "endl" << endl;
	double t8 = getTickCount();
	cout << "融合所需时间: " << (t8 - t7) / getTickFrequency() << endl;
	imwrite("pano.bmp", pano);
	system("pause");
	return 0;
}

