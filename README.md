# ImageStitch
写这个库主要是因为自己在做图像拼接项目时，深感在使用opencv实现图像拼接时资料欠缺，很多资料不够全面或者不能够很好
地针对opencv。
该库主要包括下面几个部分源代码的重构以及解析： 
第一部分代码：特征点检测 
第二部分代码：特征点匹配和恢复内外参数 
第三部分代码：图像融合

一些参考文档：

下面是我的一些经验： 
1.安装opencv库 opencv有两种安装方式：一是直接从官网上安装，这样做不能使用SIFT等一些受到保护的算法；二是从github
上下载opencv和对应的contrib版本，使用CMake编译。在使用CMake编译时，注意：
1）opencv版本和opencv_contrib版本要对应; 
2)使用CMake编译时需要勾选BUILD_opencv_world, OPENCV_ENABLE_NOFREE。如果不勾选BUILD_opencv_world会生成一大堆库.lib文件，勾选BUILD_opencv_world之后，只会生成opencv_world342.lib和opencv_world342d.lib两个lib文件，这在后续配置很方便。勾选OPENCV_ENABLE_NOFREE能够让你使用SIFT等有专利的算法； 
3)附加依赖项只要包含opencv_world342d.lib就行了，不要连opencv_world342.lib一起加上去。

#由于很多东西需要图片，所以放到知乎上说 
2.特征点检测 链接： 
3.特征匹配和恢复相机内外参 链接： 
4.图像融合 链接：