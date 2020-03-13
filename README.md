# ImageStitch
写这个库主要是因为自己在做图像拼接项目时，深感在使用opencv实现图像拼接时资料欠缺，很多资料不够全面或者不能够很好地针对opencv。<br>
该库主要包括下面几个部分源代码的重构：<br> 
第一部分代码：特征点检测<br> 
第二部分代码：特征点匹配<br> 
第三部分代码：计算单应性矩阵<br> 
第四部分代码：恢复相机内参数<br> 
第五部分代码：圆柱面投影变换<br> 
第六部分代码：动态规划法寻找最佳缝合线<br> 
第七部分代码：图像融合<br> 
<br> 
<br> 
一些参考链接：<br> 
https://me.csdn.net/zhaocj<br> 
《图像局部特征检测和描述  基于OpenCV源码分析的算法与实现》赵春江_北京：人民邮电出版社_2018.07_286_14414755<br> 

图像拼接项目值得看的论文：<br> 

一些Opencv库配置经验：<br>  
1.安装opencv库 opencv有两种安装方式：一是直接从官网上安装，这样做不能使用SIFT等一些受到保护的算法；二是从github<br> 
上下载opencv和对应的contrib版本，使用CMake编译。在使用CMake编译时，注意：<br> 
1）opencv版本和opencv_contrib版本要对应; <br> 
2)使用CMake编译时需要勾选BUILD_opencv_world, OPENCV_ENABLE_NOFREE。如果不勾选BUILD_opencv_world会生成一大堆库.lib文件，勾选BUILD_opencv_world之后，只会生成opencv_world342.lib和opencv_world342d.lib两个lib文件，这在后续配置很方便。勾选OPENCV_ENABLE_NOFREE能够让你使用SIFT等有专利的算法； 
3)附加依赖项只要包含opencv_world342d.lib就行了，不要连opencv_world342.lib一起加上去。<br> 

另有一些说明见知乎：https://zhuanlan.zhihu.com/p/112800029<br> 
