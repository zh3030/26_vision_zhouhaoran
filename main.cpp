/**
 * @file main.cpp
 * @brief 主程序入口，负责视频读取、处理循环和结果展示
 * @author Zhou Haoran
 * @date 2026-05-20
 */
#include <opencv2/opencv.hpp>
#include "buff_detector.h"
#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#define VIDEO_TEST

int main(int argc, char** argv) {

	std::string path = "../test1.mp4";   // 默认路径
    if (argc >= 2) {
		path = argv[1];                  // 使用命令行参数
    } else {
		std::cout << "Usage: " << argv[0] << " <path>" << std::endl;
        std::cout << "No path provided, using default: " << path << std::endl;
    }
#ifdef VIDEO_TEST
    cv::VideoCapture cap(path);         
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << path << std::endl;
        return -1;
    }
#endif // VIDEO_TEST

    BuffParams params;
    if(loadBuffParams("../config/buff_params.yaml", params))
    {
		std::cout << "BuffParams loaded successfully in main.cpp\n";
    }
    else
    {
		std::cout << "Failed to load BuffParams in main.cpp, using defaults\n";
    }
	buff b(params);
	std::vector<cv::Point3f> points = params.world_points; // 世界坐标系下的固定参考点（中心R和各扇叶位置)
#ifdef VIDEO_TEST
	cv::Mat frame, initial_frame, result_frame;
	std::vector<cv::Point> center;
	cv::Rect roi; int jump = 5; // 前几帧使用全图进行初始化，获取初始ROI，后续帧使用ROI进行处理，减少计算量
#ifdef BUFF_WRITE
	std::cout << "Initializing video writer..." << std::endl;
	int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
	int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
	double fps = cap.get(cv::CAP_PROP_FPS);
	if (fps <= 0.0) fps = 30.0;   // 防止读取失败

	cv::VideoWriter writer("../output.avi",                 
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		fps,
		cv::Size(frame_width, frame_height));
	if (!writer.isOpened()) 
    {
		std::cerr << "Error: Could not open video writer." << std::endl;
		return -1;
	}
#endif // BUFF_WRITE
#ifdef BUFF_TIME
	cv::TickMeter tm;
#endif // BUFF_TIME
	while (cap.isOpened())
	{
		#ifdef BUFF_TIME
		tm.reset(); tm.start();
		#endif // BUFF_TIME
		cap >> initial_frame;
		if (initial_frame.empty()) break;
		
		
		if (roi.empty() || jump > 0)
		{
			roi = cv::Rect(0, 0, initial_frame.cols, initial_frame.rows);
			frame = initial_frame(roi);
			b.run(frame);
			auto result_roi = b.get_roi_info();
			roi = cv::Rect(result_roi.cropX, result_roi.cropY, result_roi.cropradius * 2, result_roi.cropradius * 2);
			--jump;
		}
		else
		{
			frame = initial_frame(roi);
			b.run(frame);
		}
		
		//cv::Point center = b.get_center();
		//const std::deque<AngleData>& angle_deque = b.get_angle_info();
		//if (angle_deque.size() > 1)
		//{
			//	double angle_diff = angle_deque[1].angle_d - angle_deque[0].angle_d;
			//	if (angle_diff > 180) angle_diff -= 360;
			//	else if (angle_diff < -180) angle_diff += 360;
			//	double time_diff = angle_deque[1].current_time_d - angle_deque[0].current_time_d;
			//	if (time_diff < 1e-3) time_diff = 1000 / 60;
			//	std::cout << angle_diff / time_diff << std::endl;
			//	std::cout << angle_deque[0].distance_d << std::endl;
			//}
			//double predicted = predictAngle(angle_deque);
			//cv::Point predicted_point = center + cv::Point(static_cast<int>(10 * std::cos(predicted)), static_cast<int>(100 * std::sin(predicted)));
			//cv::circle(frame, predicted_point, 50, cv::Scalar(0, 0, 255), -1); 
			// 此处原计划用于测试角速度，后续计划封装，但未完成。
			#ifdef BUFF_WRITE
			writer.write(initial_frame);
			#endif // BUFF_WRITE



			cv::imshow("frame", frame);
			if (cv::waitKey(10) == 27) break; // 按下 ESC 键退出
			#ifdef BUFF_TIME
			tm.stop();
			std::cout << "Frame processing time: " << tm.getTimeMilli() << " ms" << std::endl;
			#endif
	}
	#ifdef BUFF_WRITE
	writer.release();
	#endif // BUFF_WRITE
	cap.release();
#endif // VIDEO_TEST

#ifndef VIDEO_TEST
	cv::Mat image = cv::imread(path);
	b.run(image);
	std::vector<cv::Point> center = b.get_points();

	cv::Mat rvec, tvec;
	if(SolvePNPWithCenter(params, center, points, rvec, tvec)) getReprojectError(image, params, center, points, rvec, tvec);
	std::cout << "Rotation Vector:\n" << rvec << std::endl;
	std::cout << "Translation Vector:\n" << tvec << std::endl;
	
	cv::Point2f world_point;
	for(const auto& point : center)
	{
		imageToWorldZ0(point, rvec, tvec, params.camera_matrix, world_point);
		std::cout << world_point << std::endl;
	}
	std::cout << "World coordinates of the first point (z=0 plane): (" << world_point.x << ", " << world_point.y << ")\n";	

	cv::imshow("result", image);
	cv::waitKey(0);
#endif // VIDEO_TEST

	return 0;
}