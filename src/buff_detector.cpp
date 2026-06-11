#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>          // cv::findContours, cv::dilate, cv::GaussianBlur 等
#include <opencv2/core/ocl.hpp>         // cv::ocl::useOpenCL, cv::UMat

#include "buff_detector.h"

// ---------- 构造函数 ----------
buff::buff(const BuffParams& params)
    : m_params(params),
    div_data(m_params.sector_count + 1, std::vector<BuffData>()),
#ifdef ADD_POINTS
    m_vec_center(m_params.sector_count + 1, cv::Point(-1, -1)),
#endif
    roi_info({ 0, 0, 0 }),
    angle_deque()
{
    std::cout << "buff build!" << std::endl;
}

// ---------- 主处理流程 ----------
void buff::run(cv::Mat& input) {
    cv::Mat sub_image, transform;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<BuffData> vec_data;
    cv::Point center_r(0, 0);
    double current_angle = 0.0;
    std::vector<cv::Point> temp_centers(m_params.sector_count + 1, cv::Point(-1, -1));
    std::vector<std::vector<BuffData>> temp_div(m_params.sector_count + 1);

    // ---------- 主程序 ----------

#ifdef BUFF_TIME
    cv::TickMeter tm;
    tm.reset(); tm.start();
#endif

    // 1. 图像预处理
    subImage(input, sub_image);

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "subImage: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif

    // 2. 查找轮廓
    findContours(sub_image, contours, hierarchy);

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "findContours: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif

    // 3. 初始化轮廓数据（过滤无效轮廓）
    initializeContours(contours, hierarchy, vec_data);

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "initializeContours: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif

    // 4. 提取圆心 R（必须成功，否则跳过本帧）
    if (!extractCenterR(vec_data, temp_div, center_r)) {
#ifdef BUFF_TEST
        std::cout << "Center R not found, skip frame." << std::endl;
#endif
        return; // 根据需求可改为保留上一帧结果，此处选择丢弃
    }
    temp_centers[0] = center_r;

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "extractCenterR: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif

    // 5. 提取待激活圆
    extractInactiveCircleCenter(vec_data, center_r, temp_div, current_angle, temp_centers);

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "extractInactiveCircleCenter: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif

    // 6. 提取已激活圆
    extractActivatedCircleCenter(vec_data, center_r, current_angle, temp_div, temp_centers);

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "extractActivatedCircleCenter: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif

    // 7. 将剩余轮廓分配至对应扇区（扇叶）
    processContours(vec_data, center_r, current_angle, temp_div);

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "processContours: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif

    // 8. 提取扇叶角点
    findCorners(input, center_r, temp_div);

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "findCorners: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif

    // 9. 更新成员状态

    // 根据待激活圆与 R 的距离动态调整 ROI 大小，更新角度历史队列，本用于预测，但未完成
    
    double k = 2.5;
    m_center = center_r;
    div_data = std::move(temp_div);

    double distance = std::sqrt(geometry_utils::square_distance(m_center, div_data[1][0].center_d));
    roi_info.cropradius = k * distance;
    if (roi_info.cropradius > std::min(input.cols, input.rows) / 2.5) {
        k -= 0.5;
        roi_info.cropradius = std::min(input.cols, input.rows) / 2.5;
    }
    if (roi_info.cropradius < std::min(input.cols, input.rows) / 5) {
        k += 0.5;
        roi_info.cropradius = std::min(input.cols, input.rows) / 5;
    }
    roi_info.cropX = std::max(0, m_center.x - static_cast<int>(roi_info.cropradius));
    roi_info.cropY = std::max(0, m_center.y - static_cast<int>(roi_info.cropradius));
    current_angle_m = current_angle;
    angle_data.angle_d = current_angle;
    angle_data.current_BUFF_TIME_d = cv::getTickCount() / cv::getTickFrequency();
    angle_data.distance_d = distance;
    if (angle_deque.empty()) angle_deque.push_back(angle_data);
    if ((std::abs(angle_data.angle_d - angle_deque.back().angle_d)) > 1e-2 &&
        (std::abs(angle_data.distance_d - angle_deque.back().distance_d)) > 1e-2
        ) angle_deque.push_back(angle_data);
    if (angle_deque.size() >= m_params.max_vector_capacity) angle_deque.pop_front();

#ifdef ADD_POINTS
    m_vec_center = std::move(temp_centers);
#endif

#ifdef BUFF_TEST
    drawDebugInfo(input);
#endif

#ifdef BUFF_TIME
    tm.stop();
    std::cout << "update state: " << tm.getTimeMilli() << " ms\n";
    tm.reset(); tm.start();
#endif
}

// ---------- 对外接口 ----------
#ifdef ADD_POINTS
std::vector<cv::Point> buff::get_points() const { return m_vec_center; }
#endif

const std::vector<std::vector<BuffData>>& buff::get_div_data() const { return div_data; }
const cv::Point& buff::get_center() const { return m_center; }
const std::deque<AngleData>& buff::get_angle_info() const { return angle_deque; }
const RoiData& buff::get_roi_info() const { return roi_info; }

// ---------- 图像预处理 ----------
void buff::subImage(const cv::Mat& input, cv::Mat& output) {
    static cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, m_params.morph_kernel_size);

    // 尝试启用 OpenCL（如果环境支持）
    bool useOpenCL = cv::ocl::useOpenCL();  // 检测当前OpenCL是否可用
    if (useOpenCL) {
        try {
            cv::UMat u_input, u_output;
            input.copyTo(u_input);   // 将数据上传到 UMat

            std::vector<cv::UMat> u_channels(3);
            cv::split(u_input, u_channels);
            cv::UMat u_b = u_channels[0];
            cv::UMat u_g = u_channels[1];
            cv::UMat u_r = u_channels[2];

            cv::subtract(u_r, u_b, u_output);
            cv::threshold(u_output, u_output, m_params.threshold_value, 255, cv::THRESH_BINARY);
            cv::GaussianBlur(u_output, u_output, cv::Size(3, 3), 0);
            cv::dilate(u_output, u_output, kernel);   // kernel 为 Mat 会被自动转换

            u_output.copyTo(output);                  // 下载结果
            return;                                   // GPU 处理成功，直接返回
        }
        catch (const cv::Exception& e) {
            // 如果 GPU 路径出错（如显存不足），回退到 CPU
            std::cerr << "[subImage] GPU acceleration failed, falling back to CPU: "
                << e.what() << std::endl;
        }
    }

    // ---------- CPU 回退路径 ----------
    cv::Mat temp;
    std::vector<cv::Mat> channels(3);
    cv::split(input, channels);
    cv::Mat b = channels[0], g = channels[1], r = channels[2];
    cv::subtract(r, b, temp);
    cv::threshold(temp, temp, m_params.threshold_value, 255, cv::THRESH_BINARY);
    cv::GaussianBlur(temp, temp, cv::Size(3, 3), 0);
    cv::dilate(temp, temp, kernel);
    std::cout << "GPU not used, processed on CPU." << std::endl;
    output = temp;
}

void buff::findContours(const cv::Mat& image,
    std::vector<std::vector<cv::Point>>& contours,
    std::vector<cv::Vec4i>& hierarchy) {
    cv::findContours(image, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
}

// ---------- 轮廓初始化 ----------
void buff::initializeContours(std::vector<std::vector<cv::Point>>& contours,
    std::vector<cv::Vec4i>& hierarchy,
    std::vector<BuffData>& vec_data) {
    vec_data.clear();
    vec_data.reserve(contours.size());
    for (size_t i = 0; i < contours.size(); ++i) {
        BuffData data;

        data.area_d = cv::contourArea(contours[i]);
        if (data.area_d < m_params.min_contour_area) continue;

        getHierarchyInfo(hierarchy, static_cast<int>(i), data.father_index_d, data.child_sum_d);
        data.contour_d = std::move(contours[i]);
        data.bounding_rect_d = cv::boundingRect(data.contour_d);
        data.length_d = cv::arcLength(data.contour_d, true);
        data.circularity_d = 4 * CV_PI * data.area_d / (data.length_d * data.length_d);
        data.length_width_ratio_d = static_cast<double>(data.bounding_rect_d.width) /
            data.bounding_rect_d.height;

        // 计算轮廓中心
        cv::Moments m = cv::moments(data.contour_d);
        if (m.m00 != 0) {
            data.center_d = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
        }

        vec_data.push_back(std::move(data));
    }
}

// ---------- 圆心 R 提取 ----------
bool buff::extractCenterR(std::vector<BuffData>& vec_data,
    std::vector<std::vector<BuffData>>& div_data,
    cv::Point& center_out) {
    double min_area = std::numeric_limits<double>::max();
    int best_idx = -1;
    for (size_t i = 0; i < vec_data.size(); ++i) {
        const auto& d = vec_data[i];

        if (d.length_width_ratio_d < m_params.length_width_ratio_min ||
            d.length_width_ratio_d > m_params.length_width_ratio_max) continue;
        if (d.father_index_d != -1) continue; // 需为最外层轮廓
        if (d.area_d > m_params.min_contour_area && d.area_d < min_area) {
            min_area = d.area_d;
            best_idx = static_cast<int>(i);
        }
    }
    if (best_idx == -1) return false;

    vec_data[best_idx].state_index_d = RmEnergyState::CenterR;
    vec_data[best_idx].position_index_d = 0;
    center_out = vec_data[best_idx].center_d;

    div_data[0].push_back(std::move(vec_data[best_idx]));
    vec_data.erase(vec_data.begin() + best_idx);
    return true;
}

// ---------- 待激活圆提取 ----------
bool buff::extractInactiveCircleCenter(std::vector<BuffData>& vec_data,
    const cv::Point& center_r,
    std::vector<std::vector<BuffData>>& div_data,
    double& current_angle,
    std::vector<cv::Point>& centers_out) {
    for (size_t i = 0; i < vec_data.size(); ++i) {
        if (vec_data[i].child_sum_d <= 1) continue;
        if (vec_data[i].circularity_d < m_params.min_circularity) continue;

        vec_data[i].state_index_d = RmEnergyState::InactiveCircle;
        vec_data[i].angle_d = geometry_utils::angle_from_center(center_r, vec_data[i].center_d);
        current_angle = vec_data[i].angle_d; // 作为参考零位
        vec_data[i].position_index_d = 1;
        centers_out[1] = vec_data[i].center_d;
        div_data[1].push_back(std::move(vec_data[i]));
        vec_data.erase(vec_data.begin() + i);
        return true;
    }
    return false;
}

// ---------- 已激活圆提取 ----------
void buff::extractActivatedCircleCenter(std::vector<BuffData>& vec_data,
    const cv::Point& center_r,
    const double current_angle,
    std::vector<std::vector<BuffData>>& div_data,
    std::vector<cv::Point>& centers_out) {
    // 反向遍历避免删除时下标错乱（size_t类型在0处--会变成负数导致死循环）
    for (int i = static_cast<int>(vec_data.size()) - 1; i >= 0; --i) {
        if (vec_data[i].circularity_d < m_params.min_circularity) continue;
        if (vec_data[i].area_d < m_params.min_contour_area) continue;
        if (vec_data[i].father_index_d != -1) continue;
        if (vec_data[i].child_sum_d > 1) continue; // 待激活圆已单独处理

        vec_data[i].state_index_d = RmEnergyState::ActivatedCircle;
        vec_data[i].angle_d = geometry_utils::angle_from_center(center_r, vec_data[i].center_d);
        vec_data[i].position_index_d = get_index(vec_data[i].angle_d, current_angle);
        if (vec_data[i].position_index_d < 0 || vec_data[i].position_index_d > m_params.sector_count)
            continue; // 无效索引

        centers_out[vec_data[i].position_index_d] = vec_data[i].center_d;
        div_data[vec_data[i].position_index_d].push_back(std::move(vec_data[i]));
        vec_data.erase(vec_data.begin() + i);
    }
}

// ---------- 轮廓分组（扇叶分配） ----------
void buff::processContours(std::vector<BuffData>& vec_data,
    const cv::Point& center_r,
    double current_angle,
    std::vector<std::vector<BuffData>>& div_data) {
    // 将剩余轮廓（扇叶）分配到最近的已存在扇区
    while (!vec_data.empty()) {
        // 计算当前轮廓角度
        vec_data[0].angle_d = geometry_utils::angle_from_center(center_r, vec_data[0].center_d);
        double best_dist = std::numeric_limits<double>::max();
        int best_sector = -1;
        // 在非空扇区中寻找角度最接近的
        for (size_t sec = 1; sec < div_data.size(); ++sec) {
            if (div_data[sec].empty()) continue;
            double dist = std::abs(div_data[sec][0].angle_d - vec_data[0].angle_d);
            // 处理角度跨越0度的情况
            dist = std::min(dist, 360.0 - dist);
            if (dist < best_dist) {
                best_dist = dist;
                best_sector = static_cast<int>(sec);
            }
        }
        if (best_sector != -1) {
            // 检查是否与已有轮廓距离过近（重复，主要是已激活圆的内外层都会检出，故需要舍弃）
            if (geometry_utils::square_distance(div_data[best_sector][0].center_d,
                vec_data[0].center_d) < m_params.duplicate_distance_threshold) {
                vec_data.erase(vec_data.begin());
                continue;
            }
            vec_data[0].position_index_d = best_sector;
            div_data[best_sector].push_back(std::move(vec_data[0]));
        }
        // 未找到任何匹配扇区则丢弃
        vec_data.erase(vec_data.begin());
    }
}

// ---------- 角点提取 ----------
void buff::findCorners(cv::Mat& image, const cv::Point& center_r,
    std::vector<std::vector<BuffData>>& div_data) {
#ifdef BUFF_TIME
    cv::TickMeter tm;
#endif
    for (size_t sec = 1; sec < div_data.size(); ++sec) {
        if (div_data[sec].size() <= 1) continue; // 无扇叶
        // 第一个元素应为该扇区的圆（激活圆或待激活圆）
        bool has_activated_circle = (div_data[sec][0].state_index_d == RmEnergyState::ActivatedCircle);
        if (div_data[sec].size() == 2) {
            extractUnoccludedBladeCorners(div_data[sec][1], image);
        }
        else if (div_data[sec].size() > 2 && has_activated_circle) {
            extractPartiallyOccludedBladeCorners(div_data[sec], image);
        }
        else if (div_data[sec].size() > 2 && !has_activated_circle && div_data[sec][0].state_index_d == RmEnergyState::InactiveCircle) // 提取待激活圆的指示扇叶并标注角点
        {
            extractInactiveBladeCornersByInnerCenters(div_data[sec], image);
        }
    }
}

void buff::extractUnoccludedBladeCorners(BuffData& blade, cv::Mat& image) {
    std::vector<cv::Point> hull, final_hull;
    cv::convexHull(blade.contour_d, hull, true, false);
    cv::approxPolyDP(hull, final_hull, m_params.approx_poly_epsilon * 2, true);
    blade.corners_d = std::move(final_hull);

#ifdef BUFF_TEST
    for (size_t i = 0; i < blade.corners_d.size(); ++i) {
        cv::putText(image, "UC", blade.corners_d[i] + cv::Point(10, -10), // "UC" = Unoccluded Corner
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::circle(image, blade.corners_d[i], 5, cv::Scalar(255, 255, 255), -1);
        cv::line(image, blade.corners_d[i], blade.corners_d[(i + 1) % blade.corners_d.size()], cv::Scalar(0, 255, 0), 2);
    }
#endif

}

void buff::extractPartiallyOccludedBladeCorners(std::vector<BuffData>& sector_data,
    cv::Mat& image) {
    cv::Point center_r = sector_data[0].center_d;

    // 合并除第一个（圆）外的所有轮廓点
    std::vector<cv::Point> merged = mergeContoursExceptFirst(sector_data);
    if (merged.empty()) return;

	for (size_t i = 1; i < sector_data.size(); ++i) {
		sector_data[i].state_index_d = RmEnergyState::ActivatedBladePartiallyOccluded;
	}

    cv::Moments m = cv::moments(merged);
    cv::Point blade_center;
    if (m.m00 != 0) {
        blade_center = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
    }
    else {
        blade_center = merged[0]; // fallback
    }
    std::vector<cv::Point> hull, final_hull, final_hull2;
    cv::convexHull(merged, hull);
    cv::approxPolyDP(hull, final_hull, m_params.approx_poly_epsilon * 3, true);
    cv::approxPolyDP(final_hull, final_hull2, m_params.approx_poly_epsilon * 3, true); // 二次近似进一步简化)

#ifdef BUFF_TEST
    for (size_t i = 0; i < final_hull2.size(); ++i) {
        cv::putText(image, "POC", final_hull2[i] + cv::Point(10, -10),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        cv::circle(image, final_hull2[i], 5, cv::Scalar(255, 255, 255), -1);
        cv::line(image, final_hull2[i], final_hull2[(i + 1) % final_hull2.size()], cv::Scalar(255, 0, 0), 2); // 蓝色线条连接圆心和角点)
    }
#endif
    sector_data[0].corners_d = std::move(final_hull2);
}

// 待激活圆所在扇区：基于内轮廓中心与最小外接矩形的角点提取
void buff::extractInactiveBladeCornersByInnerCenters(std::vector<BuffData>& sector_data,
    cv::Mat& image) {
    if (sector_data.size() <= 2) return; // 至少需要一个圆+一些轮廓
    cv::Point circle_center = sector_data[0].center_d;
    std::vector<cv::Point> centers;   // 收集符合条件的内轮廓中心

    for (size_t i = 1; i < sector_data.size(); ++i) {
        BuffData& d = sector_data[i];
        if (d.father_index_d != -1) {          // 有父轮廓（内轮廓）
            if (d.child_sum_d > 0) continue;   // 自身还有子轮廓则跳过
            centers.push_back(d.center_d);
            continue;
        }
        // 最外层轮廓，无子轮廓时绘制其最小外接矩形
        if (d.child_sum_d != -1) continue;     // 有子轮廓则跳过
        cv::RotatedRect rect = cv::minAreaRect(d.contour_d);
        std::array<cv::Point2f, 4> vertices;
        rect.points(vertices.data());
#ifdef BUFF_TEST
        for (size_t j = 0; j < 4; ++j) {
            cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 255, 0), 2);
            cv::circle(image, vertices[j], 5, cv::Scalar(255, 255, 0), -1);
        }
#endif
    }

    // 利用收集到的内轮廓中心生成凸包角点
    if (!centers.empty()) {
        std::vector<cv::Point> hull;
        cv::convexHull(centers, hull);
        // 计算偏移角点：相邻凸包点之和减去圆中心
        // 相当于将内轮廓中心作为参考，先得到四条半径与内圆的交点，再用圆心推得所需要的角点
        // 同时将生成的角点存入扇区首个元素（圆）中
        sector_data[0].corners_d.clear();
        for (size_t i = 0; i < hull.size(); ++i) {
            cv::Point corner = hull[i] + hull[(i + 1) % hull.size()] - circle_center;
            sector_data[0].corners_d.push_back(corner);
#ifdef BUFF_TEST
            cv::circle(image, corner, 5, cv::Scalar(255, 255, 0), -1);
#endif
        }
    }
}

std::vector<cv::Point> buff::mergeContoursExceptFirst(const std::vector<BuffData>& data_vec) {
    std::vector<cv::Point> result;
    if (data_vec.size() <= 1) return result;
    if (data_vec.size() == 2) {
        result = data_vec[1].contour_d;
        return result;
    }
    size_t total = 0;
    for (size_t i = 1; i < data_vec.size(); ++i)
        total += data_vec[i].contour_d.size();
    result.reserve(total);
    for (size_t i = 1; i < data_vec.size(); ++i)
        result.insert(result.end(), data_vec[i].contour_d.begin(), data_vec[i].contour_d.end());
    return result;
}

// ---------- 辅助计算 ----------
int buff::get_index(double angle, double reference_angle) const {
    double diff = reference_angle - angle;
    // 标准化到[-180, 180)
    while (diff < -180.0) diff += 360.0;
    while (diff >= 180.0) diff -= 360.0;
    double index_diff = diff / m_params.sector_angle;
    int idx = static_cast<int>(std::round(index_diff));
    // 映射到 1 ~ sector_count
    idx = ((idx % m_params.sector_count) + m_params.sector_count) % m_params.sector_count;
    return idx + 1; // 偏移，0 为 R，1 为待激活圆
}

void buff::getHierarchyInfo(const std::vector<cv::Vec4i>& hierarchy, int index,
    int& hasParent, int& childCount) {
    hasParent = (hierarchy[index][3] != -1) ? hierarchy[index][3] : -1;
    int child = hierarchy[index][2];
    if (child == -1) {
        childCount = -1;
        return;
    }
    int cnt = 0;
    while (child != -1) {
        ++cnt;
        child = hierarchy[child][0];
    }
    childCount = cnt;
}

// ---------- 调试可视化 ----------
void buff::drawDebugInfo(cv::Mat& image) {
    // 绘制所有轮廓及索引
    //for (size_t sec = 0; sec < div_data.size(); ++sec) {
    //    for (const auto& data : div_data[sec]) {
    //        if (data.contour_d.empty()) continue;
    //        cv::drawContours(image, std::vector<std::vector<cv::Point>>{data.contour_d}, -1,
    //            cv::Scalar(255, 255, 255));
    //        cv::putText(image, std::to_string(data.position_index_d),
    //            data.center_d + cv::Point(-10, -10),
    //            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    //    }
    //}
    // 绘制 R 标
    if (!div_data[0].empty()) {
        cv::circle(image, div_data[0][0].center_d, 5, cv::Scalar(0, 255, 0), -1);
        cv::putText(image, "R", div_data[0][0].center_d + cv::Point(10, -10),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    // 绘制圆心
    for (size_t sec = 1; sec < div_data.size(); ++sec) {
        if (div_data[sec].empty()) continue;
        if (div_data[sec][0].state_index_d == RmEnergyState::InactiveCircle || div_data[sec][0].state_index_d == RmEnergyState::ActivatedCircle)
            cv::circle(image, div_data[sec][0].center_d, 5, cv::Scalar(0, 0, 255), -1);
        std::string label = (div_data[sec][0].state_index_d == RmEnergyState::ActivatedCircle) ? "AC" : "IC";
        cv::putText(image, label, div_data[sec][0].center_d + cv::Point(10, -10),
            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
}

// ---------- 参数加载 ----------
bool loadBuffParams(const std::string& filename, BuffParams& params) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[BuffParams] Cannot open YAML file: " << filename << std::endl;
        return false;
    }

    // 预处理
    fs["threshold_value"] >> params.threshold_value;
    int kw = params.morph_kernel_size.width, kh = params.morph_kernel_size.height;
    fs["morph_kernel_width"]  >> kw;
    fs["morph_kernel_height"] >> kh;
    params.morph_kernel_size = cv::Size(kw, kh);

    // 轮廓过滤
    fs["min_contour_area"]   >> params.min_contour_area;
    fs["min_circularity"]    >> params.min_circularity;
    fs["length_width_ratio_min"] >> params.length_width_ratio_min;
    fs["length_width_ratio_max"] >> params.length_width_ratio_max;

    // 扇叶角点提取
    fs["approx_poly_epsilon"] >> params.approx_poly_epsilon;

    // 轮廓分组
    fs["duplicate_distance_threshold"] >> params.duplicate_distance_threshold;

    // 能量机关结构
    fs["sector_count"] >> params.sector_count;   // 需要将 sector_count 改为非 const
    fs["sector_angle"] >> params.sector_angle;

    // 预测队列容量
    fs["max_vector_capacity"] >> params.max_vector_capacity;

    // 相机内参
    fs["camera_matrix"] >> params.camera_matrix;
    fs["dist_coeffs"] >> params.dist_coeffs;

    cv::FileNode wn = fs["world_points"];
    params.world_points.clear();
    if (wn.isSeq()) {
        for (const auto& item : wn) {
            cv::Point3f pt;
            item >> pt;           // 每个 item 是 [x, y, z]
            params.world_points.push_back(pt);
        }
    }

    fs.release();
    return true;
}

bool SolvePNPWithCenter(const BuffParams& params, const std::vector<cv::Point>& pixels, 
    const std::vector<cv::Point3f>& object_points,cv::Mat& rvec, cv::Mat& tvec) {
    std::vector<cv::Point2f> pixels_points;
    std::vector <cv::Point3f> world_points;
    for(int i = pixels.size() - 1; i >= 0; --i) {
        if (i < 0) break;
        if (pixels[i].x >= 0 && pixels[i].y >= 0) {
            pixels_points.push_back(static_cast<cv::Point2f>(pixels[i]));
            world_points.push_back(static_cast<cv::Point3f>(object_points[i]));
        }
    }
    std::cout << "Pixels points size: " << pixels_points.size() << std::endl;
    cv::solvePnP(world_points, pixels_points, params.camera_matrix, params.dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_AP3P);
    return true;
}   

bool getReprojectError(cv::Mat& image, const BuffParams& params, const std::vector<cv::Point>& pixels, 
    const std::vector<cv::Point3f>& object_points, const cv::Mat& rvec, const cv::Mat& tvec) {
    std::vector<cv::Point2f> projected_points;
    std::vector <cv::Point3f> world_points;
    for(size_t i = 0; i < pixels.size(); ++i) {
        world_points.push_back(static_cast<cv::Point3f>(object_points[i]));
    }
    cv::projectPoints(world_points, rvec, tvec, params.camera_matrix, params.dist_coeffs, projected_points);
    for(auto const& pt : projected_points) {
        cv::circle(image, pt, 5, cv::Scalar(0, 255, 255), -1);
    }
    double total_error = 0.0;
    int count = 0;
    for(size_t i = 0; i < pixels.size(); ++i) {
        if (pixels[i].x > 0 && pixels[i].y > 0) {
            std::cout << i << std::endl;
            double error = cv::norm(projected_points[count] - static_cast<cv::Point2f>(pixels[i]));
            total_error += error;
            ++count;
            std::cout << pixels[i] << std::endl;
            std::cout << "Count" << count << " Error is :" << error << std::endl;
        }
    }
    double mean_error = (count > 0) ? total_error / count : std::numeric_limits<double>::max();
    return mean_error < 5.0; // 设定一个合理的误差阈值
}

bool imageToWorldZ0(const cv::Point2f& image_point,
                    const cv::Mat& rvec, const cv::Mat& tvec,
                    const cv::Mat& K_input,
                    cv::Point2f& world_point,
                    const double& eps)
{
    cv::Mat K;
    if (K_input.type() != CV_64F)
        K_input.convertTo(K, CV_64F);
    else
        K = K_input;

    // ---------- 1. 输入校验 ----------
    if (rvec.empty() || tvec.empty() || K.empty()) {
        std::cerr << "Input matrices are empty." << std::endl;
        return false;
    }
    // 检查是否有 NaN 或 Inf
    if (cv::checkRange(rvec) == false || cv::checkRange(tvec) == false || 
        cv::checkRange(K) == false) {
        std::cerr << "Input contains NaN/Inf." << std::endl;
        return false;
    }

    // 内参
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    if (fx <= 0 || fy <= 0) {
        std::cerr << "Focal length must be positive." << std::endl;
        return false;
    }

    // ---------- 2. 旋转矩阵 ----------
    cv::Mat R;
    cv::Rodrigues(rvec, R);   // 3×3

    // ---------- 3. 归一化方向 ----------
    double x = (image_point.x - cx) / fx;
    double y = (image_point.y - cy) / fy;
    cv::Mat d = (cv::Mat_<double>(3, 1) << x, y, 1.0);

    // ---------- 4. 检查射线是否与平面平行 ----------
    // 世界 z=0 平面的法向量在相机坐标系下为 R 的第三列
    cv::Mat normal_in_cam = R.col(2);   // 3×1
    double dot_product = d.dot(normal_in_cam);  // 射线方向与平面法向量的点积
    // 如果点积接近 0，则射线与平面平行，无交点
    if (std::abs(dot_product) < eps) {
        std::cerr << "Ray is parallel to the plane, no intersection." << std::endl;
        return false;
    }

    // ---------- 5. 构建并求解 A * [X; Y; lambda] = -t ----------
    cv::Mat A(3, 3, CV_64F);
    R.col(0).copyTo(A.col(0));    // r1
    R.col(1).copyTo(A.col(1));    // r2
    d.copyTo(A.col(2));
    A.col(2) = -A.col(2);         // -d

    cv::Mat b = -tvec;

    cv::Mat solution;
    // 使用 SVD 分解，即使矩阵病态也能给出最小范数解
    if (!cv::solve(A, b, solution, cv::DECOMP_SVD)) {
        std::cerr << "Linear solver failed." << std::endl;
        return false;
    }

    // ---------- 6. 检查结果有效性 ----------
    double X = solution.at<double>(0);
    double Y = solution.at<double>(1);
    double lambda = solution.at<double>(2);

    if (std::isnan(X) || std::isnan(Y) || std::isnan(lambda) ||
        std::isinf(X) || std::isinf(Y) || std::isinf(lambda)) {
        std::cerr << "Solution contains NaN/Inf." << std::endl;
        return false;
    }

    if (lambda <= eps) {
        std::cerr << "Intersection behind camera (lambda <= 0)." << std::endl;
        return false;
    }

    world_point = cv::Point2f(static_cast<float>(X), static_cast<float>(Y));
    return true;
}
