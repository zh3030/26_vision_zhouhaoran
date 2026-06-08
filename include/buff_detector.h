#pragma once
/**
 * @file buff_detector.h
 * @brief 能量机关视觉处理核心类定义
 *
 * 包含能量机关视觉处理的所有核心数据结构、参数定义和函数声明。
 * 设计为模块化、易于调试和扩展的接口，供实现文件 buff_detector.cpp 使用。
 *
 * @author Zhou Haoran
 * @date 2026-05-20
 */

// 均由CMake控制

/// 开启调试绘制
#define BUFF_TEST
/// 预留宏
#define BUFF_COUT
/// 开启耗时统计
#define BUFF_TIME
/// 开启辅助点输出
#define ADD_POINTS

/** @} */

#include <vector>
#include <deque>

#include <opencv2/core.hpp>            // cv::Point, cv::Rect, cv::Mat, cv::fastAtan2 等

// 前向声明 OpenCV 图像处理所需类型（实际使用由实现文件包含）
namespace cv {
    class Mat;
    template<typename T> class Point_;
    typedef Point_<int> Point2i;
    typedef Point2i Point;
    typedef Point_<float> Point2f;
    class TickMeter;
}

/**
 * @struct BuffParams
 * @brief 能量机关视觉处理的所有可调参数
 */
struct BuffParams {
    // 预处理
    int threshold_value = 200;                                   ///< 红蓝通道差分二值化阈值
    cv::Size morph_kernel_size = cv::Size(5, 5);                 ///< 形态学操作核大小
    // 轮廓过滤
    double min_contour_area = 100.0;                             ///< 最小轮廓面积
    double min_circularity = 0.6;                                ///< 最小圆形度
    double length_width_ratio_min = 0.5;                         ///< 长宽比下限
    double length_width_ratio_max = 1.5;                         ///< 长宽比上限
    // 扇叶角点提取
    double approx_poly_epsilon = 3.0;                            ///< 多边形逼近精度
    // 轮廓分组
    double duplicate_distance_threshold = 100.0;                 ///< 同一扇区重复轮廓判定距离
    // 能量机关结构
    int sector_count = 5;                                        ///< 待激活扇叶数量（不含R和待激活圆）
    double sector_angle = 72.0;                                  ///< 每个扇区角度 = 360/sector_count
    // 用于预测向量的最大容量
    int max_vector_capacity = 10;                                ///< 角度队列最大长度
    // 相机内参（用于后续位姿估计）
    cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) << 
    1600.0, 0.0, 1280.0, 0.0, 1600.0, 720.0, 0.0, 0.0, 1.0);     ///< 相机内参矩阵
    cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_32F);          ///< 相机畸变系数
    // 世界坐标系下的固定参考点（中心R和各扇叶位置）
    std::vector<cv::Point3f> world_points =                      ///< 世界坐标系下的固定参考点（中心R和各扇叶位置)
    {
        cv::Vec3f(640, 360, 110), // center R
        cv::Vec3f(990, 360, 0),   // 0°
        cv::Vec3f(748.156, 692.87, 0),   // 72°
        cv::Vec3f(356.844, 565.725, 0),   // 144°
        cv::Vec3f(356.844, 154.275, 0),   // 216°
        cv::Vec3f(748.156, 27.130, 0)     // 288°
    };
};

/**
 * @enum RmEnergyState
 * @brief 能量机关各元素状态枚举
 */
enum class RmEnergyState {
    Unknown,                        ///< 未知
    ActivatedBladeUnoccluded,       ///< 已激活扇叶（无遮挡）
    ActivatedBladePartiallyOccluded,///< 已激活扇叶（部分遮挡）
    ActivatedCircle,                ///< 已激活圆（装甲板圆）
    InactiveBlade,                  ///< 未激活扇叶
    InactiveCircle,                 ///< 待激活圆（指示圆）
    CenterR,                        ///< 中心 R 标
    Count                           ///< 枚举计数
};

/**
 * @struct AngleData
 * @brief 存储一帧的角度、距离与时间信息
 */
struct AngleData {
    double angle_d = 0.0;       ///< 当前角度（度）
    double distance_d = 0.0;    ///< 待激活圆到 R 中心的距离
    double current_BUFF_TIME_d = 0.0;///< 时间戳（秒）
};

/**
 * @struct RoiData
 * @brief 动态 ROI 裁剪区域信息
 */
struct RoiData {
    int cropX;      ///< ROI 左上角 X 坐标
    int cropY;      ///< ROI 左上角 Y 坐标
    int cropradius; ///< ROI 半径
};

/**
 * @struct BuffData
 * @brief 单个轮廓对应的属性数据
 */
struct BuffData {
    int position_index_d = -1;                  ///< 扇区索引：0=R, 1=待激活圆, 2~5 逆时针扇叶
    int child_sum_d = -1;                       ///< 直接子轮廓数量，-1 表示无子轮廓
    int father_index_d = -1;                    ///< 父轮廓索引，-1 表示最外层
    RmEnergyState state_index_d = RmEnergyState::Unknown; ///< 类型状态
    double angle_d = 0.0;                       ///< 相对 R 中心的角度（度）
    double length_width_ratio_d = 0.0;          ///< 长宽比
    double circularity_d = 0.0;                 ///< 圆形度
    double area_d = 0.0;                        ///< 面积
    double length_d = 0.0;                      ///< 轮廓周长
    cv::Point center_d = cv::Point();           ///< 轮廓重心
    cv::Rect bounding_rect_d = cv::Rect();      ///< 外接矩形
    std::vector<cv::Point> corners_d;           ///< 角点（部分遮挡时存储在扇区首个元素）
    std::vector<cv::Point> contour_d;           ///< 轮廓点集
};

/**
 * @namespace geometry_utils
 * @brief 几何计算工具函数（模板实现，定义在头文件）
 */
namespace geometry_utils {

    /**
     * @brief 计算两点欧氏距离的平方
     * @tparam T 点类型，需含有 x, y 成员
     * @param a 第一个点
     * @param b 第二个点
     * @return 平方距离
     */
    template <typename T>
    double square_distance(const T& a, const T& b) {
        return std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2);
    }

    /**
     * @brief 计算两点的中点
     * @tparam T 点类型，需可构造 T(x, y)
     * @param a 第一个点
     * @param b 第二个点
     * @return 中点坐标
     */
    template <typename T>
    T middle_point(const T& a, const T& b) {
        return T((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
    }

    /**
     * @brief 计算从中心指向目标点的向量与 Y 轴正方向的夹角
     * @tparam T 点类型，需含有 x, y 成员
     * @param center 中心点
     * @param point 目标点
     * @return 夹角（度），范围 [0, 360)
     */
    template <typename T>
    double angle_from_center(const T& center, const T& point) {
        return cv::fastAtan2(point.x - center.x, center.y - point.y);
    }
}

/**
 * @class buff
 * @brief 能量机关视觉处理核心类
 *
 * 负责从图像中检测能量机关的各个组件（R标、待激活圆、已激活圆、扇叶），
 * 提取角点、计算角度，并输出用于后续预测的数据。
 */
class buff {
public:
    /**
     * @brief 构造函数，初始化内部数据结构
     * @param params 可调参数，默认使用 BuffParams()
     */
    buff(const BuffParams& params = BuffParams());

    /**
     * @brief 世界坐标系下的固定参考点
     */
    const std::vector<cv::Point> world_points =
    {
        cv::Point(640, 360),   // center R
        cv::Point(990, 360),   // 0°
        cv::Point(748, 693),   // 72°
        cv::Point(357, 566),   // 144°
        cv::Point(357, 154),   // 216°
        cv::Point(748, 27)     // 288°
    };

    /**
     * @brief 主处理函数，对一帧图像执行完整的检测流程
     * @param input 输入图像（BGR 顺序），会原地绘制调试信息
     */
    void run(cv::Mat& input);

    /** @name 对外数据接口 */
    ///@{
#ifdef ADD_POINTS
    /**
     * @brief 获取最新一帧中各扇区的中心点
     * @return 包含 R、待激活圆、各扇叶中心的点向量，索引与扇区索引一致
     */
    std::vector<cv::Point> get_points() const;
#endif

    /**
     * @brief 获取按扇区分组后的轮廓数据
     * @return 二维数组：div_data[0]=R, [1]=待激活圆, [2..5]=各扇叶
     */
    const std::vector<std::vector<BuffData>>& get_div_data() const;

    /**
     * @brief 获取检测到的 R 标中心
     * @return R 标中心点
     */
    const cv::Point& get_center() const;

    /**
     * @brief 获取角度历史信息队列
     * @return 最近若干帧的 AngleData 双端队列
     */
    const std::deque<AngleData>& get_angle_info() const;

    /**
     * @brief 获取动态 ROI 信息
     * @return 当前帧计算出的 ROI 裁剪参数
     */
    const RoiData& get_roi_info() const;
    ///@}

private:
    BuffParams m_params;                            ///< 全部可调参数
    AngleData angle_data;                           ///< 当前帧角度数据
    RoiData roi_info;                               ///< 当前帧 ROI 信息
    cv::Mat sub_image;                              ///< 预处理后的二值图像（暂存）
    double current_angle_m = 0.0;                   ///< 当前参考角度（待激活圆角度）
    std::deque<AngleData> angle_deque;              ///< 角度历史队列

#ifdef ADD_POINTS
    std::vector<cv::Point> m_vec_center;            ///< 各扇区中心点（条件编译输出）
#endif

    cv::Point m_center;                             ///< R 标中心
    std::vector<std::vector<BuffData>> div_data;   ///< 分组数据，索引见结构说明

    // ---------- 图像处理 ----------
    /**
     * @brief 图像预处理：红蓝通道差分、二值化、滤波、膨胀
     * @param input  输入彩色图像
     * @param output 输出二值图像
     */
    void subImage(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief 查找轮廓并建立层级关系
     * @param image      输入二值图像
     * @param contours   输出轮廓容器
     * @param hierarchy  输出层级关系
     */
    void findContours(const cv::Mat& image,
        std::vector<std::vector<cv::Point>>& contours,
        std::vector<cv::Vec4i>& hierarchy);

    // ---------- 轮廓初始化 ----------
    /**
     * @brief 遍历原始轮廓，过滤并填充 BuffData 属性
     * @param contours  输入原始轮廓
     * @param hierarchy 轮廓层级
     * @param vec_data  输出填充好的 BuffData 列表
     */
    void initializeContours(std::vector<std::vector<cv::Point>>& contours,
        std::vector<cv::Vec4i>& hierarchy,
        std::vector<BuffData>& vec_data);

    // ---------- 圆心 R 提取 ----------
    /**
     * @brief 从轮廓列表中提取中心 R 标
     * @param vec_data   输入/输出轮廓数据（匹配项会被移出）
     * @param div_data   输出分组容器，R 标放入 div_data[0]
     * @param center_out 输出 R 标的中心坐标
     * @return 是否成功提取
     */
    bool extractCenterR(std::vector<BuffData>& vec_data,
        std::vector<std::vector<BuffData>>& div_data,
        cv::Point& center_out);

    // ---------- 待激活圆提取 ----------
    /**
     * @brief 提取待激活圆（内轮廓且至少含一个子轮廓）
     * @param vec_data    输入/输出轮廓数据
     * @param center_r    R 标中心
     * @param div_data    输出分组容器
     * @param current_angle 输出待激活圆的角度（作为零位参考）
     * @param centers_out 输出中心点数组，索引1存入待激活圆中心
     * @return 是否成功提取
     */
    bool extractInactiveCircleCenter(std::vector<BuffData>& vec_data,
        const cv::Point& center_r,
        std::vector<std::vector<BuffData>>& div_data,
        double& current_angle,
        std::vector<cv::Point>& centers_out);

    // ---------- 已激活圆提取 ----------
    /**
     * @brief 提取已激活圆（高圆形度、无子轮廓、最外层）
     * @param vec_data    输入/输出轮廓数据
     * @param center_r    R 标中心
     * @param current_angle 当前待激活圆参考角度（用于扇区划分）
     * @param div_data    输出分组容器
     * @param centers_out 输出中心点数组
     */
    void extractActivatedCircleCenter(std::vector<BuffData>& vec_data,
        const cv::Point& center_r, const double current_angle,
        std::vector<std::vector<BuffData>>& div_data,
        std::vector<cv::Point>& centers_out);

    // ---------- 轮廓分组（扇叶分配） ----------
    /**
     * @brief 将剩余轮廓分配到角度最近的已有扇区
     * @param vec_data    输入/输出剩余轮廓
     * @param center_r    R 标中心
     * @param current_angle 当前待激活圆参考角度
     * @param div_data    输入/输出分组容器
     */
    void processContours(std::vector<BuffData>& vec_data,
        const cv::Point& center_r,
        double current_angle,
        std::vector<std::vector<BuffData>>& div_data);

    // ---------- 角点提取 ----------
    /**
     * @brief 对各扇区分别提取角点（无遮挡、部分遮挡、待激活扇叶）
     * @param image     用于绘图的图像
     * @param center_r  R 标中心
     * @param div_data  分组数据
     */
    void findCorners(cv::Mat& image, const cv::Point& center_r,
        std::vector<std::vector<BuffData>>& div_data);

    /**
     * @brief 无遮挡扇叶角点提取（凸包 + 多边形逼近）
     * @param blade 扇叶轮廓数据
     * @param image 用于绘图的图像
     */
    void extractUnoccludedBladeCorners(BuffData& blade, cv::Mat& image);

    /**
     * @brief 部分遮挡扇叶角点提取（合并碎片轮廓 + 两次多边形逼近）
     * @param sector_data 该扇区所有数据（首个元素为已激活圆）
     * @param image       用于绘图的图像
     */
    void extractPartiallyOccludedBladeCorners(std::vector<BuffData>& sector_data,
        cv::Mat& image);

    /**
     * @brief 待激活圆所在扇区的角点提取（基于内轮廓中心与最小外接矩形）
     * @param sector_data 扇区数据
     * @param image       用于绘图的图像
     */
    void extractInactiveBladeCornersByInnerCenters(std::vector<BuffData>& sector_data,
        cv::Mat& image);

    /**
     * @brief 合并扇区中除第一个元素外的所有轮廓点
     * @param data_vec 扇区数据
     * @return 合并后的点集
     */
    std::vector<cv::Point> mergeContoursExceptFirst(const std::vector<BuffData>& data_vec);

    // ---------- 辅助计算 ----------
    /**
     * @brief 根据角度差计算所属扇区索引
     * @param angle           待求角度
     * @param reference_angle 参考角度（待激活圆角度）
     * @return 扇区索引（1 ~ sector_count）
     */
    int get_index(double angle, double reference_angle) const;

    /**
     * @brief 解析轮廓层级，获得父轮廓索引与子轮廓数量
     * @param hierarchy  轮廓层级关系
     * @param index      当前轮廓索引
     * @param hasParent  输出父轮廓索引，-1 表示无父轮廓
     * @param childCount 输出子轮廓数量，-1 表示无子轮廓
     */
    void getHierarchyInfo(const std::vector<cv::Vec4i>& hierarchy, int index,
        int& hasParent, int& childCount);

    // ---------- 调试可视化 ----------
    /**
     * @brief 在图像上绘制 R 标、各扇区圆心等调试信息
     * @param image 输入/输出图像，原地绘制
     */
    void drawDebugInfo(cv::Mat& image);
};
/**
 * @brief 从 YAML 文件加载 BuffParams 参数，缺失项保留默认值
 * @param filename YAML 文件路径
 * @param params   输出参数结构体（会覆盖已有值）
 * @return 读取成功返回 true，文件无法打开或格式错误返回 false
 */
bool loadBuffParams(const std::string& filename, BuffParams& params);

/**
 * @brief 使用提取的中心点和世界坐标点进行 PnP 位姿估计
 * @param params        视觉处理参数，包含相机内参等
 * @param pixels        提取的像素坐标点，索引与 world_points 一一对应
 * @param object_points 世界坐标系下的参考点，索引与 pixels 一一对应
 * @param rvec          输出旋转向量
 * @param tvec          输出平移向量
 * @return 成功返回 true，输入点不足或计算失败返回 false
 */
bool SolvePNPWithCenter(const BuffParams& params, const std::vector<cv::Point>& pixels, 
    const std::vector<cv::Point3f>& object_points,cv::Mat& rvec, cv::Mat& tvec);
    
bool getReprojectError(cv::Mat& image, const BuffParams& params, const std::vector<cv::Point>& pixels, 
    const std::vector<cv::Point3f>& object_points, const cv::Mat& rvec, const cv::Mat& tvec);
