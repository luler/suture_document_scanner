import io  # 用于处理 IO 流
import itertools  # 导入 itertools 库，用于创建迭代器的工具，例如组合
import math  # 导入 math 库，用于数学运算
from typing import Optional  # 用于类型提示，表示变量可以是指定类型或 None

import cv2  # 导入 OpenCV 库，用于图像处理
import numpy as np  # 导入 NumPy 库，用于数值计算，尤其是在处理图像时
from fastapi import FastAPI, File, UploadFile, Form, \
    HTTPException  # 导入 FastAPI 用于创建 Web API，File 用于接收文件上传，UploadFile 代表上传的文件
from fastapi.responses import StreamingResponse  # 导入 StreamingResponse 用于返回流式响应，例如图像数据
from pylsd.lsd import lsd  # 导入 pylsd 库的 lsd 函数，用于线段检测
from scipy.spatial import distance as dist  # 导入 scipy.spatial 库的 distance 模块并将其重命名为 dist，用于计算距离

from pyimagesearch import imutils  # 导入 pyimagesearch 包中的 imutils 模块，可能包含一些图像处理的实用函数
from pyimagesearch import transform  # 导入 pyimagesearch 包中的 transform 模块，可能包含图像变换的函数

app = FastAPI()  # 创建 FastAPI 应用程序实例


class DocScanner(object):
    """一个图像扫描器"""

    def __init__(self, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        """
        初始化 DocScanner 类。

        Args:
            MIN_QUAD_AREA_RATIO (float): 如果一个轮廓的角点形成的四边形覆盖的面积小于原始图像的
                MIN_QUAD_AREA_RATIO，则该轮廓将被拒绝。默认为 0.25。
            MAX_QUAD_ANGLE_RANGE (int): 如果一个轮廓的内角范围超过 MAX_QUAD_ANGLE_RANGE，
                则该轮廓也会被拒绝。默认为 40。
        """
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE

    def filter_corners(self, corners, min_dist=20):
        """过滤掉彼此距离小于 min_dist 的角点"""

        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= min_dist
                       for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners

    def angle_between_vectors_degrees(self, u, v):
        """返回两个向量之间的角度，以度为单位"""
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(self, p1, p2, p3):
        """
        返回由线段 p2 到 p1 和线段 p2 到 p3 形成的角，以度为单位。
        """
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):
        """
        返回四边形最大和最小内角之间的范围。
        输入的四边形必须是一个 NumPy 数组，顶点按顺时针顺序排列，从左上角顶点开始。
        """
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)

    def get_corners(self, img):
        """
        返回在输入图像中找到的角点的列表 (以 (x, y) 元组表示)。通过适当的
        预处理和过滤，它应该最多输出 10 个潜在的角点。
        这是 get_contours 使用的实用函数。输入图像应该是经过缩放和 Canny 滤波后的图像。
        """
        lines = lsd(img)  # 使用 LSD 算法检测图像中的线段

        corners = []
        if lines is not None:
            lines = lines.squeeze().astype(np.int32).tolist()  # 将检测到的线段转换为整数坐标列表
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)  # 创建一个用于绘制水平线的空白画布
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)  # 创建一个用于绘制垂直线的空白画布
            for line in lines:
                x1, y1, x2, y2, _ = line
                if abs(x2 - x1) > abs(y2 - y1):  # 判断是否为近似水平线
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])  # 对水平线的端点按 x 坐标排序
                    cv2.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255,
                             2)  # 在水平线画布上绘制加粗的线
                else:  # 判断是否为近似垂直线
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])  # 对垂直线的端点按 y 坐标排序
                    cv2.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255,
                             2)  # 在垂直线画布上绘制加粗的线

            lines = []
            (contours, hierarchy) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)  # 在水平线画布上查找轮廓
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]  # 选择最长的两个水平线轮廓
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)  # 清空水平线画布
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))  # 重塑轮廓形状
                min_x = np.amin(contour[:, 0], axis=0) + 2  # 找到轮廓的最小 x 坐标
                max_x = np.amax(contour[:, 0], axis=0) - 2  # 找到轮廓的最大 x 坐标
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))  # 计算最小 x 坐标对应的 y 坐标的平均值
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))  # 计算最大 x 坐标对应的 y 坐标的平均值
                lines.append((min_x, left_y, max_x, right_y))  # 添加线段信息
                cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)  # 在画布上绘制细线
                corners.append((min_x, left_y))  # 添加角点
                corners.append((max_x, right_y))  # 添加角点

            (contours, hierarchy) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_NONE)  # 在垂直线画布上查找轮廓
            contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]  # 选择最长的两个垂直线轮廓
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)  # 清空垂直线画布
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))  # 重塑轮廓形状
                min_y = np.amin(contour[:, 1], axis=0) + 2  # 找到轮廓的最小 y 坐标
                max_y = np.amax(contour[:, 1], axis=0) - 2  # 找到轮廓的最大 y 坐标
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))  # 计算最小 y 坐标对应的 x 坐标的平均值
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))  # 计算最大 y 坐标对应的 x 坐标的平均值
                lines.append((top_x, min_y, bottom_x, max_y))  # 添加线段信息
                cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)  # 在画布上绘制细线
                corners.append((top_x, min_y))  # 添加角点
                corners.append((bottom_x, max_y))  # 添加角点

            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)  # 找到水平线和垂直线相交的点
            corners += zip(corners_x, corners_y)  # 将相交点添加到角点列表

        corners = self.filter_corners(corners)  # 过滤距离过近的角点
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        return (len(cnt) == 4 and cv2.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO
                and self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)

    def get_contour(self, rescaled_image):
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        gray = cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
        gray = cv2.GaussianBlur(gray, (7, 7), 0)  # 对灰度图进行高斯模糊

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))  # 创建一个矩形结构元素
        dilated = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)  # 对灰度图进行闭运算，填充小的孔洞

        edged = cv2.Canny(dilated, 0, CANNY)  # 使用 Canny 边缘检测算法检测边缘
        test_corners = self.get_corners(edged)  # 获取检测到的角点

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):  # 从角点中选取 4 个点进行组合
                points = np.array(quad)
                points = transform.order_points(points)  # 对选取的点进行排序，使其按特定顺序排列
                points = np.array([[p] for p in points], dtype="int32")  # 将点转换为 OpenCV 轮廓所需的格式
                quads.append(points)  # 将形成的四边形添加到列表中

            quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]  # 按面积降序排序并选择前 5 个四边形
            quads = sorted(quads, key=self.angle_range)  # 按内角范围升序排序

            if quads:
                approx = quads[0]
                if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                    approx_contours.append(approx)

        (cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 在边缘图像中查找轮廓
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]  # 按面积降序排序并选择前 5 个轮廓

        for c in cnts:
            approx = cv2.approxPolyDP(c, 80, True)  # 对轮廓进行多边形逼近
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv2.contourArea)  # 选择面积最大的有效轮廓

        return screenCnt.reshape(4, 2)  # 重塑轮廓形状为 (4, 2) 的数组

    def scan(self, image, correct_perspective: bool = True):
        RESCALED_HEIGHT = 500.0

        if not correct_perspective:
            # 在不进行透视校正的情况下转换为灰度和应用阈值
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
            sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
            thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
            return thresh

        orig = image.copy()
        rescaled_image = imutils.resize(image, height=int(RESCALED_HEIGHT))  # 缩放图像高度

        screenCnt = self.get_contour(rescaled_image)  # 获取文档的轮廓

        ratio = image.shape[0] / RESCALED_HEIGHT  # 计算原始图像与缩放后图像的高度比率
        warped = transform.four_point_transform(orig, screenCnt * ratio)  # 进行四点透视变换，校正文档视角

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)  # 将校正后的图像转换为灰度图

        sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,
                                       15)  # 应用自适应阈值，将图像转换为二值图像

        return thresh


scanner = DocScanner()  # 创建 DocScanner 类的实例


@app.post("/scan_document")
async def scan_document(
        file: UploadFile = File(...),
        correct_perspective: Optional[bool] = Form(True)
):
    """
    上传图像并执行文档扫描的端点。

    Args:
        file: 要扫描的图像文件。
        correct_perspective: 如果为 True (默认值)，则图像将被校正透视以模拟俯视图。
                           如果为 False，则将保留原始图像布局。
    """
    contents = await file.read()  # 读取上传文件的内容
    nparr = np.frombuffer(contents, np.uint8)  # 将文件内容转换为 NumPy 数组
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 使用 OpenCV 解码图像

    if image is None:
        raise HTTPException(status_code=400, detail="无法读取图像")

    processed_image = scanner.scan(image, correct_perspective=correct_perspective)  # 使用扫描器处理图像

    # 将处理后的图像转换回字节
    is_success, im_buf_arr = cv2.imencode(".png", processed_image)  # 将处理后的图像编码为 PNG 格式
    if not is_success:
        raise HTTPException(status_code=400, detail="无法编码处理后的图像")

    io_buf = io.BytesIO(im_buf_arr.tobytes())  # 将编码后的图像数据放入内存中的字节流

    return StreamingResponse(io_buf, media_type="image/png")  # 返回包含处理后图像的流式响应，媒体类型为 image/png


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)  # 当脚本直接运行时，启动 FastAPI 应用程序，监听所有接口的 8000 端口
