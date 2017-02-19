#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "QDebug"
#include "string.h"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <QMessageBox>
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void MenuBarInit();
    void show_Image(QImage* img);
    void show_Image_2(QImage* img);

    void on_action_Load_triggered();

    void on_action_Convert2Gray_triggered();

    void on_action_Binary_triggered();

    void on_action_Homogeneous_Blur_triggered();

    void on_action_Gaussian_Blur_triggered();

    void on_action_Median_Blur_triggered();

    void on_action_Bilateral_Blur_triggered();

    void on_action_Dilation_triggered();

    void on_action_Erosion_triggered();

    void on_action_Opening_triggered();

    void on_action_Closing_triggered();

    void on_action_Morphological_Gradient_triggered();

    void on_action_Top_Hat_triggered();

    void on_action_Black_Hat_triggered();

    void on_action_Discrete_Fourier_Transform_triggered();

    void on_action_Sobel_triggered();

    void on_action_Laplace_triggered();

    void on_horizontalSlider_Canny_valueChanged(int value);

    void on_action_Canny_triggered();

    void on_okButton_Canny_clicked();

    void on_action_Hough_Line_triggered();

    void on_okButton_HoughLine_clicked();

    void on_horizontalSlider_HoughLine_valueChanged(int value);

    void on_action_Hough_Circle_triggered();

    void on_okButton_HoughCircle_clicked();

    void on_horizontalSlider_HougCircle_valueChanged(int value);

    void on_action_Histogram_Equalization_triggered();

    void on_action_Histogram_Calculation_triggered();

    void on_action_Back_Projection_triggered();

    void on_horizontalSlider_BackProjection_valueChanged(int bins);

    void on_okButton_BackProjection_clicked();

    void on_action_Finding_Contours_triggered();

    void on_horizontalSlider_FindContours_valueChanged(int value);

    void on_okButton_FindContours_clicked();

    void on_action_Convex_Hull_triggered();

    void on_okButton_Hull_clicked();

    void on_horizontalSlider_Hull_valueChanged(int thresh);

    void on_action_Creating_Bounding_For_Contours_triggered();

    void on_okButton_CreatBounding_clicked();

    void on_horizontalSlider_CreatBounding_valueChanged(int thresh);

    void on_action_Image_Moments_triggered();

    void on_okButton_Image_Moments_clicked();

    void on_horizontalSlider_Image_Moments_valueChanged(int thresh);

    void on_action_Point_Polygon_Test_triggered();

    void on_action_Harris_corner_triggered();

    void on_horizontalSlider_Harris_Corners_valueChanged(int thresh);

    void on_okButton_Harris_Corners_clicked();

    void on_action_Shi_Tomasi_corner_triggered();

    void on_horizontalSlider_ShiTomasi_Corners_valueChanged(int maxCorners);

    void on_okButton_ShiTomasi_Corners_clicked();

    void on_action_Subpixeles_corner_triggered();

    void on_horizontalSlider_Subpixeles_Corners_valueChanged(int maxCorners);

    void on_okButton_Subpixeles_Corners_clicked();

    void on_action_Feature_Detection_triggered();

    void on_action_Feature_Description_triggered();

    void on_action_Feature_Matching_FLANN_triggered();

    void on_okButton_Image2_clicked();

    void on_action_Find_Object_triggered();

    void detectAndDisplay(cv::Mat frame );

    void on_action_Person_FaceDetection_triggered();

    void on_actionImagePro_About_triggered();

    void on_actionQt_About_triggered();

private:
    Ui::MainWindow *ui;

    QString fileName;
    QImage *image;

    cv::Mat src;
    cv::Mat dst;
    //face detection
    cv::String face_cascade_name = "./haarcascade_frontalface_alt.xml";
    cv::String eyes_cascade_name = "./haarcascade_eye_tree_eyeglasses.xml";
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

};

#endif // MAINWINDOW_H
