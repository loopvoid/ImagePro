#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->ImageLabel_2->hide();
    ui->okButton_Image2->hide();

    ui->spinBox_Canny->hide();
    ui->horizontalSlider_Canny->hide();
    ui->okButton_Canny->hide();

    ui->spinBox_HoughLine->hide();
    ui->horizontalSlider_HoughLine->hide();
    ui->okButton_HoughLine->hide();

    ui->spinBox_HoughCircle->hide();
    ui->horizontalSlider_HougCircle->hide();
    ui->okButton_HoughCircle->hide();

    ui->spinBox_BackProjection->hide();
    ui->horizontalSlider_BackProjection->hide();
    ui->okButton_BackProjection->hide();

    ui->spinBox_FindContours->hide();
    ui->horizontalSlider_FindContours->hide();
    ui->okButton_FindContours->hide();

    ui->spinBox_Hull->hide();
    ui->horizontalSlider_Hull->hide();
    ui->okButton_Hull->hide();

    ui->spinBox_CreatBounding->hide();
    ui->horizontalSlider_CreatBounding->hide();
    ui->okButton_CreatBounding->hide();

    ui->spinBox_Image_Moments->hide();
    ui->horizontalSlider_Image_Moments->hide();
    ui->okButton_Image_Moments->hide();

    ui->spinBox_Harris_Corners->hide();
    ui->horizontalSlider_Harris_Corners->hide();
    ui->okButton_Harris_Corners->hide();

    ui->spinBox_ShiTomasi_Corners->hide();
    ui->horizontalSlider_ShiTomasi_Corners->hide();
    ui->okButton_ShiTomasi_Corners->hide();

    ui->spinBox_Subpixeles_Corners->hide();
    ui->horizontalSlider_Subpixeles_Corners->hide();
    ui->okButton_Subpixeles_Corners->hide();


    //初始化菜单栏
    ui->actionImagePro_About->setEnabled(true);
    ui->actionQt_About->setEnabled(true);
    ui->action_Back_Projection->setEnabled(false);
    ui->action_Bilateral_Blur->setEnabled(false);
    ui->action_Binary->setEnabled(false);
    ui->action_Black_Hat->setEnabled(false);
    ui->action_Camera_calibration->setEnabled(false);
    ui->action_Canny->setEnabled(false);
    ui->action_Closing->setEnabled(false);
    ui->action_Closing->setEnabled(false);
    ui->action_Convert2Gray->setEnabled(false);
    ui->action_Convex_Hull->setEnabled(false);
    ui->action_Creating_Bounding_For_Contours->setEnabled(false);
    ui->action_Dilation->setEnabled(false);
    ui->action_Discrete_Fourier_Transform->setEnabled(false);
    ui->action_DIY_corner->setEnabled(false);
    ui->action_Erosion->setEnabled(false);
    ui->action_Feature_Description->setEnabled(true);
    ui->action_Feature_Detection->setEnabled(false);
    ui->action_Feature_Matching_FLANN->setEnabled(true);
    ui->action_Finding_Contours->setEnabled(false);
    ui->action_Find_Object->setEnabled(true);
    ui->action_Gaussian_Blur->setEnabled(false);
    ui->action_Harris_corner->setEnabled(false);
    ui->action_Histogram_Calculation->setEnabled(false);
    ui->action_Histogram_Comparison->setEnabled(false);
    ui->action_Histogram_Equalization->setEnabled(false);
    ui->action_Homogeneous_Blur->setEnabled(false);
    ui->action_Hough_Circle->setEnabled(false);
    ui->action_Hough_Line->setEnabled(false);
    ui->action_Image_Moments->setEnabled(false);
    ui->action_Laplace->setEnabled(false);
    ui->action_Median_Blur->setEnabled(false);
    ui->action_Morphological_Gradient->setEnabled(false);
    ui->action_Opening->setEnabled(false);
    ui->action_Person_FaceDetection->setEnabled(true);
    ui->action_Point_Polygon_Test->setEnabled(false);
    ui->action_Save->setEnabled(false);
    ui->action_Setero_Calibrarion->setEnabled(false);
    ui->action_Shi_Tomasi_corner->setEnabled(false);
    ui->action_Sobel->setEnabled(false);
    ui->action_Subpixeles_corner->setEnabled(false);
    ui->action_Template_Matching->setEnabled(false);
    ui->action_Top_Hat->setEnabled(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::MenuBarInit()
{
    ui->actionImagePro_About->setEnabled(true);
    ui->actionQt_About->setEnabled(true);
    ui->action_Back_Projection->setEnabled(true);
    ui->action_Bilateral_Blur->setEnabled(true);
    ui->action_Binary->setEnabled(true);
    ui->action_Black_Hat->setEnabled(true);
    ui->action_Camera_calibration->setEnabled(true);
    ui->action_Canny->setEnabled(true);
    ui->action_Closing->setEnabled(true);
    ui->action_Closing->setEnabled(true);
    ui->action_Convert2Gray->setEnabled(true);
    ui->action_Convex_Hull->setEnabled(true);
    ui->action_Creating_Bounding_For_Contours->setEnabled(true);
    ui->action_Dilation->setEnabled(true);
    ui->action_Discrete_Fourier_Transform->setEnabled(true);
    ui->action_DIY_corner->setEnabled(true);
    ui->action_Erosion->setEnabled(true);
    ui->action_Feature_Description->setEnabled(true);
    ui->action_Feature_Detection->setEnabled(true);
    ui->action_Feature_Matching_FLANN->setEnabled(true);
    ui->action_Finding_Contours->setEnabled(true);
    ui->action_Find_Object->setEnabled(true);
    ui->action_Gaussian_Blur->setEnabled(true);
    ui->action_Harris_corner->setEnabled(true);
    ui->action_Histogram_Calculation->setEnabled(true);
    ui->action_Histogram_Comparison->setEnabled(true);
    ui->action_Histogram_Equalization->setEnabled(true);
    ui->action_Homogeneous_Blur->setEnabled(true);
    ui->action_Hough_Circle->setEnabled(true);
    ui->action_Hough_Line->setEnabled(true);
    ui->action_Image_Moments->setEnabled(true);
    ui->action_Laplace->setEnabled(true);
    ui->action_Median_Blur->setEnabled(true);
    ui->action_Morphological_Gradient->setEnabled(true);
    ui->action_Opening->setEnabled(true);
    ui->action_Person_FaceDetection->setEnabled(true);
    ui->action_Point_Polygon_Test->setEnabled(true);
    ui->action_Save->setEnabled(true);
    ui->action_Setero_Calibrarion->setEnabled(true);
    ui->action_Shi_Tomasi_corner->setEnabled(true);
    ui->action_Sobel->setEnabled(true);
    ui->action_Subpixeles_corner->setEnabled(true);
    ui->action_Template_Matching->setEnabled(true);
    ui->action_Top_Hat->setEnabled(true);
}

void MainWindow::show_Image(QImage* img)
{
    ui->ImageLabel->clear();
    ui->ImageLabel->setPixmap(QPixmap::fromImage(*img));
}
void MainWindow::show_Image_2(QImage* img)
{
    ui->ImageLabel_2->clear();
    ui->ImageLabel_2->setPixmap(QPixmap::fromImage(*img));
}
//Load
void MainWindow::on_action_Load_triggered()
{
    MenuBarInit();

    fileName = QFileDialog::getOpenFileName(this,tr("ImageProcess"),".",tr("Image Files(*.jpg *.png *.bmp)"));
    std::string str;
    str = fileName.toStdString();
    std::cout<<str<<std::endl;
    src = cv::imread(fileName.toStdString());
    dst = src;

    image = new QImage(fileName);
    show_Image(image);
}

//Convert to Gray
void MainWindow::on_action_Convert2Gray_triggered()
{
    if(src.channels() != 1)
    {
        cv::cvtColor(src,src,CV_RGB2GRAY);
    }
    cv::namedWindow("Gray Image",1);
    imshow("Gray Image",src);
}

//Binary
void MainWindow::on_action_Binary_triggered()
{
    if(src.channels() != 1)
    {
        cv::cvtColor(src,src,CV_RGB2GRAY);
    }
    cv::threshold(src, src, 0, 255, CV_THRESH_OTSU);
    cv::namedWindow("Binary Image",1);
    imshow("Binary Image",src);
}

//Homogeneous_Blur
void MainWindow::on_action_Homogeneous_Blur_triggered()
{
    blur( src, dst, cv::Size( 9, 9 ), cv::Point(-1,-1));
    cv::namedWindow("Homogeneous Blur",1);
    imshow("Homogeneous Blur",dst);
}

//Gaussian_Blur
void MainWindow::on_action_Gaussian_Blur_triggered()
{
    GaussianBlur( src, dst, cv::Size( 9, 9 ), 0, 0 );
    cv::namedWindow("Gaussian Blur",1);
    imshow("Gaussian Blur",dst);
}

//Median_Blur
void MainWindow::on_action_Median_Blur_triggered()
{
    medianBlur ( src, dst, 9 );
    cv::namedWindow("Median Blur",1);
    imshow("Median Blur",dst);
}

//Bilateral_Blur
void MainWindow::on_action_Bilateral_Blur_triggered()
{
    bilateralFilter ( src, dst, 3, 6, 3 );
    cv::namedWindow("Bilateral Blur",1);
    imshow("Bilateral Blur",dst);
}

//Dilation
void MainWindow::on_action_Dilation_triggered()
{
    int erosion_size = 3;
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                           cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                           cv::Point( erosion_size, erosion_size ) );
    cv::erode( src, dst, element);
    cv::namedWindow("Dilation",1);
    cv::imshow("Dilation",dst);
}

//Erosion
void MainWindow::on_action_Erosion_triggered()
{
    int erosion_size = 3;
    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                           cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                           cv::Point( erosion_size, erosion_size ) );
    cv::dilate( src, dst, element);
    cv::namedWindow("Erosion",1);
    cv::imshow("Erosion",dst);
}

//Opening
void MainWindow::on_action_Opening_triggered()
{
    int morph_size = 3;
    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                         cv::Size( 2*morph_size + 1, 2*morph_size+1 ),
                                         cv::Point( morph_size, morph_size ) );
    cv::morphologyEx( src, dst, cv::MORPH_OPEN , element );
    cv::namedWindow("Opening",1);
    imshow("Opening",dst);
}

//Closing
void MainWindow::on_action_Closing_triggered()
{
    int morph_size = 3;
    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                         cv::Size( 2*morph_size + 1, 2*morph_size+1 ),
                                         cv::Point( morph_size, morph_size ) );
    cv::morphologyEx( src, dst, cv::MORPH_CLOSE , element );
    cv::namedWindow("Closing",1);
    imshow("Closing",dst);
}

//Morphological_Gradient
void MainWindow::on_action_Morphological_Gradient_triggered()
{
    int morph_size = 3;
    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                         cv::Size( 2*morph_size + 1, 2*morph_size+1 ),
                                         cv::Point( morph_size, morph_size ) );
    cv::morphologyEx( src, dst, cv::MORPH_GRADIENT , element );
    cv::namedWindow("Morphological Gradient",1);
    imshow("Morphological Gradient",dst);
}

//Top_Hat
void MainWindow::on_action_Top_Hat_triggered()
{
    int morph_size = 3;
    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                         cv::Size( 2*morph_size + 1, 2*morph_size+1 ),
                                         cv::Point( morph_size, morph_size ) );
    cv::morphologyEx( src, dst, cv::MORPH_TOPHAT , element );
    cv::namedWindow("Top Hat",1);
    imshow("Top Hat",dst);
}

//Black_Hat
void MainWindow::on_action_Black_Hat_triggered()
{
    int morph_size = 3;
    cv::Mat element = getStructuringElement( cv::MORPH_RECT,
                                         cv::Size( 2*morph_size + 1, 2*morph_size+1 ),
                                         cv::Point( morph_size, morph_size ) );
    cv::morphologyEx( src, dst, cv::MORPH_BLACKHAT , element );
    cv::namedWindow("Black Hat",1);
    imshow("Black Hat",dst);
}

//Discrete_Fourier_Transform
void MainWindow::on_action_Discrete_Fourier_Transform_triggered()
{
    if(src.channels() != 1)
    {
        cv::cvtColor(src,src,CV_RGB2GRAY);
    }

    cv::Mat padded;                            //expand input image to optimal size
    int m = cv::getOptimalDFTSize( src.rows );
    int n = cv::getOptimalDFTSize( src.cols ); // on the border add zero values
    cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    cv::dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    cv::split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    cv::magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    cv::Mat magI = planes[0];

    magI += cv::Scalar::all(1);                    // switch to logarithmic scale
    cv::log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    cv::namedWindow("Input Image",1);
    imshow("Input Image"       , src   );    // Show the result
    cv::namedWindow("Spectrum Magnitude",1);
    imshow("Spectrum Magnitude", magI);
}

//Sobel Edges
void MainWindow::on_action_Sobel_triggered()
{
    cv::Mat grad;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    cv::GaussianBlur( src, src, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    if(src.channels() != 1)
        cvtColor( src, src, CV_BGR2GRAY );
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    cv::Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    cv::Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    cv::namedWindow("Sobel Edges",1);
    imshow( "Sobel Edges", grad );
}

//Laplace Edges
void MainWindow::on_action_Laplace_triggered()
{
      int kernel_size = 3;
      int scale = 1;
      int delta = 0;
      int ddepth = CV_16S;

      /// Remove noise by blurring with a Gaussian filter
      cv::GaussianBlur( src, src, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );

      /// Convert the image to grayscale
      if(src.channels() != 1)
              cvtColor( src, src, CV_BGR2GRAY );

      /// Apply Laplace function
      cv::Mat abs_dst;

      cv::Laplacian( src, dst, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT );
      cv::convertScaleAbs( dst, abs_dst );

      /// Show what you got
      cv::namedWindow("Laplace",1);
      imshow( "Laplace", abs_dst );
}

//Canny
void MainWindow::on_action_Canny_triggered()
{
    ui->spinBox_Canny->show();
    ui->horizontalSlider_Canny->show();
    ui->okButton_Canny->show();
}
void MainWindow::on_horizontalSlider_Canny_valueChanged(int value)
{
    value = ui->horizontalSlider_Canny->value();
    cv::Mat tmp,dst;
    if(src.channels() != 1)
    {
        cv::cvtColor(src,tmp,CV_RGB2GRAY);
    }
    cv::Canny(tmp,dst,value,200);
    cv::namedWindow("Canny Edges",1);
    imshow("Canny Edges",dst);
}
void MainWindow::on_okButton_Canny_clicked()
{
    ui->spinBox_Canny->hide();
    ui->horizontalSlider_Canny->hide();
    ui->okButton_Canny->hide();
}

//Hough Lines
void MainWindow::on_action_Hough_Line_triggered()
{
    ui->spinBox_HoughLine->show();
    ui->horizontalSlider_HoughLine->show();
    ui->okButton_HoughLine->show();
}
void MainWindow::on_horizontalSlider_HoughLine_valueChanged(int value)
{
    value = ui->horizontalSlider_HoughLine->value();
    cv::Mat tmp;
    cv::Mat dst_add_line = src.clone();
    if(src.channels() != 1)
    {
        cv::cvtColor(src,tmp,CV_RGB2GRAY);
    }
    GaussianBlur( tmp, tmp, cv::Size( 9, 9 ), 0, 0 );
    cv::Canny(tmp, dst, 100, 200);
    cv::namedWindow("Canny",1);
    imshow("Canny",dst);
    cv::vector<cv::Vec4i> lines;
    cv::HoughLinesP( dst, lines, 1, CV_PI/180, value, 30, 10 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        cv::Vec4i l = lines[i];
        cv::line( dst_add_line, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 1, CV_AA);
    }
    cv::namedWindow("Detected Lines",1);
    imshow("Detected Lines", dst_add_line);
}
void MainWindow::on_okButton_HoughLine_clicked()
{
    ui->spinBox_HoughLine->hide();
    ui->horizontalSlider_HoughLine->hide();
    ui->okButton_HoughLine->hide();
}

//Hough Circles
void MainWindow::on_action_Hough_Circle_triggered()
{
    ui->spinBox_HoughCircle->show();
    ui->horizontalSlider_HougCircle->setMaximum(src.rows);
    ui->horizontalSlider_HougCircle->show();
    ui->okButton_HoughCircle->show();
}
void MainWindow::on_horizontalSlider_HougCircle_valueChanged(int value)
{
    value = ui->horizontalSlider_HougCircle->value();
    cv::Mat src_gray;
    cv::Mat src_tmp = src.clone();
    /// Convert it to gray
    cv::cvtColor( src_tmp, src_gray, CV_BGR2GRAY );

    /// Reduce the noise so we avoid false circle detection
    cv::GaussianBlur( src_gray, src_gray, cv::Size(9, 9), 2, 2 );

    cv::vector<cv::Vec3f> circles;

    /// Apply the Hough Transform to find the circles
    cv::HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, value, 0 );

    /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ )
    {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        cv::circle( src_tmp, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
          // circle outline
        cv::circle( src_tmp, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );
    }
    cv::namedWindow("Hough Circles",1);
    imshow( "Hough Circles", src_tmp );
}
void MainWindow::on_okButton_HoughCircle_clicked()
{
    ui->spinBox_HoughCircle->hide();
    ui->horizontalSlider_HougCircle->hide();
    ui->okButton_HoughCircle->hide();
}

//Histogram_Equalization
void MainWindow::on_action_Histogram_Equalization_triggered()
{
    cv::Mat src_tmp = src.clone();
    if(src.channels() != 1)
    {
        cvtColor( src_tmp, src_tmp, CV_BGR2GRAY );
    }
    equalizeHist( src_tmp, dst );
    cv::namedWindow( "Equalized Image", 1 );
    imshow( "Equalized Image", dst );
}

//Histogram_Calculation
void MainWindow::on_action_Histogram_Calculation_triggered()
{
      /// Separate the image in 3 places ( B, G and R )
      cv::vector<cv::Mat> bgr_planes;
      cv::split( src, bgr_planes );

      /// Establish the number of bins
      int histSize = 256;

      /// Set the ranges ( for B,G,R) )
      float range[] = { 0, 256 } ;
      const float* histRange = { range };

      bool uniform = true; bool accumulate = false;

      cv::Mat b_hist, g_hist, r_hist;

      /// Compute the histograms:
      cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
      cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
      cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

      // Draw the histograms for B, G and R
      int hist_w = 512; int hist_h = 400;
      int bin_w = cvRound( (double) hist_w/histSize );

      cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );

      /// Normalize the result to [ 0, histImage.rows ]
      cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
      cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
      cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

      /// Draw for each channel
      for( int i = 1; i < histSize; i++ )
      {
          cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                           cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                           cv::Scalar( 255, 0, 0), 2, 8, 0  );
          cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                           cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                           cv::Scalar( 0, 255, 0), 2, 8, 0  );
          cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                           cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                           cv::Scalar( 0, 0, 255), 2, 8, 0  );
      }
      cv::namedWindow("CalcHist", 1 );
      imshow("CalcHist", histImage );
}

//Back_Projection
void MainWindow::on_action_Back_Projection_triggered()
{
    ui->spinBox_BackProjection->show();
    ui->horizontalSlider_BackProjection->setMaximum(255);
    ui->horizontalSlider_BackProjection->show();
    ui->okButton_BackProjection->show();
}
void MainWindow::on_horizontalSlider_BackProjection_valueChanged(int bins)
{
    bins = ui->horizontalSlider_BackProjection->value();
    cv::Mat hsv,hue;
    cv::cvtColor( src, hsv, CV_BGR2HSV );
    hue.create( hsv.size(), hsv.depth() );
    int ch[] = { 0, 0 };
    cv::mixChannels( &hsv, 1, &hue, 1, ch, 1 );

    cv::MatND hist;
    int histSize = MAX( bins, 2 );
    float hue_range[] = { 0, 180 };
    const float* ranges = { hue_range };

    /// Get the Histogram and normalize it
    cv::calcHist( &hue, 1, 0, cv::Mat(), hist, 1, &histSize, &ranges, true, false );
    cv::normalize( hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat() );

    /// Get Backprojection
    cv::MatND backproj;
    calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );

    /// Draw the backproj
    cv::namedWindow("Back Projection",1);
    imshow( "Back Projection", backproj );

    /// Draw the histogram
    int w = 400; int h = 400;
    int bin_w = cvRound( (double) w / histSize );
    cv::Mat histImg = cv::Mat::zeros( w, h, CV_8UC3 );

    for( int i = 0; i < bins; i ++ )
       { cv::rectangle( histImg, cv::Point( i*bin_w, h ), cv::Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ), cv::Scalar( 0, 0, 255 ), -1 ); }

    cv::namedWindow("Histogram",1);
    imshow( "Histogram", histImg );
}
void MainWindow::on_okButton_BackProjection_clicked()
{
    ui->spinBox_BackProjection->hide();
    ui->horizontalSlider_BackProjection->hide();
    ui->okButton_BackProjection->hide();
}

//Finding_Contours
void MainWindow::on_action_Finding_Contours_triggered()
{
    ui->spinBox_FindContours->show();
    ui->horizontalSlider_FindContours->show();
    ui->okButton_FindContours->show();
}
void MainWindow::on_horizontalSlider_FindContours_valueChanged(int thresh)
{
    thresh = ui->horizontalSlider_FindContours->value();
    cv::RNG rng(12345);
    cv::Mat src_gray;
    if(src.channels()!=1)
    {
        cv::cvtColor( src, src_gray, CV_BGR2GRAY );
    }
    cv::blur( src_gray, src_gray, cv::Size(3,3) );

    cv::Mat canny_output;
    cv::vector<cv::vector<cv::Point> > contours;
    cv::vector<cv::Vec4i> hierarchy;

     // Detect edges using canny
    cv::Canny( src_gray, canny_output, thresh, thresh*2, 3 );
    /// Find contours
    cv::findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    /// Draw contours
    cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< (int)contours.size(); i++ )
    {
       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       cv::drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
    }

    /// Show in a window
    cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
}
void MainWindow::on_okButton_FindContours_clicked()
{
    ui->spinBox_FindContours->hide();
    ui->horizontalSlider_FindContours->hide();
    ui->okButton_FindContours->hide();
}

//Convex_Hull
void MainWindow::on_action_Convex_Hull_triggered()
{
    ui->spinBox_Hull->show();
    ui->horizontalSlider_Hull->show();
    ui->okButton_Hull->show();
}
void MainWindow::on_horizontalSlider_Hull_valueChanged(int thresh)
{
    thresh = ui->horizontalSlider_Hull->value();
    cv::Mat src_gray;
    cv::RNG rng(12345);

    if(src.channels()!=1)
    {
        cv::cvtColor( src, src_gray, CV_BGR2GRAY );
    }
    cv::blur( src_gray, src_gray, cv::Size(3,3) );

    cv::Mat threshold_output;
    cv::vector<cv::vector<cv::Point> > contours;
    cv::vector<cv::Vec4i> hierarchy;

    /// Detect edges using Threshold
    cv::threshold( src_gray, threshold_output, thresh, 255, cv::THRESH_BINARY );

    /// Find contours
    cv::findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

    /// Find the convex hull object for each contour
    cv::vector<cv::vector<cv::Point> > hull( contours.size() );
    for( int i = 0; i < (int)contours.size(); i++ )
       {  cv::convexHull( cv::Mat(contours[i]), hull[i], false ); }

    /// Draw contours + hull results
    cv::Mat drawing = cv::Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i<(int)contours.size(); i++ )
    {
       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       cv::drawContours( drawing, contours, i, color, 1, 8, cv::vector<cv::Vec4i>(), 0, cv::Point() );
       cv::drawContours( drawing, hull, i, color, 1, 8, cv::vector<cv::Vec4i>(), 0, cv::Point() );
    }

    /// Show in a window
    cv::namedWindow( "Hull", CV_WINDOW_AUTOSIZE );
    imshow( "Hull", drawing );
}
void MainWindow::on_okButton_Hull_clicked()
{
    ui->spinBox_Hull->hide();
    ui->horizontalSlider_Hull->hide();
    ui->okButton_Hull->hide();
}

//Creating_Bounding_For_Contours
void MainWindow::on_action_Creating_Bounding_For_Contours_triggered()
{
    ui->spinBox_CreatBounding->show();
    ui->horizontalSlider_CreatBounding->show();
    ui->okButton_CreatBounding->show();
}
void MainWindow::on_horizontalSlider_CreatBounding_valueChanged(int thresh)
{
    thresh = ui->horizontalSlider_CreatBounding->value();
    cv::Mat src_gray;
    cv::RNG rng(12345);

    if(src.channels()!=1)
    {
        cv::cvtColor( src, src_gray, CV_BGR2GRAY );
    }
    cv::blur( src_gray, src_gray, cv::Size(3,3) );
    cv::Mat threshold_output;
    cv::vector<cv::vector<cv::Point> > contours;
    cv::vector<cv::Vec4i> hierarchy;

    /// Detect edges using Threshold
    cv::threshold( src_gray, threshold_output, thresh, 255, cv::THRESH_BINARY );
    /// Find contours
    cv::findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
    /// Approximate contours to polygons + get bounding rects and circles
    cv::vector<cv::vector<cv::Point> > contours_poly( contours.size() );
    cv::vector<cv::Rect> boundRect( contours.size() );
    cv::vector<cv::Point2f>center( contours.size() );
    cv::vector<float>radius( contours.size() );
    for( int i = 0; i < (int)contours.size(); i++ )
    {
        cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );
        cv::minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
    }
    /// Draw polygonal contour + bonding rects + circles
    cv::Mat drawing = cv::Mat::zeros( threshold_output.size(), CV_8UC3 );
    for( int i = 0; i< (int)contours.size(); i++ )
    {
       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       cv::drawContours( drawing, contours_poly, i, color, 1, 8, cv::vector<cv::Vec4i>(), 0, cv::Point() );
       cv::rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
       cv::circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    }
    /// Show in a window
    cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );
}
void MainWindow::on_okButton_CreatBounding_clicked()
{
    ui->spinBox_CreatBounding->hide();
    ui->horizontalSlider_CreatBounding->hide();
    ui->okButton_CreatBounding->hide();
}

//Image_Moments
void MainWindow::on_action_Image_Moments_triggered()
{
    ui->spinBox_Image_Moments->show();
    ui->horizontalSlider_Image_Moments->show();
    ui->okButton_Image_Moments->show();
}
void MainWindow::on_horizontalSlider_Image_Moments_valueChanged(int thresh)
{
    thresh = ui->horizontalSlider_Image_Moments->value();
    cv::Mat src_gray;
    cv::RNG rng(12345);

    if(src.channels()!=1)
    {
        cv::cvtColor( src, src_gray, CV_BGR2GRAY );
    }
    cv::blur( src_gray, src_gray, cv::Size(3,3) );
    cv::Mat canny_output;
    cv::vector<cv::vector<cv::Point> > contours;
    cv::vector<cv::Vec4i> hierarchy;

      /// Detect edges using canny
    cv::Canny( src_gray, canny_output, thresh, thresh*2, 3 );
      /// Find contours
    cv::findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

      /// Get the moments
    cv::vector<cv::Moments> mu(contours.size() );
    for( int i = 0; i < (int)contours.size(); i++ )
         { mu[i] = moments( contours[i], false ); }

      ///  Get the mass centers:
    cv::vector<cv::Point2f> mc( contours.size() );
      for( int i = 0; i < (int)contours.size(); i++ )
         { mc[i] = cv::Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

      /// Draw contours
    cv::Mat drawing = cv::Mat::zeros( canny_output.size(), CV_8UC3 );
    for( int i = 0; i< (int)contours.size(); i++ )
    {
       cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       cv::drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
       cv::circle( drawing, mc[i], 4, color, -1, 8, 0 );
    }

    /// Show in a window
    cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    imshow( "Contours", drawing );

    /// Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for( int i = 0; i< (int)contours.size(); i++ )
    {
      //printf(" * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f \n", i, mu[i].m00, contourArea(contours[i]), arcLength( contours[i], true ) );
      cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
      cv::drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
      cv::circle( drawing, mc[i], 4, color, -1, 8, 0 );
    }
}
void MainWindow::on_okButton_Image_Moments_clicked()
{
    ui->spinBox_Image_Moments->hide();
    ui->horizontalSlider_Image_Moments->hide();
    ui->okButton_Image_Moments->hide();
}

//Point_Polygon_Test
void MainWindow::on_action_Point_Polygon_Test_triggered()
{
    /// Create an image
    const int r = 100;
    cv::Mat src = cv::Mat::zeros( cv::Size( 4*r, 4*r ), CV_8UC1 );

    /// Create a sequence of points to make a contour:
    cv::vector<cv::Point2f> vert(6);

    vert[0] = cv::Point( 1.5*r, 1.34*r );
    vert[1] = cv::Point( 1*r, 2*r );
    vert[2] = cv::Point( 1.5*r, 2.866*r );
    vert[3] = cv::Point( 2.5*r, 2.866*r );
    vert[4] = cv::Point( 3*r, 2*r );
    vert[5] = cv::Point( 2.5*r, 1.34*r );

    /// Draw it in src
    for( int j = 0; j < 6; j++ )
       { cv::line( src, vert[j],  vert[(j+1)%6], cv::Scalar( 255 ), 3, 8 ); }

    /// Get the contours
    cv::vector<cv::vector<cv::Point> > contours;
    cv::vector<cv::Vec4i> hierarchy;
    cv::Mat src_copy = src.clone();

    cv::findContours( src_copy, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    /// Calculate the distances to the contour
    cv::Mat raw_dist( src.size(), CV_32FC1 );

    for( int j = 0; j < src.rows; j++ )
       { for( int i = 0; i < src.cols; i++ )
            { raw_dist.at<float>(j,i) = cv::pointPolygonTest( contours[0], cv::Point2f(i,j), true ); }
       }

    double minVal; double maxVal;
    cv::minMaxLoc( raw_dist, &minVal, &maxVal, 0, 0, cv::Mat() );
    minVal = abs(minVal); maxVal = abs(maxVal);

    /// Depicting the  distances graphically
    cv::Mat drawing = cv::Mat::zeros( src.size(), CV_8UC3 );

    for( int j = 0; j < src.rows; j++ )
       { for( int i = 0; i < src.cols; i++ )
            {
              if( raw_dist.at<float>(j,i) < 0 )
                { drawing.at<cv::Vec3b>(j,i)[0] = 255 - (int) abs(raw_dist.at<float>(j,i))*255/minVal; }
              else if( raw_dist.at<float>(j,i) > 0 )
                { drawing.at<cv::Vec3b>(j,i)[2] = 255 - (int) raw_dist.at<float>(j,i)*255/maxVal; }
              else
                { drawing.at<cv::Vec3b>(j,i)[0] = 255; drawing.at<cv::Vec3b>(j,i)[1] = 255; drawing.at<cv::Vec3b>(j,i)[2] = 255; }
            }
       }

    /// Create Window and show your results
    cv::namedWindow( "Distance", CV_WINDOW_AUTOSIZE );
    imshow( "Distance", drawing );
}

//Harris_corners
void MainWindow::on_action_Harris_corner_triggered()
{
    ui->spinBox_Harris_Corners->show();
    ui->horizontalSlider_Harris_Corners->show();
    ui->horizontalSlider_Harris_Corners->setValue(200);
    ui->okButton_Harris_Corners->show();
}
void MainWindow::on_horizontalSlider_Harris_Corners_valueChanged(int thresh)
{
    thresh = ui->horizontalSlider_Harris_Corners->value();
    cv::Mat src_gray;
    cv::Mat tmp;
    if(src.channels()!=1)
    {
        cv::cvtColor( src, src_gray, CV_BGR2GRAY );
    }
    cv::Mat dst_norm, dst_norm_scaled;
    tmp = cv::Mat::zeros( src.size(), CV_32FC1 );

    /// Detector parameters
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    /// Detecting corners
    cv::cornerHarris( src_gray, tmp, blockSize, apertureSize, k, cv::BORDER_DEFAULT );

    /// Normalizing
    cv::normalize( tmp, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );

    /// Drawing a circle around corners
    for( int j = 0; j < dst_norm.rows ; j++ )
    {
        for( int i = 0; i < dst_norm.cols; i++ )
        {
            if( (int) dst_norm.at<float>(j,i) > thresh )
            {
                cv::circle( dst_norm_scaled, cv::Point( i, j ), 5,  cv::Scalar(0), 2, 8, 0 );
            }
        }
     }
    cv::namedWindow( "Harris Corners", CV_WINDOW_AUTOSIZE );
    imshow( "Harris Corners", dst_norm_scaled );
}
void MainWindow::on_okButton_Harris_Corners_clicked()
{
    ui->spinBox_Harris_Corners->hide();
    ui->horizontalSlider_Harris_Corners->hide();
    ui->okButton_Harris_Corners->hide();
}

//Shi_Tomasi_corners
void MainWindow::on_action_Shi_Tomasi_corner_triggered()
{
    ui->spinBox_ShiTomasi_Corners->show();
    ui->spinBox_ShiTomasi_Corners->setMaximum(100);
    ui->horizontalSlider_ShiTomasi_Corners->setValue(23);
    ui->horizontalSlider_ShiTomasi_Corners->setMaximum(100);
    ui->horizontalSlider_ShiTomasi_Corners->show();
    ui->okButton_ShiTomasi_Corners->show();
}
void MainWindow::on_horizontalSlider_ShiTomasi_Corners_valueChanged(int maxCorners)
{
    maxCorners = ui->horizontalSlider_ShiTomasi_Corners->value();
    cv::Mat src_gray;
    cv::RNG rng(12345);
    if(src.channels()!=1)
    {
        cv::cvtColor( src, src_gray, CV_BGR2GRAY );
    }
    cv::vector<cv::Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    /// Copy the source image
    cv::Mat copy;
    copy = src.clone();

    /// Apply corner detection
    cv::goodFeaturesToTrack( src_gray,
                   corners,
                   maxCorners,
                   qualityLevel,
                   minDistance,
                   cv::Mat(),
                   blockSize,
                   useHarrisDetector,
                   k );

    int r = 4;
    for( int i = 0; i < (int)corners.size(); i++ )
         { cv::circle( copy, corners[i], r, cv::Scalar(rng.uniform(0,255), rng.uniform(0,255),
                  rng.uniform(0,255)), -1, 8, 0 ); }

      /// Show what you got
      cv::namedWindow( "Shi-Tomasi Corners", CV_WINDOW_AUTOSIZE );
      imshow( "Shi-Tomasi Corners", copy );
}
void MainWindow::on_okButton_ShiTomasi_Corners_clicked()
{
    ui->spinBox_ShiTomasi_Corners->hide();
    ui->horizontalSlider_ShiTomasi_Corners->hide();
    ui->okButton_ShiTomasi_Corners->hide();
}

//Subpixeles_corners
void MainWindow::on_action_Subpixeles_corner_triggered()
{
    ui->spinBox_Subpixeles_Corners->setValue(10);
    ui->spinBox_Subpixeles_Corners->setMaximum(25);
    ui->spinBox_Subpixeles_Corners->show();
    ui->horizontalSlider_Subpixeles_Corners->setValue(10);
    ui->horizontalSlider_Subpixeles_Corners->setMaximum(25);
    ui->horizontalSlider_Subpixeles_Corners->show();
    ui->okButton_Subpixeles_Corners->show();
}
void MainWindow::on_horizontalSlider_Subpixeles_Corners_valueChanged(int maxCorners)
{
    maxCorners = ui->horizontalSlider_Subpixeles_Corners->value();
    cv::Mat src_gray;
    cv::RNG rng(12345);
    if(src.channels()!=1)
    {
        cv::cvtColor( src, src_gray, CV_BGR2GRAY );
    }
    cv::vector<cv::Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    /// Copy the source image
    cv::Mat copy;
    copy = src.clone();

    /// Apply corner detection
    cv::goodFeaturesToTrack( src_gray,
                          corners,
                          maxCorners,
                          qualityLevel,
                          minDistance,
                          cv::Mat(),
                          blockSize,
                          useHarrisDetector,
                          k );


     /// Draw corners detected
     //cout<<"** Number of corners detected: "<<corners.size()<<endl;
     int r = 4;
     for( int i = 0; i < (int)corners.size(); i++ )
        { cv::circle( copy, corners[i], r, cv::Scalar(rng.uniform(0,255), rng.uniform(0,255),
                                                    rng.uniform(0,255)), -1, 8, 0 ); }

     /// Show what you got
     cv::namedWindow( "Subpixeles Corners", CV_WINDOW_AUTOSIZE );
     imshow( "Subpixeles Corners", copy );

     /*
     /// Set the neeed parameters to find the refined corners
     cv::Size winSize = cv::Size( 5, 5 );
     cv::Size zeroZone = cv::Size( -1, -1 );
     cv::TermCriteria criteria = cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001 );

     /// Calculate the refined corner locations
     cv::cornerSubPix( src_gray, corners, winSize, zeroZone, criteria );

     /// Write them down
     for( int i = 0; i < corners.size(); i++ )
     {
         //cout<<" -- Refined Corner ["<<i<<"]  ("<<corners[i].x<<","<<corners[i].y<<")"<<endl;
     }
    */
}
void MainWindow::on_okButton_Subpixeles_Corners_clicked()
{
    ui->spinBox_Subpixeles_Corners->hide();
    ui->horizontalSlider_Subpixeles_Corners->hide();
    ui->okButton_Subpixeles_Corners->hide();
}

//Feature_Detection
void MainWindow::on_action_Feature_Detection_triggered()
{
    cv::Mat img = src.clone();
    int minHessian = 400;
    cv::SurfFeatureDetector detector( minHessian );
    cv::vector<cv::KeyPoint> keypoints;
    detector.detect( img, keypoints );
    cv::Mat img_keypoints;
    cv::drawKeypoints( img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

    imshow("Keypoints", img_keypoints);
}

//Feature_Description
void MainWindow::on_action_Feature_Description_triggered()
{
    ui->ImageLabel_2->show();
    ui->okButton_Image2->show();

    QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Please Open The First Image!"));
    fileName = QFileDialog::getOpenFileName(this,tr("ImageProcess"),".",tr("Image Files(*.jpg *.png *.bmp)"));
    std::string str;
    str = fileName.toStdString();
    std::cout<<str<<std::endl;
    src = cv::imread(fileName.toStdString());
    image = new QImage(fileName);
    show_Image(image);

    QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Please Open The Secend Image!"));
    fileName = QFileDialog::getOpenFileName(this,tr("ImageProcess"),".",tr("Image Files(*.jpg *.png *.bmp)"));
    str = fileName.toStdString();
    std::cout<<str<<std::endl;
    dst = cv::imread(fileName.toStdString());
    image = new QImage(fileName);
    show_Image_2(image);

    cv::Mat img_1 = src.clone();
    cv::Mat img_2 = dst.clone();

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 237;

    cv::SurfFeatureDetector detector( minHessian );

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    cv::SurfDescriptorExtractor extractor;

    cv::Mat descriptors_1, descriptors_2;

    extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors with a brute force matcher
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector< cv::DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    //-- Draw matches
    cv::Mat img_matches;
    cv::drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

    //-- Show detected matches
    imshow("Matches", img_matches );
}

//Feature_Matching_FLANN
void MainWindow::on_action_Feature_Matching_FLANN_triggered()
{
    if(ui->ImageLabel_2->isHidden())
    {
        ui->ImageLabel_2->show();

        QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Please Open The First Image!"));
        fileName = QFileDialog::getOpenFileName(this,tr("ImageProcess"),".",tr("Image Files(*.jpg *.png *.bmp)"));
        std::string str;
        str = fileName.toStdString();
        std::cout<<str<<std::endl;
        src = cv::imread(fileName.toStdString());
        image = new QImage(fileName);
        show_Image(image);

        QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Please Open The Secend Image!"));
        fileName = QFileDialog::getOpenFileName(this,tr("ImageProcess"),".",tr("Image Files(*.jpg *.png *.bmp)"));
        str = fileName.toStdString();
        std::cout<<str<<std::endl;
        dst = cv::imread(fileName.toStdString());
        image = new QImage(fileName);
        show_Image_2(image);
    }

    cv::Mat img_1 = src.clone();
    cv::Mat img_2 = dst.clone();

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    cv::SurfFeatureDetector detector( minHessian );
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    cv::SurfDescriptorExtractor extractor;
    cv::Mat descriptors_1, descriptors_2;

    extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector< cv::DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );
    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < (int)descriptors_1.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector< cv::DMatch > good_matches;

    for( int i = 0; i < (int)descriptors_1.rows; i++ )
    { if( matches[i].distance <= cv::max(2*min_dist, 0.02) )
        { good_matches.push_back( matches[i]); }
    }

      //-- Draw only "good" matches
    cv::Mat img_matches;
    cv::drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                   good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                   cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

      //-- Show detected matches
    imshow( "Good Matches", img_matches );

    /*for( int i = 0; i < (int)good_matches.size(); i++ )
    {
        printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx );
    }*/
}
void MainWindow::on_okButton_Image2_clicked()
{
    ui->ImageLabel_2->hide();
    ui->okButton_Image2->hide();
}

//Find_Object
void MainWindow::on_action_Find_Object_triggered()
{
    if(ui->ImageLabel_2->isHidden())
    {
        ui->ImageLabel_2->show();
        ui->okButton_Image2->show();

        QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Please Open Target Image!"));
        fileName = QFileDialog::getOpenFileName(this,tr("ImageProcess"),".",tr("Image Files(*.jpg *.png *.bmp)"));
        std::string str;
        str = fileName.toStdString();
        std::cout<<str<<std::endl;
        src = cv::imread(fileName.toStdString());
        image = new QImage(fileName);
        show_Image(image);

        QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Please Open Background Image!"));
        fileName = QFileDialog::getOpenFileName(this,tr("ImageProcess"),".",tr("Image Files(*.jpg *.png *.bmp)"));
        str = fileName.toStdString();
        std::cout<<str<<std::endl;
        dst = cv::imread(fileName.toStdString());
        image = new QImage(fileName);
        show_Image_2(image);
    }
    cv::Mat img_object = src.clone();
    cv::Mat img_scene = dst.clone();
    int minHessian = 400;
    cv::SurfFeatureDetector detector( minHessian );

    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
    detector.detect( img_object, keypoints_object );
    detector.detect( img_scene, keypoints_scene );

    //-- Step 2: Calculate descriptors (feature vectors)
    cv::SurfDescriptorExtractor extractor;

    cv::Mat descriptors_object, descriptors_scene;

    extractor.compute( img_object, keypoints_object, descriptors_object );
    extractor.compute( img_scene, keypoints_scene, descriptors_scene );

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector< cv::DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < (int)descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< cv::DMatch > good_matches;

    for( int i = 0; i < (int)descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
     { good_matches.push_back( matches[i]); }
    }

    cv::Mat img_matches;
    cv::drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               cv::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for( int i = 0; i < (int)good_matches.size(); i++ )
    {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    cv::Mat H = findHomography( obj, scene, CV_RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    std::vector<cv::Point2f> scene_corners(4);

    cv::perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv::line( img_matches, scene_corners[0] + cv::Point2f( img_object.cols, 0), scene_corners[1] + cv::Point2f( img_object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
    cv::line( img_matches, scene_corners[1] + cv::Point2f( img_object.cols, 0), scene_corners[2] + cv::Point2f( img_object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
    cv::line( img_matches, scene_corners[2] + cv::Point2f( img_object.cols, 0), scene_corners[3] + cv::Point2f( img_object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
    cv::line( img_matches, scene_corners[3] + cv::Point2f( img_object.cols, 0), scene_corners[0] + cv::Point2f( img_object.cols, 0), cv::Scalar( 0, 255, 0), 4 );

    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches );
}

//Person Face Detection
void MainWindow::detectAndDisplay(cv::Mat frame)
{
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

    for (int i = 0; i < (int)faces.size(); i++)
    {
        cv::Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
        cv::ellipse(frame, center, cv::Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);

        cv::Mat faceROI = frame_gray(faces[i]);
        std::vector<cv::Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

        for (int j = 0; j < (int)eyes.size(); j++)
        {
            cv::Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            cv::circle(frame, center, radius, cv::Scalar(255, 0, 0), 4, 8, 0);
        }
    }
    //-- Show what you got
    imshow("Face Detection", frame);
}
void MainWindow::on_action_Person_FaceDetection_triggered()
{
    CvCapture* capture;
    cv::Mat frame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Open 'haarcascade_frontalface_alt.xml' File Faild!")); };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Open 'haarcascade_eye_tree_eyeglasses.xml' File Faild!")); };

    //-- 2. Read the video stream
    capture = cvCaptureFromCAM(0);
    if( capture )
    {
        QMessageBox::information(this, QString("ImagePro(Jack)"), QString("Press 'q' to end this programe!"));
        while( true )
        {
            frame = cvQueryFrame(capture);

            //-- 3. Apply the classifier to the frame
            if( !frame.empty() )
            {
                detectAndDisplay( frame );
            }
            else
            { printf(" --(!) No captured frame -- Break!"); break; }

            int c = cv::waitKey(3);
            if( (char)c == 'q' ) { break; }
         }
    }
    cvReleaseCapture(&capture);
}

//About
void MainWindow::on_actionImagePro_About_triggered()
{
    QMessageBox::about(this,tr("ImagePro 1.0"),
                       tr("<h2>ImagePro 1.0</h2>"
                          "<p>Copyright &copy; 2017."
                          "<p>ImagePro is a small application to process Image."));
}
//Qt About
void MainWindow::on_actionQt_About_triggered()
{
    QMessageBox::aboutQt(this);
}
