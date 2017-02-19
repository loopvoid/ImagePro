#-------------------------------------------------
#
# Project created by QtCreator 2017-02-17T14:28:53
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ImageProcessBasedCv_1
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

INCLUDEPATH+=D:\Opencv\build\include\
D:\Opencv\build\include\opencv\
D:\Opencv\build\include\opencv2\

LIBS+=E:\opencvLib\lib\libopencv_calib3d2413.dll.a\
E:\opencvLib\lib\libopencv_contrib2413.dll.a\
E:\opencvLib\lib\libopencv_core2413.dll.a\
E:\opencvLib\lib\libopencv_features2d2413.dll.a\
E:\opencvLib\lib\libopencv_flann2413.dll.a\
E:\opencvLib\lib\libopencv_gpu2413.dll.a\
E:\opencvLib\lib\libopencv_highgui2413.dll.a\
E:\opencvLib\lib\libopencv_imgproc2413.dll.a\
E:\opencvLib\lib\libopencv_legacy2413.dll.a\
E:\opencvLib\lib\libopencv_ml2413.dll.a\
E:\opencvLib\lib\libopencv_objdetect2413.dll.a\
E:\opencvLib\lib\libopencv_nonfree2413.dll.a\
E:\opencvLib\lib\libopencv_video2413.dll.a\
E:\opencvLib\lib\libopencv_stitching2413.dll.a\
E:\opencvLib\lib\libopencv_ocl2413.dll.a\
E:\opencvLib\lib\libopencv_photo2413.dll.a\
E:\opencvLib\lib\libopencv_videostab2413.dll.a

RESOURCES += \
    icon.qrc
