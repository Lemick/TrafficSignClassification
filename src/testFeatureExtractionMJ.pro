TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

INCLUDEPATH += C:/dev/opencv/build/include

LIBS += -LC:\\dev\\opencv\\build\\x86\\vc11\\bin \
    -LC:\\dev\\opencv\\build\\x86\\vc11\\lib \
    -lopencv_core2410 \
    -lopencv_highgui2410 \
    -lopencv_imgproc2410 \
    -lopencv_features2d2410 \
    -lopencv_calib3d2410 \
    -lopencv_objdetect2410 \
    -lopencv_ml2410 \
    -lopencv_nonfree2410 \
    -lopencv_flann2410 \


