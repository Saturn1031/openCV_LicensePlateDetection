import cv2
import sys
import numpy as np
import pytesseract
from PIL import Image
import re


def preprocessing_ex(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환하여 저장
    blur = cv2.blur(gray, (5, 5))
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, 3)
    _, b_img = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 17), np.uint8)
    morph = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    return morph


def find_candidates_ex(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.minAreaRect(c) for c in contours]  # 외곽 최소 영역
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates


def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h/ w if h > w else w/ h       # 종횡비 계산

    # print(h * w)
    # print(aspect)
    chk1 = 3000 < (h * w) < 15000          # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 8.0       # 번호판 종횡비 조건

    #print(w,h)
    return (chk1 and chk2)


def preprocessing_01(image):
    blurImg = cv2.bilateralFilter(image, -1, 200, 5)

    grayImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)
    gray3channel = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)
    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # 수직 필터
    prewitt_grad_x = cv2.filter2D(gray3channel, -1, prewitt_filter_x)  # 수직 에지
    edgeImg = cv2.convertScaleAbs(prewitt_grad_x)  # 양수로 변환

    ret, img_binaryB = cv2.threshold(edgeImg, 200, 255, cv2.THRESH_BINARY)
    threshImg = img_binaryB

    se1 = np.uint8([[0, 0, 0, 0, 0],  # 구조 요소
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0]])
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k = 8  # 반복 횟수
    morphologyImg = cv2.erode(cv2.dilate(threshImg, se1, iterations=k), se2, iterations=k)  # 닫기
    morphologyImg = cv2.cvtColor(morphologyImg, cv2.COLOR_BGR2GRAY)

    return morphologyImg


def preprocessing_02(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환하여 저장
    blur = cv2.blur(gray, (5, 5))
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, 3)
    _, b_img = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 17), np.uint8)
    morph = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    # se1 = np.uint8([[1, 1, 1, 1, 1],  # 구조 요소
    #                 [1, 1, 1, 1, 1],
    #                 [1, 1, 1, 1, 1],
    #                 [1, 1, 1, 1, 1],
    #                 [1, 1, 1, 1, 1]])
    se1 = np.ones((5, 9), np.uint8)

    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, se1, iterations=3)
    # morph = cv2.erode(morph, kernel, iterations=2)

    se2 = np.ones((6, 5), np.uint8)
    morph = cv2.dilate(morph, se2, iterations=5)

    # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=3)

    return morph


def find_candidates_01(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.minAreaRect(c) for c in contours]  # 외곽 최소 영역

    # 이미지 크기 가져오기
    height, width = image.shape[:2]

    expanded_rects = []
    for rect in rects:
        center, size, angle = rect

        # 확장 계수
        expansion_factor = 1.0
        max_expansion = 1.1  # 최대 확장 제한
        step = 0.05  # 점진적 확장 단계

        while expansion_factor <= max_expansion:
            print(expansion_factor)

            # 사각형 확장
            expanded_size = (
                size[0] * 1,
                size[1] * expansion_factor
            )

            # 확장된 사각형의 4개 꼭짓점 계산
            rect_box = cv2.boxPoints((center, expanded_size, angle))
            rect_box = np.int32(rect_box)

            # 모든 꼭짓점이 이미지 내부에 있는지 확인
            if (np.all(rect_box[:, 0] >= 0) and
                    np.all(rect_box[:, 0] < width) and
                    np.all(rect_box[:, 1] >= 0) and
                    np.all(rect_box[:, 1] < height)):

                # 다음 반복을 위해 확장 계수 증가
                expansion_factor += step
            else:
                # 직전 확장 상태로 되돌리기
                expanded_size = (
                    size[0] * 1,
                    size[1] * (expansion_factor - step)
                )
                expanded_rects.append((center, expanded_size, angle))
                break

    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in expanded_rects if verify_aspect_size(size)]

    return candidates


def getWarpPerspectiveRectImg(candidate):
    src_points = np.float32(cv2.boxPoints(candidate))

    # 왼쪽이 내려간 직사각형
    if src_points[1][0] > src_points[3][0]:
        dst_points = np.float32([(0, 0),
                                 (np.sqrt((src_points[0][0] - src_points[1][0]) ** 2 + (
                                             src_points[0][1] - src_points[1][1]) ** 2), 0),
                                 (np.sqrt((src_points[0][0] - src_points[1][0]) ** 2 + (
                                             src_points[0][1] - src_points[1][1]) ** 2), np.sqrt(
                                     (src_points[1][0] - src_points[2][0]) ** 2 + (
                                                 src_points[1][1] - src_points[2][1]) ** 2)),
                                 (0, np.sqrt((src_points[1][0] - src_points[2][0]) ** 2 + (
                                             src_points[1][1] - src_points[2][1]) ** 2))])

        perspect_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        rotatedRect = cv2.warpPerspective(img, perspect_mat, np.int32((np.sqrt(
            (src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2), np.sqrt(
            (src_points[1][0] - src_points[2][0]) ** 2 + (src_points[1][1] - src_points[2][1]) ** 2))), cv2.INTER_CUBIC)

        # cv2.imshow('rotated-left_' + str(candidateNum), rotatedRect)

    # 오른쪽이 내려간 직사각형
    else:
        dst_points = np.float32(
            [(0, np.sqrt((src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2)),
             (0, 0),
             (np.sqrt((src_points[1][0] - src_points[2][0]) ** 2 + (src_points[1][1] - src_points[2][1]) ** 2), 0),
             (np.sqrt((src_points[1][0] - src_points[2][0]) ** 2 + (src_points[1][1] - src_points[2][1]) ** 2),
              np.sqrt((src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2))])

        perspect_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        rotatedRect = cv2.warpPerspective(img, perspect_mat, np.int32((np.sqrt(
            (src_points[1][0] - src_points[2][0]) ** 2 + (src_points[1][1] - src_points[2][1]) ** 2), np.sqrt(
            (src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2))), cv2.INTER_CUBIC)

        # cv2.imshow('rotated-right_' + str(candidateNum), rotatedRect)

    return rotatedRect


def getRotationMatrix2DRectImg(candidate):
    src_points = np.float32(cv2.boxPoints(candidate))

    # 왼쪽이 내려간 직사각형
    if src_points[1][0] > src_points[3][0]:
        # 회전
        affine_matrix = cv2.getRotationMatrix2D(np.asarray(np.float32(candidate[0])), np.float32(candidate[2]) - 90, 1)
        rotatedRect = cv2.warpAffine(img, affine_matrix, (img.shape[1], img.shape[0]))

        # cv2.imshow('rotated-left_' + str(candidateNum), rotatedRect)

    # 오른쪽이 내려간 직사각형
    else:
        # 회전
        affine_matrix = cv2.getRotationMatrix2D(np.asarray(np.float32(candidate[0])), np.float32(candidate[2]), 1)
        rotatedRect = cv2.warpAffine(img, affine_matrix, (img.shape[1], img.shape[0]))

        # cv2.imshow('rotated-right_' + str(candidateNum), rotatedRect)

    return rotatedRect


car_no = str(input("자동차 영상 번호 (00~09): "))
img = cv2.imread('cars/'+car_no+'.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
cv2.imshow('original',img)


# 1 전처리 단계 (hw2-2)
preprocessed = preprocessing_02(img)

cv2.imshow('plate candidate 0', preprocessed)
# cv2.imwrite('hw2_2morph.png', morph)


# 2 번호판 후보 영역 검출 (hw3-2)
candidates = find_candidates_ex(preprocessed)

img2 = img.copy()
for candidate in candidates:  # 후보 영역 표시
    pts = np.int32(cv2.boxPoints(candidate))
    cv2.polylines(img2, [pts], True, (0, 225, 255), 3)

cv2.imshow('plate candidate 1', img2)
# cv2.imwrite('hw3_2candidates.png', img)


# 3 각 후보마다 warp 변환으로 회전하여 번호판 인식
candidateNum = 0
for candidate in candidates:
    rotatedRect = getWarpPerspectiveRectImg(candidate)
    # rotatedRect = getRotationMatrix2DRectImg(candidate)

    grayRotatedRect = cv2.cvtColor(rotatedRect, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('grayRotatedRect'+str(candidateNum), grayRotatedRect)

    # (thresh, license_plate) = cv2.threshold(grayRotatedRect, 127, 255, cv2.THRESH_BINARY)

    # bilateralImg = cv2.bilateralFilter(grayRotatedRect, 11, 17, 17)
    # bilateralImg = cv2.bilateralFilter(grayRotatedRect, -1, 10, 10)
    # sharpen = np.array([[-1.0, -1.0, -1.0],
    #                     [-1.0, 9.0, -1.0],
    #                     [-1.0, -1.0, -1.0]])
    # sharpen = np.array([[-1.0, -1.0, -1.0, -1.0, -1.0],
    #                     [-1.0, -1.0, -1.0, -1.0, -1.0],
    #                     [-1.0, -1.0, 25.0, -1.0, -1.0],
    #                     [-1.0, -1.0, -1.0, -1.0, -1.0],
    #                     [-1.0, -1.0, -1.0, -1.0, -1.0]])
    # sharpenImg = cv2.filter2D(bilateralImg, -1, sharpen)

    # equalizeHist_plate = cv2.equalizeHist(grayRotatedRect)
    # cv2.imshow('equalizeHist' + str(candidateNum), equalizeHist_plate)

    bilateralImg = cv2.bilateralFilter(grayRotatedRect, 11, 17, 17)

    multi_license_plate = cv2.multiply(bilateralImg, 1.5)
    cv2.imshow('multi_license_plate' + str(candidateNum), multi_license_plate)

    (thresh, TOZERO_license_plate) = cv2.threshold(multi_license_plate, 130, 255, cv2.THRESH_TOZERO)
    cv2.imshow('TOZERO_license_plate' + str(candidateNum), TOZERO_license_plate)

    (thresh, BINARY_INV_license_plate) = cv2.threshold(TOZERO_license_plate, 160, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('BINARY_INV_license_plate' + str(candidateNum), BINARY_INV_license_plate)

    # se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # license_plate = cv2.morphologyEx(license_plate, cv2.MORPH_OPEN, se1, iterations=1)

    # license_plate = cv2.bilateralFilter(license_plate, -1, 10, 10)
    # sharpenImg = cv2.filter2D(bilateralImg, -1, sharpen)

    candidateNum += 1
    # cv2.imshow('bilateralImg' + str(candidateNum), bilateralImg)

    img_pil = Image.fromarray(BINARY_INV_license_plate)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    result = pytesseract.image_to_string(img_pil, lang='kor')
    result = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", result)
    if result:
        print(result)

cv2.waitKey()
cv2.destroyAllWindows()