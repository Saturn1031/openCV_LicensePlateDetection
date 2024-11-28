import cv2
import sys
import numpy as np

def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환하여 저장
    blur = cv2.blur(gray, (5, 5))
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, 3)
    _, b_img = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 17), np.uint8)
    morph = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    return morph

def find_candidates(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.minAreaRect(c) for c in contours]  # 외곽 최소 영역
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates

def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h/ w if h > w else w/ h       # 종횡비 계산

    chk1 = 3000 < (h * w) < 12000          # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5       # 번호판 종횡비 조건

    #print(w,h)
    return (chk1 and chk2)


car_no = str(input("자동차 영상 번호 (00~09): "))
img = cv2.imread('cars/'+car_no+'.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
cv2.imshow('original',img)

# 1 전처리 단계 (hw2-2)
preprocessed = preprocessing(img)

cv2.imshow('plate candidate 0',preprocessed)
#cv2.imwrite('hw2_2morph.png', morph)

# 2 번호판 후보 영역 검출 (hw3-2)
candidates = find_candidates(preprocessed)

img2 = img.copy()
for candidate in candidates:  # 후보 영역 표시
    pts = np.int32(cv2.boxPoints(candidate))
    cv2.polylines(img2, [pts], True, (0, 225, 255), 3)

cv2.imshow('plate candidate 1', img2)
#cv2.imwrite('hw3_2candidates.png', img)

cv2.waitKey()
cv2.destroyAllWindows()