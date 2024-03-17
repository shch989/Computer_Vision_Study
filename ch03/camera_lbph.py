import cv2

# 얼굴 검출기 및 얼굴 인식기 로드
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("lbph_classifier.yml")

# 이미지의 너비와 높이 설정
width, height = 220, 220

# 폰트 및 카메라 객체 생성
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    # 카메라에서 프레임 읽기
    connected, image = camera.read()

    # 그레이스케일 이미지로 변환
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    detections = face_detector.detectMultiScale(
        image_gray, scaleFactor=1.5, minSize=(30, 30))

    # 감지된 얼굴 주변에 사각형 및 이름 및 신뢰도 표시
    for (x, y, w, h) in detections:
        # 얼굴 영역 잘라내고 크기 조정
        image_face = cv2.resize(image_gray[y:y + h, x:x + w], (width, height))

        # 사각형 그리기
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 얼굴 인식기를 사용하여 얼굴 인식
        id, confidence = face_recognizer.predict(image_face)

        # 인식된 얼굴의 이름 설정
        name = ""
        if id == 1:
            name = 'Jones'
        elif id == 2:
            name = 'Gabriel'

        # 화면에 이름과 신뢰도 표시
        cv2.putText(image, name, (x, y + (h + 30)), font, 2, (0, 0, 255))
        cv2.putText(image, str(confidence),
                    (x, y + (h + 50)), font, 1, (0, 0, 255))

    # 화면에 이미지 표시
    cv2.imshow("Face", image)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 카메라 객체 해제
camera.release()
cv2.destroyAllWindows()
