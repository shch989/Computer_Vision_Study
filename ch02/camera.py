import cv2

# 얼굴 검출기 로드
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 비디오 캡처 객체 생성
video_capture = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = video_capture.read()

    # 프레임을 흑백으로 변환
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 감지
    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100))

    # 검출된 얼굴 주변에 사각형 그리기
    for (x, y, w, h) in detections:
        print(w, h)  # 얼굴의 가로와 세로 길이 출력
        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)  # 얼굴 주변에 초록색 사각형 그리기

    # 프레임 출력
    cv2.imshow('Video', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체 해제
video_capture.release()
cv2.destroyAllWindows()
