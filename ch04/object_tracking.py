import cv2

# CSRT 트래커를 사용하여 객체 추적기 초기화
tracker = cv2.TrackerCSRT_create()

# 비디오 캡처 객체 생성 및 비디오 파일 열기
video = cv2.VideoCapture('street.mp4')

# 첫 번째 프레임 읽기
ok, frame = video.read()

# 객체를 선택하여 추적할 바운딩 박스(ROI) 선택
bbox = cv2.selectROI(frame)

# 추적기 초기화
ok = tracker.init(frame, bbox)
print(ok)

while True:
    # 비디오에서 프레임 읽기
    ok, frame = video.read()
    if not ok:
        break

    # 객체 추적
    ok, bbox = tracker.update(frame)

    if ok:
        # 추적된 객체 주위에 사각형 그리기
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
    else:
        # 추적 실패 시 에러 메시지 표시
        cv2.putText(frame, 'Error', (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 추적된 객체가 표시된 프레임 보여주기
    cv2.imshow('Tracking', frame)

    # 'ESC' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 비디오 캡처 객체 해제 및 창 닫기
video.release()
cv2.destroyAllWindows()
