"""색상 검출 미리보기 스크립트.

데이터 수집 전에 검정 케이블과 노란 물체가 잘 검출되는지 확인한다.
실행: python scripts/preview_color_detect.py
종료: q 키
"""

import cv2
import numpy as np


YELLOW_LOWER = np.array([20, 80, 80])
YELLOW_UPPER = np.array([35, 255, 255])
CABLE_LOWER  = np.array([0, 0, 0])
CABLE_UPPER  = np.array([180, 50, 50])


def detect(img_bgr, lower, upper):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = None, None
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
    return mask, cx, cy


def main():
    from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids

    ids = get_device_ids()
    if not ids:
        print("RealSense 카메라를 찾을 수 없습니다.")
        return

    # launch_yaml.py와 동일한 순서: 2대 이상이면 ids[1]이 손목 카메라
    wrist_id = ids[1] if len(ids) >= 2 else ids[0]
    cam = RealSenseCamera(device_id=wrist_id)
    print(f"손목 카메라 {wrist_id} 연결됨 (전체 {len(ids)}대). q 키로 종료.")

    while True:
        img_rgb, _ = cam.read(img_size=(640, 480))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        yellow_mask, yx, yy = detect(img_bgr, YELLOW_LOWER, YELLOW_UPPER)
        cable_mask,  cx, cy = detect(img_bgr, CABLE_LOWER,  CABLE_UPPER)

        vis = img_bgr.copy()

        # 마스크 영역 오버레이
        vis[yellow_mask > 0] = (0, 200, 200)   # 노랑 → 청록
        vis[cable_mask > 0]  = (200, 200, 0)   # 검정 → 파랑

        # 중심점 표시
        if yx is not None:
            cv2.circle(vis, (yx, yy), 10, (0, 255, 255), -1)
            cv2.putText(vis, "yellow", (yx + 12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if cx is not None:
            cv2.circle(vis, (cx, cy), 10, (255, 100, 0), -1)
            cv2.putText(vis, "cable", (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        # 상태 텍스트
        y_status = "O" if yx is not None else "X"
        c_status = "O" if cx is not None else "X"
        cv2.putText(vis, f"yellow:{y_status}  cable:{c_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 마스크 창 (나란히)
        yellow_bgr = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
        cable_bgr  = cv2.cvtColor(cable_mask,  cv2.COLOR_GRAY2BGR)
        combined = np.hstack([vis, yellow_bgr, cable_bgr])
        cv2.imshow("preview | yellow mask | cable mask", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
