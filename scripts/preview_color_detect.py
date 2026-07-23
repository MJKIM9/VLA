"""색상 검출 미리보기 스크립트.

데이터 수집 전에 검정 케이블과 노란 물체가 잘 검출되는지 확인한다.
실행: python scripts/preview_color_detect.py
종료: q 키
"""

import cv2
import numpy as np


YELLOW_LOWER = np.array([22, 150, 120])
YELLOW_UPPER = np.array([32, 255, 255])
CABLE_LOWER  = np.array([0, 0, 0])
CABLE_UPPER  = np.array([180, 80, 80])


MIN_AREA = 50       # 이 픽셀 수 미만의 덩어리는 노이즈로 무시
CABLE_ROI_TOP    = 0.51   # 상단 컷오프
CABLE_ROI_BOTTOM = 0.65   # 하단 컷오프 (그리퍼 제외)
CABLE_ROI_LEFT   = 0.448  # 좌측 컷오프
CABLE_ROI_RIGHT  = 0.593  # 우측 컷오프
CABLE_TIP_STRIP  = 20     # ROI 라인 위 이 픽셀 범위에서 케이블 x 위치 추출


def detect(img_bgr, lower, upper, roi=None):
    """roi: (top, bottom, left, right) 비율 튜플. None이면 전체 이미지.
    roi 없음(yellow): 모든 픽셀 무게중심 — 쪼개진 경우에도 중심 올바르게 추정.
    roi 있음(cable): ROI 적용 후 하단 슬라이스에서 tip 좌표 추출."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    h, w = mask.shape
    cutoff = None
    if roi is not None:
        top, bottom, left, right = roi
        cutoff = int(h * bottom)
        top_y  = int(h * top)
        left_x = int(w * left)
        right_x = int(w * right)
        mask[:top_y] = 0
        mask[cutoff:] = 0
        mask[:, :left_x] = 0
        mask[:, right_x:] = 0
    cx, cy = None, None
    if cutoff is not None:
        # cable: ROI 하단 슬라이스 tip
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= MIN_AREA:
                strip_top = max(0, cutoff - CABLE_TIP_STRIP)
                strip = mask[strip_top:cutoff, :]
                cols = np.where(strip.any(axis=0))[0]
                if len(cols) > 0:
                    cx = int(cols.mean())
                    cy = cutoff - CABLE_TIP_STRIP // 2
    else:
        # yellow: 모든 픽셀 무게중심
        pts = np.argwhere(mask > 0)
        if len(pts) >= MIN_AREA:
            cy = int(pts[:, 0].mean())
            cx = int(pts[:, 1].mean())
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
    print("마우스 클릭으로 해당 픽셀의 HSV 값 확인 가능.")

    _last_bgr = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and _last_bgr[0] is not None:
            bgr = _last_bgr[0]
            h_img, w_img = bgr.shape[:2]
            # combined 이미지에서 첫 번째 패널(원본)만 클릭 처리
            if x < w_img:
                pixel = bgr[y, x]
                hsv_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                h, s, v = hsv_img[y, x]
                b, g, r = pixel
                print(f"[스포이드] ({x},{y}) BGR=({b},{g},{r})  HSV=({h},{s},{v})")

    cv2.namedWindow("preview | yellow mask | cable mask")
    cv2.setMouseCallback("preview | yellow mask | cable mask", on_mouse)

    while True:
        img_rgb, _ = cam.read(img_size=(640, 480))
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        _last_bgr[0] = img_bgr

        cable_roi = (CABLE_ROI_TOP, CABLE_ROI_BOTTOM, CABLE_ROI_LEFT, CABLE_ROI_RIGHT)
        yellow_mask, yx, yy = detect(img_bgr, YELLOW_LOWER, YELLOW_UPPER)
        cable_mask,  cx, cy = detect(img_bgr, CABLE_LOWER,  CABLE_UPPER, roi=cable_roi)

        vis = img_bgr.copy()

        # 케이블 ROI 사각형 표시
        h, w = img_bgr.shape[:2]
        roi_y1 = int(h * CABLE_ROI_TOP)
        roi_y2 = int(h * CABLE_ROI_BOTTOM)
        roi_x1 = int(w * CABLE_ROI_LEFT)
        roi_x2 = int(w * CABLE_ROI_RIGHT)
        cv2.rectangle(vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 1)

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
