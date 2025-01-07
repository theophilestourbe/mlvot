from Detector import detect
import numpy as np
from KalmanFilter import KalmanFilter
import cv2 as cv
 
test_file='randomball.avi'


if __name__ == '__main__':
    k_filter = KalmanFilter(d_t=0.1, u_x=1, u_y=1, std_acc=1,
                            x_std_meas=0.1, y_std_meas=0.1)
    

    cap = cv.VideoCapture(test_file)

    rect_size = np.array([25,25], dtype=int)
    rect_half = np.array([12,12], dtype=int)

    last_pos = None

    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 360))
    out2 = cv.VideoWriter('path.avi', fourcc, 20.0, (640, 360))

    path_lines = []
    while cap.isOpened():
        ret, frame = cap.read()
        #print(frame.shape)
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # apply kalman filter
        obj_centers = detect(frame)

        if len(obj_centers) > 0:
            #print('out:', obj_centers)
            detected_pos = obj_centers[0]
            #print(detected_pos)
            # track first detected object
            pred_pos, P_k = k_filter.predict()
            k_filter.update(detected_pos, pred_pos, P_k)

            # retrieve predicted position

            estimated_pos = k_filter.xk[:2]

            # draw 
            # detected
            detected = detected_pos.squeeze(1).astype(int)
            cv.circle(frame, tuple(detected), 5, (0,255,0), 3)
            # pred
            #print('pred', pred_pos.shape)
            #print('pred_val', pred_pos)
            pred = pred_pos[:2].squeeze(1).astype(int)
            #print(pred + rect_size)
            cv.rectangle(frame, tuple(pred - rect_half), tuple(pred + rect_half), (0,0,255), 1)
            # estimated
            est = estimated_pos.squeeze(1).astype(int)
            cv.rectangle(frame, tuple(est - rect_half), tuple(est + rect_half), (255,0,0), 1)

            # track path
            if last_pos is not None:
                # draw line from last_pos to current estimated pos
                path_lines.append([tuple(last_pos), tuple(est)])

            last_pos = est


        #cv.imshow('frame', gray)
        out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break

    blank_frame = (np.zeros((360, 640, 3), dtype=np.uint8) + 255).copy()
    for lines in path_lines:
        cv.line(blank_frame, lines[0], lines[1], (20,20,20), 2)
        out2.write(blank_frame)
    
    cap.release()
    out.release()
    out2.release()
    cv.destroyAllWindows()
