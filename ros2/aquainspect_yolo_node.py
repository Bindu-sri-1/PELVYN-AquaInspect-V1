import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import torch
torch.set_num_threads(2)


class AquaInspectYOLO(Node):
    def __init__(self):
        super().__init__('aquainspect_yolo')
        self.bridge = CvBridge()
        self.model = YOLO('/mnt/aquainspect/best.pt')

        self.subscription = self.create_subscription(
            Image, '/aquainspect/camera', self.image_callback, 1
        )
        self.publisher = self.create_publisher(
            Image, '/aquainspect/detections', 1
        )

        self.frame_count = 0
        self.get_logger().info(
            'AquaInspect YOLO started; listening on /aquainspect/camera'
        )

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            small_frame = cv2.resize(frame, (256, 192))

            infer_start = time.perf_counter()
            result = self.model.predict(
                small_frame, imgsz=256, conf=0.10, verbose=False
            )[0]
            infer_ms = (time.perf_counter() - infer_start) * 1000

            if self.frame_count % 5 == 0:
                self.get_logger().info(
                    f'YOLO inference time: {infer_ms:.0f} ms'
                )

            annotated = result.plot()
            out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
            out_msg.header = msg.header
            self.publisher.publish(out_msg)

            self.frame_count += 1

            if len(result.boxes) > 0:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    self.get_logger().info(
                        f'WATER HYACINTH detected: confidence={conf:.2f}'
                    )
            elif self.frame_count % 10 == 0:
                self.get_logger().info(
                    f'Frames processed: {self.frame_count}; no detection'
                )

        except Exception as e:
            self.get_logger().error(f'Frame processing error: {e}')


def main():
    rclpy.init()
    node = AquaInspectYOLO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
