#!/usr/bin/env python3
"""
replay_single_frame.py
──────────────────────
从真实 bag 中提取一帧点云 + TF，循环重复发布，

用法:
  python3 replay_single_frame.py \
      --bag /capella/zbag/outdoor_mk22_lidarbag2 \
      --rate 10 \
      --frame-index 50
"""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions


class SingleFrameReplayer(Node):
    def __init__(self, bag_path: str, rate: float, frame_index: int,
                 lidar_topic: str):
        super().__init__('single_frame_replayer')

        self.rate = rate
        self.frame_count = 0

        # ─── 1. 从 bag 中读取数据 ───
        self.get_logger().info(f'Reading bag: {bag_path}')
        self.get_logger().info(f'Target frame index: {frame_index}')

        pointcloud_msg, tf_msgs, tf_static_msgs = self._read_bag(
            bag_path, lidar_topic, frame_index
        )

        if pointcloud_msg is None:
            self.get_logger().error('Failed to read pointcloud from bag!')
            raise RuntimeError('No pointcloud found in bag')

        # 保存原始点云的 bytes 数据和 fields 信息
        self.saved_pc_msg = pointcloud_msg
        self.lidar_frame = pointcloud_msg.header.frame_id

        self.get_logger().info(
            f'Extracted frame: {pointcloud_msg.width} points, '
            f'frame_id={self.lidar_frame}, '
            f'point_step={pointcloud_msg.point_step}'
        )

        # ─── 2. 发布者 ───
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.pc_pub = self.create_publisher(PointCloud2, lidar_topic, qos)

        # ─── 3. TF 广播 ───
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.saved_tf_msgs = tf_msgs
        self.saved_tf_static_msgs = tf_static_msgs

        # 发布静态 TF
        if tf_static_msgs:
            self.tf_static_broadcaster.sendTransform(tf_static_msgs)
            self.get_logger().info(
                f'Published {len(tf_static_msgs)} static TF transforms'
            )
            for t in tf_static_msgs:
                self.get_logger().info(
                    f'  static TF: {t.header.frame_id} → {t.child_frame_id}'
                )
        else:
            self.get_logger().warn('No /tf_static found in bag!')

        if tf_msgs:
            self.get_logger().info(
                f'Loaded {len(tf_msgs)} dynamic TF transforms'
            )
            for t in tf_msgs:
                self.get_logger().info(
                    f'  dynamic TF: {t.header.frame_id} → {t.child_frame_id}'
                )
        else:
            self.get_logger().warn('No /tf found in bag!')

        # ─── 4. 定时循环发布 ───
        self.timer = self.create_timer(1.0 / self.rate, self._timer_callback)

        self.get_logger().info(
            f'Replayer started: rate={self.rate}Hz, '
            f'publishing to {lidar_topic}'
        )
        self.get_logger().info('=' * 60)

    def _read_bag(self, bag_path: str, lidar_topic: str, frame_index: int):
        """
        从 bag 中读取:
        - 第 frame_index 帧点云
        - 该帧时间戳附近的所有 TF
        - 所有 static TF
        """
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        pointcloud_msg = None
        tf_transforms = []
        tf_static_transforms = []
        pc_count = 0

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic == lidar_topic:
                if pc_count == frame_index:
                    pointcloud_msg = deserialize_message(data, PointCloud2)
                    self.get_logger().info(
                        f'Found target frame at index {pc_count}, '
                        f'timestamp={timestamp}'
                    )
                pc_count += 1

            elif topic == '/tf':
                tf_msg = deserialize_message(data, TFMessage)
                for transform in tf_msg.transforms:
                    # 去重：只保留每对 parent-child 的最后一个
                    key = (transform.header.frame_id,
                           transform.child_frame_id)
                    # 用 dict 去重
                    found = False
                    for i, existing in enumerate(tf_transforms):
                        existing_key = (existing.header.frame_id,
                                       existing.child_frame_id)
                        if existing_key == key:
                            tf_transforms[i] = transform
                            found = True
                            break
                    if not found:
                        tf_transforms.append(transform)

            elif topic == '/tf_static':
                tf_msg = deserialize_message(data, TFMessage)
                for transform in tf_msg.transforms:
                    key = (transform.header.frame_id,
                           transform.child_frame_id)
                    found = False
                    for i, existing in enumerate(tf_static_transforms):
                        existing_key = (existing.header.frame_id,
                                       existing.child_frame_id)
                        if existing_key == key:
                            tf_static_transforms[i] = transform
                            found = True
                            break
                    if not found:
                        tf_static_transforms.append(transform)

            # 如果已经拿到目标帧且已经扫过足够多的数据，可以提前退出
            if pointcloud_msg is not None and pc_count > frame_index + 10:
                break

        self.get_logger().info(f'Total pointcloud frames in bag: {pc_count}')

        return pointcloud_msg, tf_transforms, tf_static_transforms

    def _timer_callback(self):
        self.frame_count += 1
        now = self.get_clock().now().to_msg()

        # ─── 发布 dynamic TF (用当前时间戳) ───
        if self.saved_tf_msgs:
            for tf_transform in self.saved_tf_msgs:
                t = TransformStamped()
                t.header.stamp = now
                t.header.frame_id = tf_transform.header.frame_id
                t.child_frame_id = tf_transform.child_frame_id
                t.transform = tf_transform.transform
                self.tf_broadcaster.sendTransform(t)

        # ─── 发布点云 (用当前时间戳) ───
        pc_msg = PointCloud2()
        pc_msg.header.stamp = now
        pc_msg.header.frame_id = self.saved_pc_msg.header.frame_id
        pc_msg.height = self.saved_pc_msg.height
        pc_msg.width = self.saved_pc_msg.width
        pc_msg.fields = self.saved_pc_msg.fields
        pc_msg.is_bigendian = self.saved_pc_msg.is_bigendian
        pc_msg.point_step = self.saved_pc_msg.point_step
        pc_msg.row_step = self.saved_pc_msg.row_step
        pc_msg.is_dense = self.saved_pc_msg.is_dense
        pc_msg.data = self.saved_pc_msg.data  # 同一份点云数据

        self.pc_pub.publish(pc_msg)

        # 每 20 帧打印一次
        if self.frame_count % 20 == 0:
            self.get_logger().info(
                f'Replayed frame #{self.frame_count} | '
                f'{pc_msg.width} points | '
                f'rate={self.rate}Hz'
            )


def main():
    parser = argparse.ArgumentParser(
        description='从真实 bag 提取一帧点云循环发布，用于推理速度基准测试'
    )
    parser.add_argument('--bag', type=str, required=True,
                        help='bag 文件路径')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='发布频率 Hz (默认 10)')
    parser.add_argument('--frame-index', type=int, default=50,
                        help='提取第几帧点云 (默认 50，跳过前面的启动帧)')
    parser.add_argument('--lidar-topic', type=str, default='/vanjee/lidar',
                        help='点云话题名 (默认 /vanjee/lidar)')

    args = parser.parse_args()

    rclpy.init()
    node = SingleFrameReplayer(
        bag_path=args.bag,
        rate=args.rate,
        frame_index=args.frame_index,
        lidar_topic=args.lidar_topic,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()