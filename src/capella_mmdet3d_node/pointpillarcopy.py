#!/usr/bin/env python3
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


import tempfile
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
import threading
import copy
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import rclpy.duration
from sensor_msgs.msg import PointCloud2, PointField
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.structures import InstanceData

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped, Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

_PC_DTYPE_MAP = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2',
                 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}

CLASS_MAPPING = {
    'car':   'car',
    'Car':   'car',
    'truck': 'truck',
    'Truck': 'truck',
}


# ═══════════════════════════════════════════════════════════════
#  分段推理辅助：模型组件访问
# ═══════════════════════════════════════════════════════════════
def _get_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    raise RuntimeError(f"Cannot find any of {names} on {type(obj)}")


def _get_voxel_layer(model):
    return _get_attr(model.data_preprocessor, 'voxel_layer')


def _get_vfe(model):
    return _get_attr(model, 'pts_voxel_encoder', 'voxel_encoder')


def _get_mid(model):
    return _get_attr(model, 'pts_middle_encoder', 'middle_encoder')


def _get_backbone(model):
    return _get_attr(model, 'pts_backbone', 'backbone')


def _get_neck(model):
    return _get_attr(model, 'pts_neck', 'neck')


def _get_head(model):
    return _get_attr(model, 'pts_bbox_head', 'bbox_head')


def _get_pcr(model):
    try:
        vl = _get_voxel_layer(model)
        if hasattr(vl, 'point_cloud_range') and vl.point_cloud_range is not None:
            return list(vl.point_cloud_range)
    except Exception:
        pass
    try:
        cfg = model.cfg
        def _find(d):
            if isinstance(d, dict):
                if 'point_cloud_range' in d:
                    return d['point_cloud_range']
                for v in d.values():
                    r = _find(v)
                    if r is not None:
                        return r
            elif isinstance(d, (list, tuple)):
                for item in d:
                    r = _find(item)
                    if r is not None:
                        return r
        pcr = _find(dict(cfg))
        if pcr is not None:
            return list(pcr)
    except Exception:
        pass
    return None


def _filter_range(pts, pcr):
    if not isinstance(pcr, np.ndarray):
        pcr = np.array(pcr, dtype=np.float32)
    m = ((pts[:, 0] >= pcr[0]) & (pts[:, 0] <= pcr[3]) &
         (pts[:, 1] >= pcr[1]) & (pts[:, 1] <= pcr[4]) &
         (pts[:, 2] >= pcr[2]) & (pts[:, 2] <= pcr[5]))
    return pts[m]


def _norm_intensity(pts_4d, divisor):
    pts_4d[:, 3] /= divisor
    return pts_4d


# ═══════════════════════════════════════════════════════════════
#  探测 intensity 归一化方式
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def detect_intensity_transform(model, pts_np, pcr):
    print('[IntensityDetect] 探测 intensity 预处理...')
    vl = _get_voxel_layer(model)
    captured_input = {}

    def hook_pre(module, inp):
        if isinstance(inp, tuple) and len(inp) > 0:
            captured_input['pts'] = inp[0].detach().clone()

    h = vl.register_forward_pre_hook(hook_pre)
    if pts_np.shape[1] < 5:
        pts_5d = np.column_stack([pts_np[:, :4],
                                   np.zeros(len(pts_np), dtype=np.float32)])
    else:
        pts_5d = pts_np[:, :5].copy()

    try:
        _ = inference_detector(model, pts_5d)
    finally:
        h.remove()

    if 'pts' not in captured_input:
        print('[IntensityDetect] 无法截取 voxel_layer 输入, 使用默认')
        return lambda x: x, 1.0

    full_pts = captured_input['pts']
    pts_4d = pts_np[:, :4].astype(np.float32)
    if pcr is not None:
        pts_4d = _filter_range(pts_4d, pcr)

    n_full = full_pts.shape[0]
    n_manual = len(pts_4d)
    print(f'[IntensityDetect] 全模型 voxel 输入: {n_full} 点, 手动: {n_manual} 点')

    if n_full != n_manual:
        if pts_4d[:, 3].max() > 1.0:
            return lambda x: _norm_intensity(x, 255.0), 255.0
        return lambda x: x, 1.0

    idx_f = full_pts[:, 0].argsort()
    idx_m = torch.as_tensor(pts_4d, device='cuda')[:, 0].argsort()
    full_sorted = full_pts[idx_f]
    man_sorted = torch.as_tensor(pts_4d, device='cuda')[idx_m]

    xyz_diff = (full_sorted[:, :3] - man_sorted[:, :3]).abs().max().item()
    if xyz_diff > 0.01:
        if pts_4d[:, 3].max() > 1.0:
            return lambda x: _norm_intensity(x, 255.0), 255.0
        return lambda x: x, 1.0

    full_int = full_sorted[:, 3]
    man_int = man_sorted[:, 3]
    max_diff = (full_int - man_int).abs().max().item()

    if max_diff < 0.01:
        print('[IntensityDetect] intensity 完全一致, 无需变换')
        return lambda x: x, 1.0

    nonzero_mask = man_int.abs() > 0.001
    if nonzero_mask.sum() > 100:
        ratios = full_int[nonzero_mask] / man_int[nonzero_mask]
        median_ratio = ratios.median().item()

        for scale_inv, label in [(255.0, '1/255'), (65535.0, '1/65535'),
                                  (100.0, '1/100')]:
            expected_ratio = 1.0 / scale_inv
            if abs(median_ratio - expected_ratio) < 0.001:
                test_int = man_int / scale_inv
                verify_diff = (full_int - test_int).abs().max().item()
                if verify_diff < 0.1:
                    print(f'[IntensityDetect] 检测到 intensity /{scale_inv:.0f} ({label}), '
                          f'验证 max_diff={verify_diff:.6f}')
                    return lambda x: _norm_intensity(x, scale_inv), scale_inv

        scale_inv = 1.0 / median_ratio
        print(f'[IntensityDetect] 使用推断 scale: /{scale_inv:.2f}')
        return lambda x: _norm_intensity(x, scale_inv), scale_inv

    if pts_4d[:, 3].max() > 1.0:
        return lambda x: _norm_intensity(x, 255.0), 255.0
    return lambda x: x, 1.0


# ═══════════════════════════════════════════════════════════════
#  FPN levels 探测
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def detect_num_levels(model):
    backbone = _get_backbone(model)
    neck = _get_neck(model)
    head = _get_head(model)
    dummy = torch.randn(1, 64, 400, 400, device='cuda')
    x = backbone(dummy)
    if isinstance(x, torch.Tensor):
        x = [x]
    x = neck(x)
    raw = head(x)
    return len(raw[0])


# ═══════════════════════════════════════════════════════════════
#  decode_with_head
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def _decode_with_head(head, cls_scores, bbox_preds, dir_cls_preds):
    meta = [dict(box_type_3d=LiDARInstance3DBoxes, box_mode_3d=0)]
    cfg = getattr(head, 'test_cfg', None)
    try:
        return head.predict_by_feat(cls_scores, bbox_preds, dir_cls_preds,
                                     batch_input_metas=meta, cfg=cfg)
    except TypeError:
        res = head.get_bboxes(cls_scores, bbox_preds, dir_cls_preds,
                              input_metas=meta, cfg=cfg)
        out = []
        for boxes, scores, labels in res:
            r = InstanceData()
            r.bboxes_3d = boxes
            r.scores_3d = scores
            r.labels_3d = labels
            out.append(r)
        return out


# ═══════════════════════════════════════════════════════════════
#  TRT Engine 后端
# ═══════════════════════════════════════════════════════════════
class BackboneTRTEngine:
    def __init__(self, engine_path):
        if not HAS_TRT:
            raise ImportError("tensorrt not installed")
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        _dt = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
        }
        self.input_names = []
        self.output_names = []
        self.output_dtypes = {}
        self.output_shapes = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _dt.get(self.engine.get_tensor_dtype(name), torch.float32)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_dtypes[name] = dtype
                self.output_shapes[name] = shape
        self.stream = torch.cuda.Stream()

    @torch.no_grad()
    def infer(self, spatial_features):
        sf = spatial_features.contiguous().float()
        self.context.set_tensor_address('spatial_features', sf.data_ptr())
        outputs = {}
        for name in self.output_names:
            buf = torch.empty(self.output_shapes[name],
                              dtype=self.output_dtypes[name], device='cuda')
            self.context.set_tensor_address(name, buf.data_ptr())
            outputs[name] = buf
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        for k, v in outputs.items():
            if v.dtype != torch.float32:
                outputs[k] = v.float()
        return outputs


# ═══════════════════════════════════════════════════════════════
#  ONNX 多层导出（用于后续转 TRT engine）
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def export_onnx_multilevel(model, save_path):
    backbone = _get_backbone(model)
    neck = _get_neck(model)
    head = _get_head(model)
    dummy = torch.randn(1, 64, 400, 400, device='cuda')
    x = backbone(dummy)
    if isinstance(x, torch.Tensor):
        x = [x]
    x = neck(x)
    raw = head(x)
    n = len(raw[0])

    class _Wrapper(torch.nn.Module):
        def __init__(self, bb, nk, hd, nl):
            super().__init__()
            self.bb = bb
            self.nk = nk
            self.hd = hd
            self.nl = nl

        def forward(self, sf):
            x = self.bb(sf)
            if isinstance(x, torch.Tensor):
                x = [x]
            x = self.nk(x)
            r = self.hd(x)
            out = []
            for lvl in range(self.nl):
                out.extend([r[0][lvl], r[1][lvl], r[2][lvl]])
            return tuple(out)

    wrapper = _Wrapper(backbone, neck, head, n).eval().cuda()
    out_names = []
    for lvl in range(n):
        out_names.extend([f'cls_score_{lvl}', f'bbox_pred_{lvl}', f'dir_cls_pred_{lvl}'])
    torch.onnx.export(wrapper, dummy, save_path,
                       input_names=['spatial_features'],
                       output_names=out_names,
                       opset_version=11,
                       do_constant_folding=True)
    return save_path, n


# ═══════════════════════════════════════════════════════════════

class VehicleTrack:
    def __init__(self, first_center, class_name, size, track_id):
        self.id = track_id
        self.first_center = np.array(first_center)
        self.last_center = np.array(first_center)
        self.class_name = class_name
        self.size = size
        self.missed_frames = 0
        self.is_matched = False
        self.is_moving = False
        self.first_yaw = 0.0
        self.curr_center = np.array(first_center)
        self.curr_yaw = 0.0
        self.curr_size = size
        self.first_size = None
        self.penetration_confirm_frames = 0
        self.moving_suspect_frames = 0
        self._cached_outline = None
        self._outline_key = None


class Mmdet3dNode(Node):
    def __init__(self):
        super().__init__('mmdet3d_node')

        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        _demo_pth = os.path.join(_pkg_dir, 'configs', 'pth',
                                 'pointpillars_hv_fpn_nus.pth')
        _bundled_cfg = os.path.join(
            _pkg_dir, 'configs', 'pointpillars',
            'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
        )
        _default_cfg = _bundled_cfg if os.path.exists(_bundled_cfg) else ''
        self.declare_parameter('config_file',       _default_cfg)
        self.declare_parameter('checkpoint_file',   _demo_pth)
        self.declare_parameter('infer_device',      'cuda:0')
        self.declare_parameter('score_threshold',   0.35)
        self.declare_parameter('pointcloud_topic',  '/vanjee/lidar')
        self.declare_parameter('point_cloud_qos',   'best_effort')
        self.declare_parameter('target_frame',      'map')
        self.declare_parameter('vehicle_boxes_frame',        'map')
        self.declare_parameter('vehicle_raw_cloud_frame',    'map')
        self.declare_parameter('vehicle_outlines',  'map')
        self.declare_parameter('match_distance_threshold', 3.0)     # 汽车一次走了多远内算同一辆车
        self.declare_parameter('max_missed_frames', 10)       # 连续多少帧都没检测到就说明车不在了  跟frames是有关系的
        self.declare_parameter('process_every_n_frames', 1)  # 每多少帧处理一次
        self.declare_parameter('base_box_expand_x', 2.0)
        self.declare_parameter('base_box_expand_y', 4.0)
        self.declare_parameter('lidar_max_range', 80.0)    
        self.declare_parameter('max_static_tracks', 200)
        self.declare_parameter('contour_slice_z', 1.0)
        self.declare_parameter('penetration_threshold', 0.5)   #穿透到目标后方的点数占比的多少时，认为该目标已不存在。
        self.declare_parameter('penetration_confirm_frames_threshold', 2)    # 穿透连续多少帧认为才是真的消失了
        self.declare_parameter('jitter_threshold', 0.3)
        self.declare_parameter('jitter_stable_frames', 10)

        # TRT engine 参数
        _default_engine = os.path.join(_pkg_dir, 'configs', 'pth', 'pointpillars_backbone_multilevel_fp16.engine')
        
        self.declare_parameter('engine_file', _default_engine)

        config_file     = self.get_parameter('config_file').value
        checkpoint_file = self.get_parameter('checkpoint_file').value
        device          = self.get_parameter('infer_device').value
        self.score_thr  = self.get_parameter('score_threshold').value
        self.pc_topic   = self.get_parameter('pointcloud_topic').value
        pc_qos_str      = self.get_parameter('point_cloud_qos').value
        self.target_frame = self.get_parameter('target_frame').value
        self.vehicle_boxes_frame = self.get_parameter('vehicle_boxes_frame').value
        self.vehicle_raw_cloud_frame = self.get_parameter('vehicle_raw_cloud_frame').value
        self.completed_cloud_map_frame = self.get_parameter('vehicle_outlines').value
        self.process_every_n_frames = self.get_parameter('process_every_n_frames').value
        self.match_threshold = self.get_parameter('match_distance_threshold').value
        self.max_missed_frames = self.get_parameter('max_missed_frames').value
        self.base_box_expand_x = self.get_parameter('base_box_expand_x').value
        self.base_box_expand_y = self.get_parameter('base_box_expand_y').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.max_static_tracks = self.get_parameter('max_static_tracks').value
        self.contour_slice_z = self.get_parameter('contour_slice_z').value
        self.penetration_threshold = self.get_parameter('penetration_threshold').value
        self.penetration_confirm_frames_threshold = self.get_parameter('penetration_confirm_frames_threshold').value
        self.jitter_threshold = self.get_parameter('jitter_threshold').value
        self.jitter_stable_frames = self.get_parameter('jitter_stable_frames').value
        engine_file = self.get_parameter('engine_file').value
        self._device = device

        # ---- 加载模型 ----
        self._tmp_config_file = None
        if not config_file:
            ck = torch.load(checkpoint_file, map_location='cpu')
            config_str = ck.get('meta', {}).get('config', None)
            tmp = tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, prefix='mmdet3d_auto_cfg_')
            tmp.write(config_str)
            tmp.close()
            config_file = tmp.name
            self._tmp_config_file = tmp.name

        self.model = init_model(config_file, checkpoint_file, device=device)
        self.model.eval()
        self.class_names: list[str] = list(self.model.dataset_meta.get('classes', []))
        self.get_logger().info(f'Model device: {next(self.model.parameters()).device}')

        # ═══════════════════════════════════════════════════════
        #  分段推理初始化
        # ═══════════════════════════════════════════════════════
        self.get_logger().info('Initializing split inference pipeline...')

        self._voxel_layer = _get_voxel_layer(self.model)
        self._vfe = _get_vfe(self.model)
        self._mid = _get_mid(self.model)
        self._backbone = _get_backbone(self.model)
        self._neck = _get_neck(self.model)
        self._head = _get_head(self.model)

        self._pcr = _get_pcr(self.model)
        self._pcr_np = np.array(self._pcr, dtype=np.float32) if self._pcr else None
        if self._pcr:
            self.get_logger().info(f'point_cloud_range = {self._pcr}')
        else:
            self.get_logger().warn('point_cloud_range not found')

        self._num_levels = detect_num_levels(self.model)
        self.get_logger().info(f'FPN levels = {self._num_levels}')

        # 探测 intensity 归一化
        self.get_logger().info('Detecting intensity normalization...')
        np.random.seed(42)
        _n_test = 5000
        _test_pts = np.zeros((_n_test, 5), dtype=np.float32)
        if self._pcr:
            _test_pts[:, 0] = np.random.uniform(self._pcr[0] + 1, self._pcr[3] - 1, _n_test)
            _test_pts[:, 1] = np.random.uniform(self._pcr[1] + 1, self._pcr[4] - 1, _n_test)
            _test_pts[:, 2] = np.random.uniform(self._pcr[2] + 0.5, self._pcr[5] - 0.5, _n_test)
        else:
            _test_pts[:, 0] = np.random.uniform(-30, 30, _n_test)
            _test_pts[:, 1] = np.random.uniform(-30, 30, _n_test)
            _test_pts[:, 2] = np.random.uniform(-3, 1, _n_test)
        _test_pts[:, 3] = np.random.uniform(100, 50000, _n_test)

        try:
            self._intensity_fn, self._intensity_scale = detect_intensity_transform(
                self.model, _test_pts, self._pcr)
            self.get_logger().info(f'Intensity normalization: /{self._intensity_scale:.0f}')
        except Exception as e:
            self.get_logger().warn(f'Intensity detection failed ({e}), default /65535')
            self._intensity_scale = 65535.0
            self._intensity_fn = lambda x: _norm_intensity(x, 65535.0)

        # 尝试加载 TRT engine
        self._trt_engine = None
        self._backend_name = 'PyTorch-Split'

        if engine_file and os.path.isfile(engine_file) and HAS_TRT:
            try:
                self.get_logger().info(f'Loading TRT engine: {engine_file}')
                eng = BackboneTRTEngine(engine_file)
                has_multi = any('_1' in n or '_2' in n for n in eng.output_names)
                if has_multi or self._num_levels == 1:
                    self._trt_engine = eng
                    self._backend_name = 'TensorRT'
                    self.get_logger().info(
                        f'TRT engine loaded: {len(eng.output_names)} outputs')
                else:
                    self.get_logger().warn(
                        f'TRT engine only has {len(eng.output_names)} outputs (single-level), '
                        f'model needs {self._num_levels} levels. Using PyTorch-Split.')
                    del eng
            except Exception as e:
                self.get_logger().warn(f'TRT engine load failed: {e}')

        if self._trt_engine is None:
            self.get_logger().info('No TRT engine available, using PyTorch-Split (fastest fallback)')

        self.get_logger().info(f'★ Inference backend: {self._backend_name}')

        # 预热
        self.get_logger().info('Warming up split inference...')
        try:
            for _ in range(3):
                self._split_inference(_test_pts)
            torch.cuda.synchronize()
            self.get_logger().info('Warmup complete')
        except Exception as e:
            self.get_logger().warn(f'Warmup failed: {e}')

        # ═══════════════════════════════════════════════════════
        #  原有初始化（完全不变）
        # ═══════════════════════════════════════════════════════
        self._infer_group = MutuallyExclusiveCallbackGroup()
        self._timer_group = MutuallyExclusiveCallbackGroup()
        self._jitter_group = MutuallyExclusiveCallbackGroup()

        if pc_qos_str.lower() == 'reliable':
            _reliability = ReliabilityPolicy.RELIABLE
        elif pc_qos_str.lower() == 'best_effort':
            _reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            self.get_logger().warning(
                f"Invalid point_cloud_qos '{pc_qos_str}', fallback to BEST_EFFORT")
            _reliability = ReliabilityPolicy.BEST_EFFORT
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1, reliability=_reliability)

        self.pc_sub = self.create_subscription(
            PointCloud2, self.pc_topic, self._pointcloud_callback, qos,
            callback_group=self._infer_group)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, 10,
            callback_group=self._jitter_group)

        self.completed_cloud_map_pub = self.create_publisher(
            PointCloud2, '/vehicle_outlines', 10)
        self.vehicle_raw_cloud_pub = self.create_publisher(
            PointCloud2, '/vehicle_raw_cloud', 10)
        self.vehicle_boxes_pub = self.create_publisher(
            MarkerArray, '/vehicle_boxes', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.static_tracks = {}
        self.next_static_id = 0
        self.dynamic_tracks = {}
        self.next_dynamic_id = 0

        self.latest_cmd_linear = 0.0
        self.latest_cmd_angular = 0.0
        self.is_jittering = False
        self.jitter_frozen_static = None
        self.jitter_stable_count = 0
        self.last_stable_translation = None
        self.last_stable_time = None
        self.jitter_lock = threading.Lock()
        self._last_lidar_frame_id = None

        self._frame_counter = 0
        self._last_marker_ids = set()
        self._cached_marker_array = MarkerArray()
        self._cached_timestamp = None
        self._cached_marker_lock = threading.Lock()
        self._last_timing_log_time = 0.0
        self._last_status_log_time = 0.0

        self._cached_pc_dtype = None
        self._cached_pc_frame = None
        self._pc_buf = None

        self.viz_timer = self.create_timer(
            0.2, self._publish_cached_markers, callback_group=self._timer_group)
        self.jitter_timer = self.create_timer(
            0.1, self._tf_monitor_callback, callback_group=self._jitter_group)

        self.get_logger().info(
            f'VehicleDetectionNode ready | match_thr={self.match_threshold}m | '
            f'lidar_max_range={self.lidar_max_range}m | '
            f'jitter_threshold={self.jitter_threshold}m')

    # ═══════════════════════════════════════════════════════════
    #  ★ 分段推理核心 ★
    # ═══════════════════════════════════════════════════════════
    @torch.no_grad()
    def _split_inference(self, pts_np):
        pts_4d = pts_np[:, :4].astype(np.float32, copy=True)

        if self._pcr_np is not None:
            pts_4d = _filter_range(pts_4d, self._pcr_np)

        if len(pts_4d) < 1:
            empty_boxes = LiDARInstance3DBoxes(
                torch.zeros(0, 7, device=self._device))
            return (empty_boxes,
                    torch.zeros(0, device=self._device),
                    torch.zeros(0, dtype=torch.long, device=self._device))

        pts_4d = self._intensity_fn(pts_4d)
        pts_gpu = torch.as_tensor(pts_4d, device=self._device, dtype=torch.float32)

        res = self._voxel_layer(pts_gpu)
        voxels, coors = res[0], res[1]
        num_points = res[2] if len(res) > 2 else None
        if coors.dim() == 2 and coors.shape[1] == 3:
            coors = F.pad(coors, (1, 0), mode='constant', value=0)

        voxel_features = self._vfe(voxels, num_points, coors)
        spatial_features = self._mid(voxel_features, coors, 1)

        if self._trt_engine is not None:
            outputs = self._trt_engine.infer(spatial_features)
            cls_scores, bbox_preds, dir_preds = [], [], []
            for lvl in range(self._num_levels):
                cls_scores.append(
                    outputs.get(f'cls_score_{lvl}', outputs.get('cls_score')))
                bbox_preds.append(
                    outputs.get(f'bbox_pred_{lvl}', outputs.get('bbox_pred')))
                dir_preds.append(
                    outputs.get(f'dir_cls_pred_{lvl}', outputs.get('dir_cls_pred')))
        else:
            x = self._backbone(spatial_features)
            if isinstance(x, torch.Tensor):
                x = [x]
            x = self._neck(x)
            raw = self._head(x)
            cls_scores = raw[0]
            bbox_preds = raw[1]
            dir_preds = raw[2]

        results = _decode_with_head(self._head, cls_scores, bbox_preds, dir_preds)
        pred = results[0]
        return pred.bboxes_3d, pred.scores_3d, pred.labels_3d

    # ═══════════════════════════════════════════════════════════
    #  [判断是否定位是否抖动的]
    # ═══════════════════════════════════════════════════════════

    def _cmd_vel_callback(self, msg):
        self.latest_cmd_linear = msg.linear.x
        self.latest_cmd_angular = msg.angular.z

    def _tf_monitor_callback(self):
        try:
            if self._last_lidar_frame_id is None:
                return
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, self._last_lidar_frame_id, rclpy.time.Time())
        except Exception:
            return

        t = transform.transform.translation
        current_translation = np.array([t.x, t.y, t.z])
        now = self.get_clock().now()

        if self.last_stable_translation is None:
            self.last_stable_translation = current_translation
            self.last_stable_time = now
            return

        dt = (now - self.last_stable_time).nanoseconds / 1e9
        actual_displacement = np.linalg.norm(
            current_translation - self.last_stable_translation)
        expected_displacement = abs(self.latest_cmd_linear) * dt
        is_jitter = abs(actual_displacement - expected_displacement) > self.jitter_threshold

        if not self.is_jittering:
            if is_jitter:
                with self.jitter_lock:
                    self.is_jittering = True
                    self.jitter_frozen_static = copy.deepcopy(self.static_tracks)
                    self.dynamic_tracks.clear()
                self.jitter_stable_count = 0
                self.get_logger().warn(
                    f'[抖动检测] 定位跳变！'
                    f'实际位移={actual_displacement:.3f}m '
                    f'预期位移={expected_displacement:.3f}m '
                    f'阈值={self.jitter_threshold:.3f}m '
                    f'冻结static_tracks，清空dynamic_tracks')
            else:
                self.last_stable_translation = current_translation
                self.last_stable_time = now
        else:
            if not is_jitter:
                self.jitter_stable_count += 1
                if self.jitter_stable_count >= self.jitter_stable_frames:
                    with self.jitter_lock:
                        self.is_jittering = False
                        self.static_tracks = self.jitter_frozen_static
                        self.jitter_frozen_static = None
                    self.get_logger().info(
                        f'[抖动检测] 定位恢复稳定，已恢复冻结的static_tracks')
            else:
                self.jitter_stable_count = 0

            self.last_stable_translation = current_translation
            self.last_stable_time = now

    def _read_pointcloud(self, msg: PointCloud2) -> np.ndarray | None:
        n_pts = msg.width * msg.height
        if n_pts == 0:
            return None

        frame_id = msg.header.frame_id
        if self._cached_pc_frame != frame_id or self._cached_pc_dtype is None:
            fields_sorted = sorted(msg.fields, key=lambda f: f.offset)
            dt_list = []
            pos = 0
            for f in fields_sorted:
                if f.offset > pos:
                    dt_list.append(('_pad%d' % pos, 'u1', f.offset - pos))
                np_type = _PC_DTYPE_MAP.get(f.datatype, 'u1')
                dt_list.append((f.name, np_type))
                pos = f.offset + np.dtype(np_type).itemsize
            if msg.point_step > pos:
                dt_list.append(('_pad_end', 'u1', msg.point_step - pos))
            self._cached_pc_dtype = np.dtype(dt_list)
            self._cached_pc_frame = frame_id

        arr = np.frombuffer(msg.data, dtype=self._cached_pc_dtype)
        field_names = {f.name for f in msg.fields}
        has_intensity = 'intensity' in field_names
        has_ring = 'ring' in field_names

        if self._pc_buf is None or self._pc_buf.shape[0] < n_pts:
            self._pc_buf = np.empty((n_pts, 5), dtype=np.float32)
        buf = self._pc_buf[:n_pts]

        ax = arr['x']
        ay = arr['y']
        az = arr['z']
        buf[:, 0] = ax if ax.dtype == np.float32 else ax.astype(np.float32)
        buf[:, 1] = ay if ay.dtype == np.float32 else ay.astype(np.float32)
        buf[:, 2] = az if az.dtype == np.float32 else az.astype(np.float32)

        valid = np.isfinite(buf[:, 0])
        valid &= np.isfinite(buf[:, 1])
        valid &= np.isfinite(buf[:, 2])
        n_valid = int(valid.sum())
        if n_valid == 0:
            return None

        if has_intensity:
            ai = arr['intensity']
            buf[:, 3] = ai if ai.dtype == np.float32 else ai.astype(np.float32)
        else:
            buf[:, 3] = 0.0

        if has_ring:
            ar = arr['ring']
            buf[:, 4] = ar if ar.dtype == np.float32 else ar.astype(np.float32)
        else:
            buf[:, 4] = 0.0

        if n_valid == n_pts:
            return buf
        return buf[valid].copy()

    def _extract_transform_matrix(self, transform: TransformStamped) -> tuple:
        t = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z])
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [    2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qw*qx)],
            [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]])
        return R, t

    def _id_tracks(self, detections, tracks_dict):
        matches = []
        unmatched_dets = []
        unmatched_tracks = set(tracks_dict.keys())

        if not tracks_dict or not detections:
            return [], list(range(len(detections))), set(tracks_dict.keys()) if tracks_dict else set()

        det_centers = np.array([d['center'] for d in detections], dtype=np.float32)
        track_ids = list(tracks_dict.keys())
        track_centers = np.array([tracks_dict[tid].last_center for tid in track_ids],
                                 dtype=np.float32)

        diff = det_centers[:, None, :] - track_centers[None, :, :]
        dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

        # 构建类别匹配 mask: (D, T) bool
        det_classes = [d['class_name'] for d in detections]
        track_classes = [tracks_dict[tid].class_name for tid in track_ids]
        class_mask = np.array([[dc == tc for tc in track_classes] for dc in det_classes],
                              dtype=bool)

        # 把不同类别的距离设为 inf
        dist_matrix_masked = dist_matrix.copy()
        dist_matrix_masked[~class_mask] = np.inf

        # 按距离从小到大贪心匹配
        flat_indices = np.argsort(dist_matrix_masked.ravel())
        n_tracks = len(track_ids)
        det_used = set()
        track_used = set()

        for flat_idx in flat_indices:
            di = int(flat_idx // n_tracks)
            ti = int(flat_idx % n_tracks)
            d = float(dist_matrix_masked[di, ti])
            if d >= self.match_threshold:
                break
            if di in det_used or ti in track_used:
                continue
            matches.append((di, track_ids[ti]))
            det_used.add(di)
            track_used.add(ti)

        unmatched_dets = [i for i in range(len(detections)) if i not in det_used]
        matched_track_id_set = {track_ids[ti] for ti in track_used}
        unmatched_tracks = set(tracks_dict.keys()) - matched_track_id_set

        return matches, unmatched_dets, unmatched_tracks

    def _match_dynamic_track(self, center, dynamic_tracks):
        center_np = np.array(center)
        best_id = None
        min_dist = float('inf')
        for dyn_id, dyn_info in dynamic_tracks.items():
            dist = np.linalg.norm(center_np - np.array(dyn_info['center']))
            if dist < self.match_threshold and dist < min_dist:
                min_dist = dist
                best_id = dyn_id
        return best_id

    def _is_moving_check(self, current_center, first_center, prior_size, first_yaw):
        dx = current_center[0] - first_center[0]
        dy = current_center[1] - first_center[1]
        cos_b, sin_b = math.cos(-first_yaw), math.sin(-first_yaw)
        local_x = dx * cos_b - dy * sin_b
        local_y = dx * sin_b + dy * cos_b
        bl, bw, bh = prior_size
        inside_x = abs(local_x) <= bl / 2
        inside_y = abs(local_y) <= bw / 2
        center_inside_base = inside_x and inside_y
        is_moving = not center_inside_base
        return is_moving, center_inside_base

    def _lidar_can_confirm_empty(self, pts_lidar, region_center_lidar,
                                   region_size, _precomputed=None):
        """穿透比例 = 穿透点数 / (穿透点数 + 内部点数)"""
        d_target = float(np.linalg.norm(region_center_lidar))
        if d_target > self.lidar_max_range or d_target < 0.5:
            return False
        direction = region_center_lidar / d_target
        l, w, h = region_size
        half_span = max(l, w) / 2.0
        angle_tol = math.atan2(half_span, d_target)

        if _precomputed is not None:
            pts_valid, dists_valid, dirs = _precomputed
        else:
            dists = np.linalg.norm(pts_lidar, axis=1)
            valid = dists > 0.1
            pts_valid = pts_lidar[valid]
            dists_valid = dists[valid]
            if len(pts_valid) == 0:
                return False
            dirs = pts_valid / dists_valid[:, np.newaxis]

        if len(pts_valid) == 0:
            return False

        cos_angles = dirs @ direction
        cos_tol = math.cos(angle_tol)
        in_cone = cos_angles >= cos_tol
        if np.sum(in_cone) == 0:
            return False

        cone_pts = pts_valid[in_cone]
        cone_dists = dists_valid[in_cone]
        target_z = region_center_lidar[2]
        z_min = target_z - h / 2.0 + 0.3
        z_max = target_z + h / 2.0 + 0.3
        above_ground = (cone_pts[:, 2] > z_min) & (cone_pts[:, 2] < z_max)
        cone_dists_filtered = cone_dists[above_ground]

        if len(cone_dists_filtered) == 0:
            return False

        d_near = d_target - half_span
        d_far = d_target + half_span
        n_in_region = np.sum(
            (cone_dists_filtered >= d_near) & (cone_dists_filtered <= d_far))

        if n_in_region <= 3:
            return True

        n_through = np.sum(cone_dists_filtered > d_far)
        if n_through == 0:
            return False

        ratio = float(n_through) / (n_through + n_in_region)
        return ratio >= self.penetration_threshold

    def _batch_lidar_confirm_empty(self, pts_lidar, track_infos, _precomputed):
        """批量检查多个 track 的穿透情况。
        track_infos: list of (track_id, center_lidar, actual_size)
        返回 {track_id: True/False}
        """
        if not track_infos:
            return {}

        pts_valid, dists_valid, dirs = _precomputed
        if len(pts_valid) == 0:
            return {tid: False for tid, _, _ in track_infos}

        centers = np.array([info[1] for info in track_infos], dtype=np.float32)
        d_targets = np.linalg.norm(centers, axis=1)
        directions = centers / np.maximum(d_targets[:, np.newaxis], 1e-6)

        all_cos_angles = dirs @ directions.T

        half_spans = np.array([max(s[0], s[1]) / 2.0 for _, _, s in track_infos],
                              dtype=np.float32)
        angle_tols = np.arctan2(half_spans, d_targets)
        cos_tols = np.cos(angle_tols)

        results = {}
        for i, (tid, center_lidar, actual_size) in enumerate(track_infos):
            d_target = float(d_targets[i])
            if d_target > self.lidar_max_range or d_target < 0.5:
                results[tid] = False
                continue

            cos_tol = float(cos_tols[i])
            in_cone = all_cos_angles[:, i] >= cos_tol
            if np.sum(in_cone) == 0:
                results[tid] = False
                continue

            cone_pts = pts_valid[in_cone]
            cone_dists = dists_valid[in_cone]

            target_z = float(center_lidar[2])
            l, w, h = actual_size
            z_min = target_z - h / 2.0 + 0.3
            z_max = target_z + h / 2.0 + 0.3
            above_ground = (cone_pts[:, 2] > z_min) & (cone_pts[:, 2] < z_max)
            cone_dists_filtered = cone_dists[above_ground]

            if len(cone_dists_filtered) == 0:
                results[tid] = False
                continue

            half_span = max(l, w) / 2.0
            d_near = d_target - half_span
            d_far = d_target + half_span
            n_in_region = np.sum(
                (cone_dists_filtered >= d_near) & (cone_dists_filtered <= d_far))

            if n_in_region <= 3:
                results[tid] = True
                continue

            n_through = np.sum(cone_dists_filtered > d_far)
            if n_through == 0:
                results[tid] = False
                continue

            ratio = float(n_through) / (n_through + n_in_region)
            results[tid] = ratio >= self.penetration_threshold

        return results

    def create_box_marker(self, center, size, color, marker_id, ns,
                          frame_id, header_stamp, yaw=0.0):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        half_yaw = yaw * 0.5
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(half_yaw)
        marker.pose.orientation.w = math.cos(half_yaw)
        cx, cy, cz = center
        l, w, h = size
        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = cz
        marker.scale.x = l
        marker.scale.y = w
        marker.scale.z = 0.1
        r, g, b, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        return marker

    def create_line_marker(self, start, end, color, marker_id,
                           frame_id, header_stamp):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = "displacement_lines"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.pose.orientation.w = 1.0
        r, g, b, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        p1 = Point()
        p1.x = start[0]; p1.y = start[1]; p1.z = start[2]
        marker.points.append(p1)
        p2 = Point()
        p2.x = end[0]; p2.y = end[1]; p2.z = end[2]
        marker.points.append(p2)
        return marker

    def create_center_point_marker(self, center, color, marker_id, ns,
                                   frame_id, header_stamp):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = header_stamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        cx, cy, _ = center
        marker.pose.position.x = cx
        marker.pose.position.y = cy
        marker.pose.position.z = self.contour_slice_z
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35
        r, g, b, a = color
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        return marker

    def _pointcloud_callback(self, msg: PointCloud2):
        self._frame_counter += 1
        if self._frame_counter % self.process_every_n_frames != 0:
            return
        self._last_lidar_frame_id = msg.header.frame_id
        try:
            self._process_pointcloud(msg)
        except Exception as e:
            self.get_logger().error(f'Inference callback error: {e}')

    def _process_pointcloud(self, msg: PointCloud2):
        with self.jitter_lock:
            if self.is_jittering:
                return

        for track in self.static_tracks.values():
            track.is_matched = False

        _t0 = time.perf_counter()
        source_frame = msg.header.frame_id
        target_frame = self.target_frame

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame,
                rclpy.time.Time.from_msg(msg.header.stamp),
                timeout=rclpy.duration.Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return

        R, t = self._extract_transform_matrix(transform)
        _t1 = time.perf_counter()

        pts = self._read_pointcloud(msg)
        if pts is None or len(pts) < 10:
            return
        _t2 = time.perf_counter()

        # ---- ★ 分段推理 ★ ----
        try:
            bboxes, scores, labels = self._split_inference(pts)
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            torch.cuda.empty_cache()
            return
        _t3 = time.perf_counter()

        # ---- 置信度过滤 ----
        keep = (scores > self.score_thr).nonzero(as_tuple=False).squeeze(1)
        if keep.numel() > 0:
            bboxes = bboxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        else:
            labels = labels[:0]

        current_detections = []

        if keep.numel() > 0:
            for i in range(len(labels)):
                label = int(labels[i].item())
                if label >= len(self.class_names):
                    continue
                class_name = self.class_names[label]
                if class_name not in CLASS_MAPPING:
                    continue
                class_name = CLASS_MAPPING[class_name]

                bbox = bboxes[i]
                center = bbox.center
                if center.dim() == 2:
                    center = center[0]
                cx_local = float(center[0])
                cy_local = float(center[1])
                cz_local = float(center[2])
                try:
                    yaw_local = float(bboxes[i].tensor[0, 6])
                except Exception:
                    yaw_local = 0.0

                center_map = self.transform_point(
                    (cx_local, cy_local, cz_local), R, t)

                try:
                    model_l = float(bboxes[i].tensor[0, 3])
                    model_w = float(bboxes[i].tensor[0, 4])
                    model_h = float(bboxes[i].tensor[0, 5])
                except Exception as e:
                    self.get_logger().warning(
                        f'Failed to extract bbox size: {e}. Skipping.')
                    continue

                # 校正 PointPillars 90° yaw 的问题
                if class_name in ('car', 'truck') and model_w > model_l:
                    model_l, model_w = model_w, model_l
                    yaw_local = yaw_local + math.pi / 2.0
                    yaw_local = math.atan2(math.sin(yaw_local), math.cos(yaw_local))

                current_detections.append({
                    'center': center_map,
                    'class_name': class_name,
                    'model_bbox_size': (model_l, model_w, model_h),
                    'local_center': (cx_local, cy_local, cz_local),
                    'yaw_local': yaw_local,
                    'bbox': bbox,
                })

        _t4 = time.perf_counter()

        matches, unmatched_dets, unmatched_static_ids = self._id_tracks(
            current_detections, self.static_tracks)

        bbox_points_cache = {}
        tracks_to_delete = []
        matched_dynamic_ids = set()

        for det_idx, track_id in matches:
            det = current_detections[det_idx]
            if track_id not in self.static_tracks:
                continue
            track = self.static_tracks[track_id]

            yaw_map = self._transform_yaw_to_map(det.get('yaw_local', 0.0), R)
            det['yaw_map'] = yaw_map

            track.last_center = np.array(det['center'])
            track.missed_frames = 0
            track.is_matched = True
            track.penetration_confirm_frames = 0
            track.curr_center = np.array(det['center'])
            track.curr_size = det['model_bbox_size']
            track.curr_yaw = yaw_map

            is_moving, center_inside_base = self._is_moving_check(
                det['center'], track.first_center, track.size, track.first_yaw)

            if is_moving:
                track.moving_suspect_frames += 1
                if track.moving_suspect_frames >= 3:
                    if not track.is_moving:
                        self.get_logger().info(
                            f'Track {track_id} ({track.class_name}) confirmed MOVING '
                            f'after {track.moving_suspect_frames} consecutive frames. '
                            f'Converting to dynamic.')
                    track.is_moving = True
                    del self.static_tracks[track_id]

                    matched_dyn_id = self._match_dynamic_track(
                        det['center'], self.dynamic_tracks)

                    if matched_dyn_id is not None:
                        dyn_info = self.dynamic_tracks[matched_dyn_id]
                        dyn_info['center'] = det['center']
                        dyn_info['size'] = det['model_bbox_size']
                        dyn_info['yaw'] = yaw_map
                        dyn_info['label'] = det['class_name']
                        dyn_info['local_center'] = det['local_center']
                        dyn_info['local_yaw'] = det['yaw_local']
                        cx_l, cy_l, cz_l = det['local_center']
                        l, w, h = det['model_bbox_size']
                        y_l = det['yaw_local']
                        if det_idx not in bbox_points_cache:
                            bbox_points_cache[det_idx] = self._extract_bbox_points(
                                pts, cx_l, cy_l, cz_l, l, w, h, y_l)
                        dyn_info['bbox_points'] = bbox_points_cache[det_idx]
                        dyn_info['is_matched'] = True
                        matched_dynamic_ids.add(matched_dyn_id)
                    else:
                        new_dyn_id = self.next_dynamic_id
                        self.next_dynamic_id += 1
                        cx_l, cy_l, cz_l = det['local_center']
                        l, w, h = det['model_bbox_size']
                        y_l = det['yaw_local']
                        if det_idx not in bbox_points_cache:
                            bbox_points_cache[det_idx] = self._extract_bbox_points(
                                pts, cx_l, cy_l, cz_l, l, w, h, y_l)
                        self.dynamic_tracks[new_dyn_id] = {
                            'center': det['center'],
                            'size': det['model_bbox_size'],
                            'yaw': yaw_map,
                            'label': det['class_name'],
                            'local_center': det['local_center'],
                            'local_yaw': det['yaw_local'],
                            'bbox_points': bbox_points_cache[det_idx],
                            'is_matched': True,
                        }
                        matched_dynamic_ids.add(new_dyn_id)
                    continue
            else:
                track.moving_suspect_frames = 0

            det['track_id'] = track_id
            det['is_moving'] = is_moving
            det['center_inside_base'] = center_inside_base

        remaining_unmatched_dets = []
        for det_idx in unmatched_dets:
            det = current_detections[det_idx]
            matched_dyn_id = self._match_dynamic_track(
                det['center'], self.dynamic_tracks)

            if matched_dyn_id is not None:
                yaw_map = self._transform_yaw_to_map(
                    det.get('yaw_local', 0.0), R)
                det['yaw_map'] = yaw_map
                dyn_info = self.dynamic_tracks[matched_dyn_id]
                dyn_info['center'] = det['center']
                dyn_info['size'] = det['model_bbox_size']
                dyn_info['yaw'] = yaw_map
                dyn_info['label'] = det['class_name']
                dyn_info['local_center'] = det['local_center']
                dyn_info['local_yaw'] = det['yaw_local']
                cx_l, cy_l, cz_l = det['local_center']
                l, w, h = det['model_bbox_size']
                y_l = det['yaw_local']
                if det_idx not in bbox_points_cache:
                    bbox_points_cache[det_idx] = self._extract_bbox_points(
                        pts, cx_l, cy_l, cz_l, l, w, h, y_l)
                dyn_info['bbox_points'] = bbox_points_cache[det_idx]
                dyn_info['is_matched'] = True
                matched_dynamic_ids.add(matched_dyn_id)
                det['is_dynamic_matched'] = True
                det['dyn_id'] = matched_dyn_id
            else:
                remaining_unmatched_dets.append(det_idx)

        for det_idx in remaining_unmatched_dets:
            det = current_detections[det_idx]
            yaw_map = self._transform_yaw_to_map(det.get('yaw_local', 0.0), R)
            det['yaw_map'] = yaw_map

            model_l, model_w, model_h = det['model_bbox_size']
            base_l = model_l + self.base_box_expand_x
            base_w = model_w + self.base_box_expand_y
            base_h = model_h

            new_id = self.next_static_id
            self.next_static_id += 1

            new_track = VehicleTrack(
                first_center=det['center'], class_name=det['class_name'],
                size=(base_l, base_w, base_h), track_id=new_id)
            new_track.is_matched = True
            new_track.is_moving = False
            new_track.first_yaw = yaw_map
            new_track.curr_center = np.array(det['center'])
            new_track.curr_size = det['model_bbox_size']
            new_track.curr_yaw = yaw_map
            new_track.first_size = det['model_bbox_size']
            self.static_tracks[new_id] = new_track

            self.get_logger().info(
                f'New static track {new_id} ({det["class_name"]}) created at '
                f'({det["center"][0]:.1f}, {det["center"][1]:.1f}). '
                f'Base box: ({base_l:.2f}x{base_w:.2f}x{base_h:.2f}m)')

            det['track_id'] = new_id
            det['is_moving'] = False
            det['center_inside_base'] = True

        _t5 = time.perf_counter()

        _pts_xyz3 = pts[:, :3]
        _dists_pre = np.linalg.norm(_pts_xyz3, axis=1)
        _valid_pre = _dists_pre > 0.1
        _pts_v_pre = _pts_xyz3[_valid_pre]
        _dists_v_pre = _dists_pre[_valid_pre]
        if len(_pts_v_pre) > 0:
            _dirs_pre = _pts_v_pre / _dists_v_pre[:, np.newaxis]
        else:
            _dirs_pre = np.empty((0, 3), dtype=np.float32)
        _precomp = (_pts_v_pre, _dists_v_pre, _dirs_pre)

        track_infos = []
        for track_id in unmatched_static_ids:
            if track_id in tracks_to_delete:
                continue
            track = self.static_tracks[track_id]
            track.is_matched = False
            center_lidar = R.T @ (track.first_center - t)
            actual_size = track.first_size if track.first_size is not None else track.size
            track_infos.append((track_id, center_lidar, actual_size))

        batch_results = self._batch_lidar_confirm_empty(
            pts[:, :3], track_infos, _precomp)

        for track_id, _, _ in track_infos:
            lidar_confirms_empty = batch_results[track_id]
            track = self.static_tracks[track_id]
            if lidar_confirms_empty:
                track.penetration_confirm_frames += 1
                if track.penetration_confirm_frames >= self.penetration_confirm_frames_threshold:
                    tracks_to_delete.append(track_id)
                    self.get_logger().info(
                        f'Static track {track_id} confirmed gone by lidar penetration '
                        f'({track.penetration_confirm_frames} consecutive frames), deleting')
            else:
                track.penetration_confirm_frames = 0

        _t6 = time.perf_counter()

        for tid in tracks_to_delete:
            if tid in self.static_tracks:
                del self.static_tracks[tid]

        all_dynamic_ids = set(self.dynamic_tracks.keys())
        unmatched_dynamic_ids = all_dynamic_ids - matched_dynamic_ids
        for dyn_id in unmatched_dynamic_ids:
            del self.dynamic_tracks[dyn_id]

        if unmatched_dynamic_ids:
            self.get_logger().debug(
                f'Removed {len(unmatched_dynamic_ids)} unmatched dynamic track(s): '
                f'{unmatched_dynamic_ids}')

        static_ids = [tid for tid, tk in self.static_tracks.items()
                      if not tk.is_moving]
        if len(static_ids) > self.max_static_tracks:
            static_ids_sorted = sorted(
                static_ids,
                key=lambda tid: np.linalg.norm(
                    self.static_tracks[tid].first_center - t),
                reverse=True)
            overflow = len(static_ids) - self.max_static_tracks
            for tid in static_ids_sorted[:overflow]:
                self.get_logger().info(
                    f'Static track FIFO: evicting track {tid} '
                    f'(dist={np.linalg.norm(self.static_tracks[tid].first_center - t):.1f}m)')
                del self.static_tracks[tid]
            self.get_logger().warn(
                f'Static track overflow: evicted {overflow} farthest track(s), '
                f'limit={self.max_static_tracks}')

        marker_array = MarkerArray()
        marker_id = 0
        vehicle_raw_points_local = []

        _should_log_status = (time.time() - self._last_status_log_time >= 3.0)
        if _should_log_status:
            self._last_status_log_time = time.time()

        for det in current_detections:
            if 'track_id' not in det or 'is_dynamic_matched' in det:
                continue
            track_id = det['track_id']
            if track_id not in self.static_tracks:
                continue

            is_moving = det['is_moving']
            center_inside_base = det['center_inside_base']
            center_map = det['center']
            class_name = det['class_name']
            model_l, model_w, model_h = det['model_bbox_size']

            track = self.static_tracks[track_id]
            base_center = track.first_center
            base_l, base_w, base_h = track.size

            cx_local, cy_local, cz_local = det['local_center']
            yaw_local = det.get('yaw_local', 0.0)
            det_idx_viz = current_detections.index(det)
            if det_idx_viz not in bbox_points_cache:
                bbox_points_cache[det_idx_viz] = self._extract_bbox_points(
                    pts, cx_local, cy_local, cz_local, model_l, model_w, model_h,
                    yaw_local)
            bbox_points_local = bbox_points_cache[det_idx_viz]

            if len(bbox_points_local) > 0:
                vehicle_raw_points_local.append(bbox_points_local)

            base_marker = self.create_box_marker(
                center=base_center, size=(base_l, base_w, base_h),
                color=(0.0, 1.0, 0.0, 0.3), marker_id=track_id,
                ns="base_boxes", frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp, yaw=track.first_yaw)
            marker_array.markers.append(base_marker)

            if is_moving:
                current_color = (1.0, 0.0, 0.0, 1.0)
            else:
                current_color = (0.0, 0.0, 1.0, 1.0)

            current_marker = self.create_box_marker(
                center=center_map, size=(model_l, model_w, model_h),
                color=current_color, marker_id=track_id + 10000,
                ns="current_boxes", frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp, yaw=det['yaw_map'])
            marker_array.markers.append(current_marker)

            if is_moving:
                line_marker = self.create_line_marker(
                    start=base_center, end=center_map,
                    color=(1.0, 1.0, 0.0, 1.0), marker_id=track_id + 20000,
                    frame_id=self.vehicle_boxes_frame,
                    header_stamp=msg.header.stamp)
                marker_array.markers.append(line_marker)

            if not is_moving:
                center_point_color = (0.0, 0.0, 1.0, 1.0)
                center_point_ns = "center_points_static"
            else:
                center_point_color = (1.0, 0.0, 0.0, 1.0)
                center_point_ns = "center_points_suspect"

            center_marker = self.create_center_point_marker(
                center=center_map, color=center_point_color,
                marker_id=track_id + 30000, ns=center_point_ns,
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp)
            marker_array.markers.append(center_marker)

            status_str = "MOVING" if is_moving else "STATIC"
            if _should_log_status:
                self.get_logger().info(
                    f'[Static ID:{track_id}] {class_name} | '
                    f'Base:({base_center[0]:.1f},{base_center[1]:.1f}) | '
                    f'Curr:({center_map[0]:.1f},{center_map[1]:.1f}) | '
                    f'CenterInside:{center_inside_base} | {status_str}')

        for track_id, track in self.static_tracks.items():
            if track.is_matched or track.is_moving:
                continue
            track.curr_center = track.first_center
            track.curr_yaw = track.first_yaw
            track.curr_size = (track.first_size
                               if track.first_size is not None else track.size)
            l, w, h = track.size
            color = (0.0, 1.0, 0.0, 0.3)
            ns = "memory_static"
            marker = self.create_box_marker(
                center=track.first_center, size=(l, w, h), color=color,
                marker_id=track_id, ns=ns, frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp, yaw=track.first_yaw)
            marker_array.markers.append(marker)

        for dyn_id, dyn_info in self.dynamic_tracks.items():
            center = dyn_info['center']
            size = dyn_info['size']
            yaw = dyn_info['yaw']
            bbox_points = dyn_info['bbox_points']

            if len(bbox_points) > 0:
                vehicle_raw_points_local.append(bbox_points)

            dyn_marker = self.create_box_marker(
                center=center, size=size, color=(1.0, 0.0, 0.0, 1.0),
                marker_id=dyn_id, ns="dynamic_boxes",
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp, yaw=yaw)
            marker_array.markers.append(dyn_marker)

            dyn_center_marker = self.create_center_point_marker(
                center=center, color=(1.0, 0.0, 0.0, 1.0),
                marker_id=dyn_id + 40000, ns="center_points_dynamic",
                frame_id=self.vehicle_boxes_frame,
                header_stamp=msg.header.stamp)
            marker_array.markers.append(dyn_center_marker)

        self.get_logger().debug(
            f'Frame status: {len(self.static_tracks)} static, '
            f'{len(self.dynamic_tracks)} dynamic')

        all_outline_points = []
        for track_id, track in self.static_tracks.items():
            outline_pts = self._generate_outline_from_track(track)
            if len(outline_pts) > 0:
                all_outline_points.append(outline_pts)

        if all_outline_points:
            combined_outlines = np.vstack(all_outline_points)
            map_header = Header()
            map_header.stamp = msg.header.stamp
            map_header.frame_id = self.completed_cloud_map_frame
            self._publish_completed_cloud_map(combined_outlines, map_header)

        if vehicle_raw_points_local:
            combined_raw = np.vstack(vehicle_raw_points_local)
            combined_raw_map = self.transform_pointcloud(combined_raw, R, t)
            raw_header = Header()
            raw_header.stamp = msg.header.stamp
            raw_header.frame_id = self.vehicle_raw_cloud_frame
            self._publish_vehicle_raw_cloud(combined_raw_map, raw_header)

        current_marker_ids = {(m.ns, m.id) for m in marker_array.markers}
        for ns, mid in self._last_marker_ids:
            if (ns, mid) not in current_marker_ids:
                del_marker = Marker()
                del_marker.header.frame_id = self.vehicle_boxes_frame
                del_marker.header.stamp = msg.header.stamp
                del_marker.ns = ns
                del_marker.id = mid
                del_marker.action = Marker.DELETE
                marker_array.markers.append(del_marker)
        self._last_marker_ids = current_marker_ids

        _t7 = time.perf_counter()
        with self._cached_marker_lock:
            self._cached_marker_array = marker_array
            self._cached_timestamp = msg.header.stamp

        _now = time.time()
        if _now - self._last_timing_log_time >= 3.0:
            self._last_timing_log_time = _now
            self.get_logger().info(
                f'[TIMING] total={(_t7-_t0)*1e3:.1f}ms | '
                f'tf={(_t1-_t0)*1e3:.1f}ms | '
                f'read={(_t2-_t1)*1e3:.1f}ms | '
                f'infer={(_t3-_t2)*1e3:.1f}ms | '
                f'postproc={(_t4-_t3)*1e3:.1f}ms | '
                f'match={(_t5-_t4)*1e3:.1f}ms | '
                f'track={(_t6-_t5)*1e3:.1f}ms | '
                f'publish={(_t7-_t6)*1e3:.1f}ms | '
                f'static={len(self.static_tracks)} | '
                f'dynamic={len(self.dynamic_tracks)} | '
                f'backend={self._backend_name}')

    def _publish_cached_markers(self):
        with self._cached_marker_lock:
            if not self._cached_marker_array.markers or self._cached_timestamp is None:
                return
            ma = copy.deepcopy(self._cached_marker_array)
        now_stamp = self.get_clock().now().to_msg()
        for marker in ma.markers:
            marker.header.stamp = now_stamp
        self.vehicle_boxes_pub.publish(ma)

    def _extract_bbox_points(self, pts, cx, cy, cz, l, w, h, yaw):
        margin = 0.1
        xyz = pts[:, :3]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        half_l = l / 2.0
        half_w = w / 2.0
        half_z = h / 2.0
        abs_cos = abs(cos_yaw)
        abs_sin = abs(sin_yaw)
        half_x = abs_cos * half_l + abs_sin * half_w
        half_y = abs_sin * half_l + abs_cos * half_w
        x_min, x_max = cx - half_x - margin, cx + half_x + margin
        y_min, y_max = cy - half_y - margin, cy + half_y + margin
        z_min, z_max = cz - half_z - margin, cz + half_z + margin
        aabb_mask = (
            (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
            (xyz[:, 1] >= y_min) & (xyz[:, 1] <= y_max) &
            (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max))
        cand_xyz = xyz[aabb_mask]
        if cand_xyz.size == 0:
            return cand_xyz
        shifted = cand_xyz - np.array([cx, cy, cz], dtype=np.float32)
        cos_r = cos_yaw
        sin_r = -sin_yaw
        local_x = shifted[:, 0] * cos_r - shifted[:, 1] * sin_r
        local_y = shifted[:, 0] * sin_r + shifted[:, 1] * cos_r
        local_z = shifted[:, 2]
        obb_mask = (
            (np.abs(local_x) <= half_l + margin) &
            (np.abs(local_y) <= half_w + margin) &
            (np.abs(local_z) <= half_z + margin))
        return cand_xyz[obb_mask]

    def _transform_yaw_to_map(self, yaw_local, R):
        dir_local = np.array([
            math.cos(yaw_local), math.sin(yaw_local), 0.0])
        dir_map = R @ dir_local
        return math.atan2(dir_map[1], dir_map[0])

    def transform_point(self, point_local, R, t):
        p_local = np.array(point_local)
        p_map = R @ p_local + t
        return (float(p_map[0]), float(p_map[1]), float(p_map[2]))

    def transform_pointcloud(self, points_local, R, t):
        if len(points_local) == 0:
            return points_local
        return (R @ points_local.T).T + t

    def _generate_outline_from_track(self, track):
        key = (float(track.curr_center[0]), float(track.curr_center[1]),
               float(track.curr_center[2]),
               float(track.curr_size[0]), float(track.curr_size[1]),
               float(track.curr_size[2]), float(track.curr_yaw))
        if track._outline_key == key and track._cached_outline is not None:
            return track._cached_outline
        cx, cy = track.curr_center[0], track.curr_center[1]
        l, w, h = track.curr_size
        yaw = track.curr_yaw
        resolution = 0.5
        x_vals = np.arange(-l/2, l/2, resolution)
        y_vals = np.arange(-w/2, w/2, resolution)
        xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
        local_x = xx.ravel()
        local_y = yy.ravel()
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        map_x = cx + local_x * cos_y - local_y * sin_y
        map_y = cy + local_x * sin_y + local_y * cos_y
        map_z = np.full_like(map_x, self.contour_slice_z)
        result = np.column_stack([map_x, map_y, map_z]).astype(np.float32)
        track._cached_outline = result
        track._outline_key = key
        return result

    def _publish_vehicle_raw_cloud(self, points, header):
        if len(points) == 0:
            return
        intensity = np.full((len(points), 1), 999.0, dtype=np.float32)
        points_with_intensity = np.hstack(
            [points[:, :3].astype(np.float32), intensity])
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = self.vehicle_raw_cloud_frame
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = points_with_intensity.tobytes()
        self.vehicle_raw_cloud_pub.publish(msg)

    def _publish_completed_cloud_map(self, points, header):
        if len(points) == 0:
            return
        intensity = np.full((len(points), 1), 2048.0, dtype=np.float32)
        points_with_intensity = np.hstack(
            [points[:, :3].astype(np.float32), intensity])
        msg = PointCloud2()
        msg.header = header
        msg.header.frame_id = self.completed_cloud_map_frame
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = points_with_intensity.tobytes()
        self.completed_cloud_map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Mmdet3dNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        tmp_file = node._tmp_config_file
        node.destroy_node()
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
        rclpy.shutdown()


if __name__ == '__main__':
    main()