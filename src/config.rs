//! The `ComponentConfig` accessors this task needs.
//!
//! Kept local to the component: a component should not depend on an application's config
//! module.

use cu29::prelude::{ComponentConfig, CuResult};
use cu29::units::si::f64::{Frequency, Time};
use cu29::units::si::frequency::hertz;
use cu29::units::si::time::second;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_sensors::imu::ImuCalib;
use kornia_slam::mapping::culling::KeyframeCullingConfig;
use kornia_slam::tracking::KeyframePolicy;

use crate::error::VioError;
use crate::fusion::FusionConfig;
use crate::imu::InertialConfig;
use crate::parked::StationaryEdgeConfig;
use crate::stationary::StaticInitConfig;
use crate::wheel::{WheelOdomConfig, mat3_from_rows};

/// What the tracker does with its map after a sustained tracking loss: the `on_tracking_loss`
/// key, spelled `"keep_map"` or `"reset_map"`.
///
/// `reset_map` is the setting for a long-running graph, and the reason is worth stating because
/// the obvious improvement is a trap. Relocalization does recover onto an existing map, so
/// keeping it looks strictly better, and it does stop the visible re-bootstrap thrash. But
/// kornia-slam never culls keyframes, so without a `max_keyframes` bound the drop is the ONLY
/// bound on map size: with it off, an 18 h run on a Jetson Orin with an OAK-D at 640x400 reached
/// 46,649 keyframes and 1 GB RSS, and per-frame cost spiralled until the tracker ran at ~0.1 fps
/// while the rest of the graph kept running, so the stall was SILENT. Visible thrash beats a
/// silent stall.
///
/// `keep_map`, the default, is the setting for an offline replay that must accumulate one map
/// across losses, so its output can be compared frame for frame with another run of the tracker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnTrackingLoss {
    /// Re-bootstrap into the existing map.
    #[default]
    KeepMap,
    /// Drop the map and start a new tracker world.
    ResetMap,
}

/// What the tracker does when an insertion takes the map to `max_keyframes` and culling (if on)
/// cannot make room: the `at_keyframe_bound` key, spelled `"drop_map"` or `"evict_oldest"`.
///
/// Culling removes only redundant keyframes, and a robot driving through new space makes few:
/// its keyframes stay under ORB-SLAM's 90 % coverage, so the bound used to drop the map mid-drive.
/// `evict_oldest` keeps the map as a sliding window instead, until its landmark slots reach
/// `max_keyframes * orb_keypoints`: evicted landmarks keep their slots, so that bound still drops.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AtKeyframeBound {
    /// Drop the map and start a new tracker world.
    #[default]
    DropMap,
    /// Evict the oldest keyframes down to one below the bound.
    EvictOldest,
}

/// Reads the optional `at_keyframe_bound` policy, defaulting to [`AtKeyframeBound::DropMap`].
/// `evict_oldest` needs `max_keyframes`: that is what it acts at.
pub(crate) fn at_keyframe_bound(
    config: Option<&ComponentConfig>,
    max_keyframes: Option<usize>,
) -> CuResult<AtKeyframeBound> {
    let policy = config
        .map(|c| c.get_value::<AtKeyframeBound>(AT_KEYFRAME_BOUND))
        .transpose()?
        .flatten()
        .unwrap_or_default();
    if policy == AtKeyframeBound::EvictOldest && max_keyframes.is_none() {
        return Err(format!(
            "`{AT_KEYFRAME_BOUND}: evict_oldest` acts when the map reaches `{MAX_KEYFRAMES}`, \
             which is not set"
        )
        .into());
    }
    Ok(policy)
}

/// Reads the optional `on_tracking_loss` policy, defaulting to [`OnTrackingLoss::KeepMap`].
pub(crate) fn on_tracking_loss(config: Option<&ComponentConfig>) -> CuResult<OnTrackingLoss> {
    Ok(config
        .map(|c| c.get_value::<OnTrackingLoss>(ON_TRACKING_LOSS))
        .transpose()?
        .flatten()
        .unwrap_or_default())
}

/// Reads an optional POSITIVE integer key.
///
/// Zero is refused rather than accepted: as a `max_keyframes` it would drop the map on every
/// insertion, which reads as "VIO is broken", not as "the config said 0". A negative RON
/// literal fails the `u32` read here rather than wrapping to ~4.3e9 and silently disabling
/// whatever it bounds.
pub(crate) fn optional_positive_usize(
    config: Option<&ComponentConfig>,
    key: &'static str,
) -> CuResult<Option<usize>> {
    let raw = config.map(|c| c.get::<u32>(key)).transpose()?.flatten();
    match raw {
        Some(0) => Err(format!("`{key}` must be greater than 0").into()),
        other => Ok(other.map(|v| v as usize)),
    }
}

/// The `inertial` block of a `vio` component config.
///
/// One key, all-or-nothing, with `deny_unknown_fields` and no `serde(default)` on anything
/// physical: the extrinsic and the four noise densities have no defensible default, and a
/// silently-defaulted one is not a degraded answer but a plausible, wrong map. Omitting the
/// whole `inertial` key leaves the tracker on the stereo-only path it ships with.
///
/// ```ron
/// ( id: "vio",
///   type: "cu_kornia_vio::StereoVio",
///   config: {
///     "inertial": (
///       // R_BC, ROW-major, mapping camera axes into IMU/body axes: X_body = R_BC * X_cam.
///       // "cam" is the RECTIFIED left camera — the frame the tracker's own poses live in.
///       t_bc_rotation: [1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0],
///       // Lever arm, metres, IMU origin relative to that camera. NOT optional and NOT
///       // zero-by-default: a rotation-only extrinsic is a silent lever-arm error, which
///       // injects a spurious specific force w x (w x e) that grows with angular rate.
///       t_bc_translation: [0.0, 0.0, 0.0],
///       gyro_noise: 0.0,        // rad/s/sqrt(Hz)
///       accel_noise: 0.0,       // m/s^2/sqrt(Hz)
///       gyro_bias_noise: 0.0,   // rad/s^2/sqrt(Hz)
///       accel_bias_noise: 0.0,  // m/s^3/sqrt(Hz)
///       rate_hz: 200.0,
///     ),
///   },
/// ),
/// ```
///
/// The values above are placeholders and every one of them is rejected: the densities must be
/// strictly positive, and an identity `t_bc_rotation` would only ever be correct on a device
/// where the IMU and the rectified camera share axes exactly. The device's own `T_BC` cannot be
/// relied on (the OAK-D's `imu_to_camera_extrinsics` query can fail on a given unit), so the key
/// must carry a measured value or be absent. There is deliberately no default; see
/// `t_bc_rotation`.
///
/// This is the WIRE form, and crate-private for that reason: its numbers are plain `f64` because
/// that is what a RON file holds. [`InertialRon::to_config`] is the one place they become the
/// unit-typed [`InertialConfig`] the tracker takes.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct InertialRon {
    /// `R_BC`, ROW-major. See the type doc for the convention and why it is not column-major.
    ///
    /// NO DEFAULT. RTAB-Map's hardcoded first-generation OAK-D transform is the obvious
    /// candidate, and it fails the gravity cross-check: measured on a Jetson Orin with an OAK-D
    /// at 640x400, a static accelerometer read of `(-1.34, +9.56, +0.38)` mapped through it lands
    /// on camera -x, 82 deg from where an upright module puts gravity (camera -y), on a unit whose
    /// decoded picture is upright and landscape, so the module is not rolled. A published
    /// per-board value that fails the one free check is worse than no value: it looks
    /// calibrated. The rotation comes from a measurement of the unit in use or it is absent.
    pub t_bc_rotation: [f64; 9],
    /// `t_BC` in metres. No default either; a zero lever arm is a silent error under rotation.
    pub t_bc_translation: [f64; 3],
    /// Gyroscope white noise density, rad/s/sqrt(Hz).
    pub gyro_noise: f64,
    /// Accelerometer white noise density, m/s^2/sqrt(Hz).
    pub accel_noise: f64,
    /// Gyroscope bias random walk, rad/s^2/sqrt(Hz).
    pub gyro_bias_noise: f64,
    /// Accelerometer bias random walk, m/s^3/sqrt(Hz).
    pub accel_bias_noise: f64,
    /// Configured sample rate, Hz. The denominator of the interval coverage check.
    pub rate_hz: f64,
    /// Whether the inertial path may move a published pose. It gates two things, and both
    /// matter: the switch of local BA to the 15-DOF inertial solve once initialized, and the
    /// COMMIT of an accepted initialization — `apply_initialization` scales and rotates the
    /// whole map onto the estimated gravity and bumps the world generation, which goes on the
    /// wire as `reset_epoch` and makes every consumer wipe its trail and re-derive its anchor.
    /// With this off, the factors are still built, `try_initialize` still runs and its
    /// acceptance is still counted (`init_accepted` with `initialized=false` in the stop line)
    /// and logged with the gravity and bias it found, but nothing is applied: the map, the
    /// world generation and the bias stay where the visual path left them.
    ///
    /// Defaults to `false` here even though [`InertialConfig::new`] defaults it to `true`:
    /// reading a config to turn the inertial path ON is already the risky step, and the
    /// numbers above are what a user needs to decide whether the extrinsic is right
    /// BEFORE it is allowed to rotate the world. Gating only the local-BA switch would not be
    /// enough: a yaw-unverified extrinsic would still re-register the world at each of the three
    /// staged inertial-initialization refinements (Campos et al., ORB-SLAM3, IEEE T-RO 2021),
    /// three published-pose jumps in the first 15 s of every run.
    #[serde(default)]
    pub enable_inertial_ba: bool,
    /// Initialize while parked, from a still window; see [`StaticInitRon`]. Absent, the
    /// inertial path waits for the robot to move, as it always has.
    #[serde(default)]
    pub static_init: Option<StaticInitRon>,
    /// Largest gyro bias norm the tracker believes, rad/s; see [`crate::BiasLimit`]. Absent is
    /// [`crate::BiasLimit::DEFAULT_GYRO_RAD_S`].
    #[serde(default)]
    pub gyro_bias_limit: Option<f64>,
    /// Largest accelerometer bias norm, m/s^2. Absent is
    /// [`crate::BiasLimit::DEFAULT_ACCEL_M_S2`].
    #[serde(default)]
    pub accel_bias_limit: Option<f64>,
}

/// The `static_init` block inside `inertial`.
///
/// ```ron
/// static_init: (
///   enabled: true,
///   // Optional; each defaults to the value derived in `StaticInitConfig::from_noise`.
///   window_s: 1.2,            // contiguous IMU edges the still window must span
///   accel_rms_max: 0.186,     // m/s^2, accelerometer vector spread
///   gyro_rms_max: 0.0118,     // rad/s, gyroscope vector spread
///   gyro_mean_max: 0.02,      // rad/s, largest believable gyro bias; capped at gyro_bias_limit
///   gravity_tolerance: 0.3,   // m/s^2, | |mean accel| - 9.81 |
/// ),
/// ```
///
/// Off unless `enabled`. The spread ceilings default to four times what the block's own noise
/// densities predict at `rate_hz`, so a config with honest densities needs nothing else. Like the
/// rest of `inertial`, it moves the world only with `enable_inertial_ba` on: with it off an
/// accepted still window is counted and logged, not applied.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StaticInitRon {
    /// Off unless set.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub window_s: Option<f64>,
    #[serde(default)]
    pub accel_rms_max: Option<f64>,
    #[serde(default)]
    pub gyro_rms_max: Option<f64>,
    #[serde(default)]
    pub gyro_mean_max: Option<f64>,
    #[serde(default)]
    pub gravity_tolerance: Option<f64>,
}

impl StaticInitRon {
    /// The config over `defaults`, or `None` when disabled. Validated either way.
    fn to_config(&self, defaults: StaticInitConfig) -> Result<Option<StaticInitConfig>, VioError> {
        let mut cfg = defaults;
        for (field, value, slot) in [
            (
                "static_init.accel_rms_max",
                self.accel_rms_max,
                &mut cfg.accel_rms_max,
            ),
            (
                "static_init.gyro_rms_max",
                self.gyro_rms_max,
                &mut cfg.gyro_rms_max,
            ),
            (
                "static_init.gyro_mean_max",
                self.gyro_mean_max,
                &mut cfg.gyro_mean_max,
            ),
            (
                "static_init.gravity_tolerance",
                self.gravity_tolerance,
                &mut cfg.gravity_tolerance,
            ),
        ] {
            if let Some(v) = value {
                *slot = v;
            }
            if !slot.is_finite() || *slot <= 0.0 {
                return Err(VioError::InertialParamInvalid {
                    field,
                    value: *slot,
                });
            }
        }
        if let Some(v) = self.window_s {
            if !v.is_finite() || v <= 0.0 {
                return Err(VioError::InertialParamInvalid {
                    field: "static_init.window_s",
                    value: v,
                });
            }
            cfg.window = cu29::units::si::f64::Time::new::<cu29::units::si::time::second>(v);
        }
        Ok(self.enabled.then_some(cfg))
    }
}

impl InertialRon {
    /// Converts to the tracker's config, validating the extrinsic and the densities.
    ///
    /// The rotation arrives ROW-major because that is how a human writes a matrix, and
    /// `Mat3F64` is glam-backed COLUMN-major. The transpose below is therefore mandatory —
    /// and it is an easy mistake to make, because a transposed rotation
    /// is still a valid rotation and nothing downstream can report it. It is spelled as ONE
    /// named operation rather than nine hand-placed indices, because the indices are where the
    /// mistake gets made; `test_rotation_is_read_row_major` pins it either way.
    ///
    /// # Errors
    ///
    /// [`VioError::InertialExtrinsicNotRigid`] or [`VioError::InertialParamInvalid`].
    pub fn to_config(&self) -> Result<InertialConfig, VioError> {
        // Row-major on the wire, glam is column-major: reading the rows as columns gives R^T.
        let rotation = Mat3F64::from_cols_array(&self.t_bc_rotation).transpose();
        let translation = Vec3F64::from_array(self.t_bc_translation);
        let calib = ImuCalib {
            gyro_noise: self.gyro_noise,
            accel_noise: self.accel_noise,
            gyro_bias_noise: self.gyro_bias_noise,
            accel_bias_noise: self.accel_bias_noise,
        };
        let mut cfg = InertialConfig::new(
            Pose3d::from_rt(rotation, translation),
            calib,
            Frequency::new::<hertz>(self.rate_hz),
        )?;
        cfg.enable_inertial_ba = self.enable_inertial_ba;
        for (field, value, slot) in [
            (
                "gyro_bias_limit",
                self.gyro_bias_limit,
                &mut cfg.bias_limit.gyro_rad_s,
            ),
            (
                "accel_bias_limit",
                self.accel_bias_limit,
                &mut cfg.bias_limit.accel_m_s2,
            ),
        ] {
            if let Some(value) = value {
                if !value.is_finite() || value <= 0.0 {
                    return Err(VioError::InertialParamInvalid { field, value });
                }
                *slot = value;
            }
        }
        if let Some(block) = &self.static_init {
            cfg.static_init = block.to_config(StaticInitConfig::from_noise(
                &cfg.calib,
                cfg.gates.nominal_rate,
            ))?;
        }
        // A still window's mean gyro rate IS its bias estimate, and one over the gyro bias limit
        // is always refused after the window is accepted. Capped here, the window itself is
        // refused for its gyro mean, which names the cause, instead of a found window refused
        // on every `init_retry`.
        if let Some(s) = cfg.static_init.as_mut() {
            s.gyro_mean_max = s.gyro_mean_max.min(cfg.bias_limit.gyro_rad_s);
        }
        Ok(cfg)
    }
}

/// Reads the optional `inertial` block.
///
/// A malformed or partial block is a hard error, not a fallback to the stereo-only path: a
/// config that carries the key means to turn the inertial path on, and starting visual-only
/// because a field was misspelled is the silent failure this whole module is written against.
pub(crate) fn optional_inertial(config: Option<&ComponentConfig>) -> CuResult<Option<InertialRon>> {
    Ok(config
        .map(|c| c.get_value::<InertialRon>(INERTIAL))
        .transpose()?
        .flatten())
}

/// The `wheel_odom` block of a `vio` component config.
///
/// ```ron
/// "wheel_odom": (
///   enabled: true,
///   // Camera pose in the base frame (the URDF base -> rectified left camera), ROW-major.
///   // The fallback until a `CameraMount` arrives on the bus; a fixed mount needs nothing else.
///   cam_in_base_rotation: [0.0, 0.0, 1.0,  -1.0, 0.0, 0.0,  0.0, -1.0, 0.0],
///   cam_in_base_translation: [0.1, 0.0, 0.9],
///   translation_noise: 0.02,   // m per sqrt(m) of path
///   rotation_noise: 0.05,      // rad per sqrt(rad) turned
///   // Optional:
///   z_sigma: 0.01,             // m, planar stiffness on z
///   tilt_sigma: 0.01,          // rad, planar stiffness on roll and pitch
///   huber_chi2: 12.59,         // 6-DOF chi-square, 95 %; absent keeps the factor quadratic
///   max_gap_s: 0.5,            // longest hole between readings an interval may span
///   heading_noise_per_m: 0.0,  // rad per sqrt(m) driven: heading drift on a straight run
///   heading_scale_sigma_per_m: 0.0, // rad/m, systematic (left/right wheel radius mismatch)
///   z_slope: 0.0,              // m per m driven, added to z_sigma in quadrature
///   clock_sigma_s: 0.005,      // s, frame-stamp uncertainty: v * sigma at each end
///   min_translation_sigma_m: 0.001, // m, floor on x/y of every interval
///   min_heading_sigma_rad: 0.001,   // rad, floor on heading of every interval
/// ),
/// ```
///
/// Absent, or with `enabled: false` (the default), the tracker behaves exactly as without the
/// block. Present, the mount and the two noise figures are required: there is no defensible
/// default for where a camera sits or how good a robot's wheels are, and a wrong one is not an
/// error but a confidently bent trajectory. The planar stiffness defaults to kornia-slam's
/// (1 cm, 0.01 rad), itself unmeasured. The distance terms default to zero, so a block without
/// them keeps the noise model it had, floors included; only the timing term (on unless
/// `clock_sigma_s: 0.0`) and the propagated cross-covariance differ.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WheelOdomRon {
    /// Off unless set, so a block can sit in a graph for an A/B without acting.
    #[serde(default)]
    pub enabled: bool,
    /// `R` of the camera in the base, ROW-major: `X_base = R X_cam + t`.
    pub cam_in_base_rotation: [f64; 9],
    /// `t` of the camera in the base, metres.
    pub cam_in_base_translation: [f64; 3],
    /// m per sqrt(m) of path.
    pub translation_noise: f64,
    /// rad per sqrt(rad) turned.
    pub rotation_noise: f64,
    #[serde(default)]
    pub z_sigma: Option<f64>,
    #[serde(default)]
    pub tilt_sigma: Option<f64>,
    #[serde(default)]
    pub huber_chi2: Option<f64>,
    #[serde(default)]
    pub max_gap_s: Option<f64>,
    #[serde(default)]
    pub heading_noise_per_m: Option<f64>,
    #[serde(default)]
    pub heading_scale_sigma_per_m: Option<f64>,
    #[serde(default)]
    pub z_slope: Option<f64>,
    #[serde(default)]
    pub clock_sigma_s: Option<f64>,
    #[serde(default)]
    pub min_translation_sigma_m: Option<f64>,
    #[serde(default)]
    pub min_heading_sigma_rad: Option<f64>,
    #[serde(default)]
    pub parked_keyframe_interval_s: Option<f64>,
}

impl WheelOdomRon {
    /// The tracker's config, or `None` when the block is disabled. Validated either way, so a
    /// disabled block with a typo in it fails at startup rather than on the day it is enabled.
    pub fn to_config(&self) -> Result<Option<WheelOdomConfig>, VioError> {
        let cam_in_base = Pose3d::from_rt(
            mat3_from_rows(&self.cam_in_base_rotation),
            Vec3F64::from_array(self.cam_in_base_translation),
        );
        let mut cfg =
            WheelOdomConfig::new(cam_in_base, self.translation_noise, self.rotation_noise)?;
        if let Some(v) = self.z_sigma {
            cfg.z_sigma = v;
        }
        if let Some(v) = self.tilt_sigma {
            cfg.tilt_sigma = v;
        }
        cfg.huber_chi2 = self.huber_chi2;
        if let Some(v) = self.max_gap_s {
            // `from_secs_f64` would panic on a negative; `validate` refuses it instead.
            cfg.max_gap = cu29::units::si::f64::Time::new::<cu29::units::si::time::second>(v);
        }
        if let Some(v) = self.heading_noise_per_m {
            cfg.heading_noise_per_m = v;
        }
        if let Some(v) = self.heading_scale_sigma_per_m {
            cfg.heading_scale_sigma_per_m = v;
        }
        if let Some(v) = self.z_slope {
            cfg.z_slope = v;
        }
        if let Some(v) = self.clock_sigma_s {
            cfg.clock_sigma = cu29::units::si::f64::Time::new::<cu29::units::si::time::second>(v);
        }
        if let Some(v) = self.min_translation_sigma_m {
            cfg.min_translation_sigma_m = v;
        }
        if let Some(v) = self.min_heading_sigma_rad {
            cfg.min_heading_sigma_rad = v;
        }
        cfg.parked_keyframe_interval = self
            .parked_keyframe_interval_s
            .map(cu29::units::si::f64::Time::new::<cu29::units::si::time::second>);
        cfg.validate()?;
        Ok(self.enabled.then_some(cfg))
    }
}

/// Reads the optional `wheel_odom` block; malformed is a hard error, like `inertial`.
pub(crate) fn optional_wheel_odom(
    config: Option<&ComponentConfig>,
) -> CuResult<Option<WheelOdomRon>> {
    Ok(config
        .map(|c| c.get_value::<WheelOdomRon>(WHEEL_ODOM))
        .transpose()?
        .flatten())
}

/// The `per_frame_fusion` block of a `vio` component config.
///
/// ```ron
/// "per_frame_fusion": (
///   enabled: true,
///   // Optional: the vision-vs-prediction gate as a 6-DOF chi-square value, scaled to the
///   // frame's actual degrees of freedom. Default 22.458, the 99.9 % quantile.
///   chi2_gate: 22.458,
///   // Optional: the camera-to-base mount's uncertainty, metres and radians. Acts only through
///   // the motion (zero on a parked robot): a turn moves the wheel factor's camera prediction
///   // by `|dtheta| * sigma`. Defaults 0.03 and 0.035.
///   extrinsic_sigma_m: 0.03,
///   extrinsic_sigma_rad: 0.035,
/// ),
/// ```
///
/// Absent, or `enabled: false` (the default), the published pose is PnP's, as it always was.
/// Enabled, each tracked frame's pose is optimized against its map matches, the IMU from the last
/// keyframe and the wheels from the last keyframe together, and a frame whose vision disagrees
/// with the motion sensors past the gate publishes their prediction instead; see
/// [`crate::fusion`]. It acts only through the `inertial` and `wheel_odom` blocks: with neither
/// there is nothing to fuse and the tracker is unchanged.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FusionRon {
    /// Off unless set.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub chi2_gate: Option<f64>,
    #[serde(default)]
    pub extrinsic_sigma_m: Option<f64>,
    #[serde(default)]
    pub extrinsic_sigma_rad: Option<f64>,
}

impl FusionRon {
    /// The tracker's config, or `None` when disabled. Validated either way.
    pub fn to_config(&self) -> Result<Option<FusionConfig>, VioError> {
        let mut cfg = FusionConfig::default();
        if let Some(g) = self.chi2_gate {
            cfg.chi2_gate = g;
        }
        if let Some(v) = self.extrinsic_sigma_m {
            cfg.extrinsic_sigma_m = v;
        }
        if let Some(v) = self.extrinsic_sigma_rad {
            cfg.extrinsic_sigma_rad = v;
        }
        cfg.validate()?;
        Ok(self.enabled.then_some(cfg))
    }
}

/// Reads and validates the optional `per_frame_fusion` block; malformed is a hard error.
pub(crate) fn optional_fusion(config: Option<&ComponentConfig>) -> CuResult<Option<FusionConfig>> {
    let ron = config
        .map(|c| c.get_value::<FusionRon>(PER_FRAME_FUSION))
        .transpose()?
        .flatten();
    match ron {
        Some(r) => r.to_config().map_err(|e| e.to_string().into()),
        None => Ok(None),
    }
}

/// The `stationary_edges` block of a `vio` component config: parked keyframes
/// ([`crate::parked`]).
///
/// ```ron
/// "stationary_edges": (
///   enabled: true,              // default false
///   // Each optional; the defaults are kornia-slam's `StationaryFactor` ones.
///   translation_sigma_m: 0.002, // zero relative camera translation
///   rotation_sigma_rad: 0.002,  // zero relative camera rotation
///   velocity_sigma_m_s: 0.01,   // zero velocity at each end
///   huber_chi2: 21.03,          // on the 12-row pose and velocity chi-square
///   min_span_s: 1.0,            // still time under which a keyframe keeps its IMU edge
/// ),
/// ```
///
/// Absent, or `enabled: false` (the default), nothing changes. Enabled, once the inertial path
/// is initialized and no refinement is pending, a parked robot withholds the keyframes its policy
/// asks for on the time bound (no interval: `wheel_odom.parked_keyframe_interval_s` applies only
/// before that), still inserts on tracking-quality triggers, and links keyframes of one stop by
/// a stationary factor that culling merges without a span limit. It needs `wheel_odom` enabled
/// and `inertial` with `enable_inertial_ba`: refused at startup otherwise. Validated even when
/// disabled.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StationaryEdgesRon {
    /// Off unless set.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub translation_sigma_m: Option<f64>,
    #[serde(default)]
    pub rotation_sigma_rad: Option<f64>,
    #[serde(default)]
    pub velocity_sigma_m_s: Option<f64>,
    #[serde(default)]
    pub huber_chi2: Option<f64>,
    #[serde(default)]
    pub min_span_s: Option<f64>,
}

impl StationaryEdgesRon {
    /// The tracker's config, or `None` when disabled. Validated either way.
    pub fn to_config(&self) -> Result<Option<StationaryEdgeConfig>, VioError> {
        let mut cfg = StationaryEdgeConfig::new();
        if let Some(v) = self.translation_sigma_m {
            cfg.translation_sigma_m = v;
        }
        if let Some(v) = self.rotation_sigma_rad {
            cfg.rotation_sigma_rad = v;
        }
        if let Some(v) = self.velocity_sigma_m_s {
            cfg.velocity_sigma_m_s = v;
        }
        if let Some(v) = self.huber_chi2 {
            cfg.huber_chi2 = Some(v);
        }
        if let Some(v) = self.min_span_s {
            // `validate` refuses a NaN or an infinity carried through here.
            cfg.min_span = Time::new::<second>(v);
        }
        cfg.validate()?;
        Ok(self.enabled.then_some(cfg))
    }
}

/// The `motion_veto` block of a `vio` component config: the hard motion veto ([`crate::veto`]).
///
/// ```ron
/// "motion_veto": (
///   enabled: true,              // default true: absent is ON with these defaults
///   rotation_deg: 10.0,         // vision vs the gyro since the last published pose
///   translation_m: 0.3,         // vision vs the wheels (camera centre)
///   rotation_deg_per_s: 2.0,    // bound growth per second since the last trusted pose
///   translation_m_per_s: 0.1,
///   motion_fraction: 0.2,       // bound growth per radian turned / metre driven since then
///   recovery_frames: 5,         // agreeing frames in a row before vision is accepted again
///   recovery_timeout_s: 3.0,    // then the map is re-anchored at the dead-reckoned pose
///   rotation_window_s: 5.0,     // cumulative check: rotation against trusted poses this old
///   translation_window_s: 0.5,  // and translation (short: frozen vision and wheel spin look alike)
/// ),
/// ```
///
/// It acts only through the gyro (`inertial`) and the wheels (`wheel_odom`): with neither it
/// checks nothing. Validated even when disabled.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MotionVetoRon {
    /// On unless set false.
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub rotation_deg: Option<f64>,
    #[serde(default)]
    pub translation_m: Option<f64>,
    #[serde(default)]
    pub rotation_deg_per_s: Option<f64>,
    #[serde(default)]
    pub translation_m_per_s: Option<f64>,
    #[serde(default)]
    pub motion_fraction: Option<f64>,
    #[serde(default)]
    pub recovery_frames: Option<u32>,
    #[serde(default)]
    pub recovery_timeout_s: Option<f64>,
    #[serde(default)]
    pub rotation_window_s: Option<f64>,
    #[serde(default)]
    pub translation_window_s: Option<f64>,
}

fn default_true() -> bool {
    true
}

impl MotionVetoRon {
    /// The tracker's config, or `None` when disabled. Validated either way.
    pub fn to_config(&self) -> Result<Option<crate::veto::MotionVetoConfig>, VioError> {
        let mut cfg = crate::veto::MotionVetoConfig::default();
        if let Some(v) = self.rotation_deg {
            cfg.rotation_rad = v.to_radians();
        }
        if let Some(v) = self.translation_m {
            cfg.translation_m = v;
        }
        if let Some(v) = self.rotation_deg_per_s {
            cfg.rotation_rad_per_s = v.to_radians();
        }
        if let Some(v) = self.translation_m_per_s {
            cfg.translation_m_per_s = v;
        }
        if let Some(v) = self.motion_fraction {
            cfg.motion_fraction = v;
        }
        if let Some(v) = self.recovery_frames {
            cfg.recovery_frames = v;
        }
        if let Some(v) = self.recovery_timeout_s {
            cfg.recovery_timeout_s = v;
        }
        if let Some(v) = self.rotation_window_s {
            cfg.rotation_window_s = v;
        }
        if let Some(v) = self.translation_window_s {
            cfg.translation_window_s = v;
        }
        cfg.validate()?;
        Ok(self.enabled.then_some(cfg))
    }
}

/// Reads and validates the optional `motion_veto` block: absent is ON with the defaults
/// (validated), malformed is a hard error.
pub(crate) fn optional_motion_veto(
    config: Option<&ComponentConfig>,
) -> CuResult<Option<crate::veto::MotionVetoConfig>> {
    let ron = config
        .map(|c| c.get_value::<MotionVetoRon>(MOTION_VETO))
        .transpose()?
        .flatten();
    match ron {
        Some(r) => r
            .to_config()
            .map_err(|e| format!("`{MOTION_VETO}`: {e}").into()),
        None => Ok(Some(crate::veto::MotionVetoConfig::default())),
    }
}

/// Reads and validates the optional `stationary_edges` block; malformed is a hard error.
pub(crate) fn optional_stationary_edges(
    config: Option<&ComponentConfig>,
) -> CuResult<Option<StationaryEdgeConfig>> {
    let ron = config
        .map(|c| c.get_value::<StationaryEdgesRon>(STATIONARY_EDGES))
        .transpose()?
        .flatten();
    match ron {
        Some(r) => r
            .to_config()
            .map_err(|e| format!("`{STATIONARY_EDGES}`: {e}").into()),
        None => Ok(None),
    }
}

/// The `keyframe_policy` block of a `vio` component config: when a keyframe is inserted.
///
/// ```ron
/// "keyframe_policy": (
///   // Each optional; absent keeps kornia-slam's `KeyframePolicy::default()` (3, 8, 0.6).
///   min_frames_between: 3,   // frames before another keyframe is considered
///   max_frames_between: 8,   // a keyframe is forced at this gap
///   ref_ratio: 0.6,          // insert when tracked < ratio * the reference keyframe's points
///   min_parallax_px: 10.0,   // insert at this median rotation-compensated parallax (VINS-Mono)
///   // ORB-SLAM3 / VINS-Mono insertion instead of a frame count: no `max_frames_between`, a
///   // keyframe is earned by parallax or a weak reference. Needs `min_parallax_px`, and with the
///   // `inertial` block on, `max_interval_s` too: an IMU edge must stay short enough to fit the
///   // ring and for BA to converge on it.
///   motion_only: false,
///   max_interval_s: 1.0,     // a keyframe is forced this long after the last one
/// ),
/// ```
///
/// Absent, the policy is the default one, as it always was. The parked gate
/// (`wheel_odom.parked_keyframe_interval_s`) still withholds what this asks for on a parked robot.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct KeyframePolicyRon {
    #[serde(default)]
    pub min_frames_between: Option<u32>,
    #[serde(default)]
    pub max_frames_between: Option<u32>,
    #[serde(default)]
    pub ref_ratio: Option<f64>,
    #[serde(default)]
    pub min_parallax_px: Option<f64>,
    #[serde(default)]
    pub motion_only: bool,
    #[serde(default)]
    pub max_interval_s: Option<f64>,
}

/// What [`KeyframePolicyRon`] configures: the policy, and the time bound beside it.
#[derive(Debug, Clone, Default)]
pub(crate) struct KeyframeInsertion {
    pub policy: KeyframePolicy,
    pub max_interval: Option<Time>,
}

impl KeyframePolicyRon {
    /// The validated policy over kornia-slam's default.
    pub fn to_config(&self) -> CuResult<KeyframeInsertion> {
        let mut policy = KeyframePolicy::default();
        if let Some(n) = self.min_frames_between {
            policy.min_frames_between = n as usize;
        }
        match (self.motion_only, self.max_frames_between) {
            (true, Some(_)) => {
                return Err("`keyframe_policy.motion_only` has no frame bound; drop \
                            `max_frames_between`"
                    .into());
            }
            (true, None) => policy.max_frames_between = usize::MAX,
            (false, Some(0)) => {
                return Err("`keyframe_policy.max_frames_between` must be greater than 0".into());
            }
            (false, Some(n)) => policy.max_frames_between = n as usize,
            (false, None) => {}
        }
        if policy.max_frames_between < policy.min_frames_between {
            return Err(format!(
                "`keyframe_policy.max_frames_between` ({}) is below `min_frames_between` ({})",
                policy.max_frames_between, policy.min_frames_between
            )
            .into());
        }
        if let Some(r) = self.ref_ratio {
            if !(r.is_finite() && r > 0.0 && r <= 1.0) {
                return Err(
                    format!("`keyframe_policy.ref_ratio` must be in (0, 1], got {r}").into(),
                );
            }
            policy.ref_ratio = r;
        }
        if let Some(px) = self.min_parallax_px {
            if !(px.is_finite() && px > 0.0) {
                return Err(format!(
                    "`keyframe_policy.min_parallax_px` must be finite and greater than 0, got {px}"
                )
                .into());
            }
            policy.min_parallax_px = Some(px);
        }
        // Without parallax nothing but a weak reference would ever earn a keyframe: a robot
        // driving down a corridor would extend one keyframe until tracking failed.
        if self.motion_only && policy.min_parallax_px.is_none() {
            return Err("`keyframe_policy.motion_only` needs `min_parallax_px`".into());
        }
        let max_interval = match self.max_interval_s {
            Some(s) if !(s.is_finite() && s > 0.0) => {
                return Err(format!(
                    "`keyframe_policy.max_interval_s` must be finite and greater than 0, got {s}"
                )
                .into());
            }
            other => other.map(Time::new::<second>),
        };
        Ok(KeyframeInsertion {
            policy,
            max_interval,
        })
    }
}

/// Reads and validates the optional `keyframe_policy` block; absent is the default policy.
pub(crate) fn keyframe_insertion(config: Option<&ComponentConfig>) -> CuResult<KeyframeInsertion> {
    config
        .map(|c| c.get_value::<KeyframePolicyRon>(KEYFRAME_POLICY))
        .transpose()?
        .flatten()
        .unwrap_or_default()
        .to_config()
}

/// The `keyframe_culling` block of a `vio` component config: ORB-SLAM's redundant-keyframe
/// culling (`kornia_slam::mapping::culling::cull_redundant_keyframes`), run when the map reaches
/// `max_keyframes`, so the map is dropped only when culling cannot make room.
///
/// ```ron
/// "keyframe_culling": (
///   enabled: true,
///   // Each optional; absent keeps `KeyframeCullingConfig::default()`.
///   redundant_ratio: 0.9,        // fraction of a keyframe's landmarks seen elsewhere
///   min_other_observers: 3,      // other keyframes that must see such a landmark
///   scale_level_tolerance: 1,    // pyramid levels coarser that still count (0: the paper's)
///   protect_newest: 3,           // newest keyframes never culled (at least 1)
///   min_keyframes: 21,           // no culling at or below this many; below `max_keyframes`
///   max_merged_imu_span_s: 3.0,  // longest IMU edge a removal may merge into
///   max_removals: 1000,          // most keyframes one pass removes
/// ),
/// ```
///
/// Absent, or `enabled: false` (the default), the map bound drops the map as it always did.
/// Enabled, it needs `max_keyframes`: that is what it acts at. It bounds keyframes only: retired
/// landmarks keep their slot, so the map is still dropped once it holds
/// `max_keyframes * orb_keypoints` landmark slots. With the `inertial` block on, the keyframe
/// spacing while parked plus 0.25 s must be at most half `max_merged_imu_span_s`: a parked
/// keyframe sits between two such IMU edges, each up to a frame longer than the interval, and a
/// merge longer than the bound is kept. That spacing is the longer of
/// `wheel_odom.parked_keyframe_interval_s` and the policy's own bound, the sooner of
/// `keyframe_policy.max_interval_s` and `max_frames_between` frames at 140 ms each.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct KeyframeCullingRon {
    /// Off unless set.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub redundant_ratio: Option<f64>,
    #[serde(default)]
    pub min_other_observers: Option<u32>,
    #[serde(default)]
    pub scale_level_tolerance: Option<u8>,
    #[serde(default)]
    pub protect_newest: Option<u32>,
    #[serde(default)]
    pub min_keyframes: Option<u32>,
    #[serde(default)]
    pub max_merged_imu_span_s: Option<f64>,
    #[serde(default)]
    pub max_removals: Option<u32>,
}

impl KeyframeCullingRon {
    /// The culling config, or `None` when disabled. Validated either way, so a disabled block
    /// with a bad value fails at startup rather than on the day it is enabled.
    pub fn to_config(&self) -> CuResult<Option<KeyframeCullingConfig>> {
        let mut cfg = KeyframeCullingConfig::default();
        if let Some(r) = self.redundant_ratio {
            if !(r.is_finite() && r > 0.0 && r <= 1.0) {
                return Err(format!(
                    "`keyframe_culling.redundant_ratio` must be in (0, 1], got {r}"
                )
                .into());
            }
            cfg.redundant_ratio = r;
        }
        for (field, value, slot) in [
            (
                "min_other_observers",
                self.min_other_observers,
                &mut cfg.min_other_observers,
            ),
            (
                "protect_newest",
                self.protect_newest,
                &mut cfg.protect_newest,
            ),
            ("min_keyframes", self.min_keyframes, &mut cfg.min_keyframes),
            ("max_removals", self.max_removals, &mut cfg.max_removals),
        ] {
            match value {
                Some(0) => {
                    return Err(format!("`keyframe_culling.{field}` must be greater than 0").into());
                }
                Some(n) => *slot = n as usize,
                None => {}
            }
        }
        if let Some(t) = self.scale_level_tolerance {
            cfg.scale_level_tolerance = t;
        }
        if let Some(s) = self.max_merged_imu_span_s {
            if !(s.is_finite() && s > 0.0) {
                return Err(format!(
                    "`keyframe_culling.max_merged_imu_span_s` must be finite and greater than 0, \
                     got {s}"
                )
                .into());
            }
            cfg.max_merged_imu_span_sec = s;
        }
        Ok(self.enabled.then_some(cfg))
    }
}

/// Reads and validates the optional `keyframe_culling` block against `max_keyframes`, the bound
/// it acts at.
pub(crate) fn keyframe_culling(
    config: Option<&ComponentConfig>,
    max_keyframes: Option<usize>,
) -> CuResult<Option<KeyframeCullingConfig>> {
    let Some(ron) = config
        .map(|c| c.get_value::<KeyframeCullingRon>(KEYFRAME_CULLING))
        .transpose()?
        .flatten()
    else {
        return Ok(None);
    };
    let Some(cfg) = ron.to_config()? else {
        return Ok(None);
    };
    match max_keyframes {
        None => Err(format!(
            "`keyframe_culling` runs when the map reaches `{MAX_KEYFRAMES}`, which is not set"
        )
        .into()),
        Some(cap) if cfg.min_keyframes >= cap => Err(format!(
            "`keyframe_culling.min_keyframes` ({}) must be below `{MAX_KEYFRAMES}` ({cap}), or \
             culling can never make room",
            cfg.min_keyframes
        )
        .into()),
        Some(cap) if cfg.protect_newest >= cap => Err(format!(
            "`keyframe_culling.protect_newest` ({}) must be below `{MAX_KEYFRAMES}` ({cap}), or \
             culling can never make room",
            cfg.protect_newest
        )
        .into()),
        Some(_) => Ok(Some(cfg)),
    }
}

/// Reads an optional finite, strictly-positive float key.
pub(crate) fn optional_positive_f64(
    config: Option<&ComponentConfig>,
    key: &'static str,
) -> CuResult<Option<f64>> {
    let raw = config.map(|c| c.get::<f64>(key)).transpose()?.flatten();
    match raw {
        Some(v) if !v.is_finite() || v <= 0.0 => {
            Err(format!("`{key}` must be finite and greater than 0, got {v}").into())
        }
        other => Ok(other),
    }
}

/// The config keys `StereoVio::new` reads, each spelled ONCE.
///
/// The macro is what makes the list trustworthy: a second hand-typed copy of the strings could
/// only be kept honest by a test scraping the reads out of the source, which is fragile under
/// line wrapping and formatting. One declaration states it instead.
///
/// Both halves come from the same source text, so the list cannot lose an entry the reads
/// still use. That direction is the dangerous one: [`deny_unknown_keys`] REFUSES anything unlisted,
/// so a key read-but-unlisted gets a correct RON rejected at startup, with the real key named as
/// the unknown one.
macro_rules! config_keys {
    ($($name:ident => $lit:literal),+ $(,)?) => {
        $(#[doc = concat!("The `", $lit, "` config key.")]
        pub(crate) const $name: &str = $lit;)+
        /// Every key [`StereoVio`](crate::StereoVio) reads from its RON config, in declaration
        /// order. Public and read-only, so an application can assert the list its own configs
        /// are written against.
        pub const CONFIG_KEYS: &[&str] = &[$($lit),+];
    };
}

config_keys! {
    ON_TRACKING_LOSS => "on_tracking_loss",
    MAX_KEYFRAMES => "max_keyframes",
    AT_KEYFRAME_BOUND => "at_keyframe_bound",
    ORB_KEYPOINTS => "orb_keypoints",
    SEARCH_RADIUS_PX => "search_radius_px",
    MAX_COVISIBLE_KEYFRAMES => "max_covisible_keyframes",
    PNP_LM_ITERATIONS => "pnp_lm_iterations",
    INERTIAL => "inertial",
    LANDMARK_BUFFERS => "landmark_buffers",
    WHEEL_ODOM => "wheel_odom",
    PER_FRAME_FUSION => "per_frame_fusion",
    KEYFRAME_POLICY => "keyframe_policy",
    KEYFRAME_CULLING => "keyframe_culling",
    STATIONARY_EDGES => "stationary_edges",
    MOTION_VETO => "motion_veto",
}

/// Refuses a config key `task` does not read.
///
/// `cu29` stores a component's config as `HashMap<String, Value>` and hands each task whatever
/// the RON carried: a key nobody reads is discarded in SILENCE — no parse error, no warning, no
/// log difference. A graph config can then carry keys under a comment asserting they are inert
/// while the deployed binary reads them, and only a measurement of the output settles which is
/// true. A typo behaves identically to a correct key whose dependency is too old, and BOTH
/// directions of the mistake read the same from outside: "the knob did nothing".
///
/// Refusing at construction makes both loud at startup instead of never. Behaviour-neutral for a
/// correct RON, which is what keeps it out of any A/B.
///
/// `task` and `known` are parameters rather than baked in so an application can apply the same
/// check, in the same `(config, task, keys)` shape, to its own tasks.
pub fn deny_unknown_keys(
    config: Option<&ComponentConfig>,
    task: &'static str,
    known: &[&str],
) -> CuResult<()> {
    let Some(config) = config else {
        return Ok(());
    };
    // Allocation-free on the success path: `Filter`'s lower size_hint is 0, so `collect` starts
    // at capacity 0 and never pushes, and `sort_unstable` sits behind the early return.
    let mut unknown: Vec<&str> = config
        .0
        .keys()
        .map(String::as_str)
        .filter(|k| !known.contains(k))
        .collect();
    if unknown.is_empty() {
        return Ok(());
    }
    unknown.sort_unstable();
    Err(format!("unknown config key(s) {unknown:?} for {task}; it reads {known:?}").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::config::Value;

    #[test]
    fn test_on_tracking_loss_reads_the_snake_case_spelling_only() {
        let parse = |text: &str| {
            let mut cfg = ComponentConfig::new();
            cfg.set(ON_TRACKING_LOSS, text.to_string());
            on_tracking_loss(Some(&cfg))
        };
        assert_eq!(parse("reset_map").ok(), Some(OnTrackingLoss::ResetMap));
        assert_eq!(parse("keep_map").ok(), Some(OnTrackingLoss::KeepMap));
        assert!(
            parse("ResetMap").is_err(),
            "one canonical spelling, no aliases"
        );
        assert!(parse("reset").is_err());
        assert_eq!(on_tracking_loss(None).ok(), Some(OnTrackingLoss::KeepMap));
    }

    #[test]
    fn test_at_keyframe_bound_reads_the_snake_case_spelling_and_needs_the_bound() {
        let parse = |text: &str, cap: Option<usize>| {
            let mut cfg = ComponentConfig::new();
            cfg.set(AT_KEYFRAME_BOUND, text.to_string());
            at_keyframe_bound(Some(&cfg), cap)
        };
        assert_eq!(
            parse("evict_oldest", Some(120)).ok(),
            Some(AtKeyframeBound::EvictOldest)
        );
        assert_eq!(parse("drop_map", None).ok(), Some(AtKeyframeBound::DropMap));
        assert!(parse("EvictOldest", Some(120)).is_err());
        assert!(
            parse("evict_oldest", None).is_err(),
            "evicting without a bound never acts"
        );
        assert_eq!(
            at_keyframe_bound(None, None).ok(),
            Some(AtKeyframeBound::DropMap)
        );
    }

    /// `optional_positive_f64` is the guard on every float geometry knob, and had no test.
    #[test]
    fn test_non_positive_or_non_finite_float_key_is_refused() {
        let mut cfg = ComponentConfig::new();
        cfg.set("r", 8.0f64);
        assert_eq!(optional_positive_f64(Some(&cfg), "r").unwrap(), Some(8.0));
        // Absent is fine — every one of these keys is optional.
        assert_eq!(optional_positive_f64(Some(&cfg), "absent").unwrap(), None);
        assert_eq!(optional_positive_f64(None, "r").unwrap(), None);

        for bad in [0.0f64, -1.0, f64::NAN, f64::INFINITY] {
            let mut c = ComponentConfig::new();
            c.set("r", bad);
            assert!(
                optional_positive_f64(Some(&c), "r").is_err(),
                "{bad} was accepted as a positive finite value"
            );
        }
    }

    fn ron(rotation: [f64; 9]) -> InertialRon {
        InertialRon {
            t_bc_rotation: rotation,
            t_bc_translation: [0.01, -0.02, 0.03],
            gyro_noise: 1.6e-4,
            accel_noise: 2.0e-3,
            gyro_bias_noise: 1.9e-5,
            accel_bias_noise: 3.0e-3,
            rate_hz: 200.0,
            enable_inertial_ba: false,
            static_init: None,
            gyro_bias_limit: None,
            accel_bias_limit: None,
        }
    }

    /// The `inertial` block read the way a graph reads it, with `extra` appended to its fields.
    fn inertial_ron(extra: &str) -> CuResult<Option<InertialRon>> {
        let text = format!(
            r#"( tasks: [ ( id: "vio", type: "t::T", config: {{ "inertial": (
                 t_bc_rotation: [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
                 t_bc_translation: [0.0, 0.0, 0.0],
                 gyro_noise: 0.00012, accel_noise: 0.0019,
                 gyro_bias_noise: 0.000002, accel_bias_noise: 0.00004,
                 rate_hz: 200.0, enable_inertial_ba: true {extra} ) }} ) ],
                 cnx: [] )"#
        );
        let cfg = cu29::config::read_configuration_str(text, None)?;
        let graph = cfg.get_graph(None)?;
        let id = graph.get_node_id_by_name("vio").expect("the node");
        let node = graph.get_node(id).expect("the node");
        optional_inertial(node.get_instance_config())
    }

    /// The bias limits: absent is the default, a value replaces its own norm only, and a
    /// non-positive or non-finite one is refused.
    ///
    /// Mutation: dropping the `*slot = value` write turned this red.
    #[test]
    fn test_the_bias_limit_keys() {
        let cfg = |extra: &str| inertial_ron(extra).unwrap().expect("present").to_config();
        assert_eq!(cfg("").unwrap().bias_limit, crate::BiasLimit::default());
        let tuned = cfg(", gyro_bias_limit: Some(0.03)").unwrap().bias_limit;
        assert_eq!(tuned.gyro_rad_s, 0.03);
        assert_eq!(tuned.accel_m_s2, crate::BiasLimit::DEFAULT_ACCEL_M_S2);
        let tuned = cfg(", accel_bias_limit: Some(0.8)").unwrap().bias_limit;
        assert_eq!(tuned.accel_m_s2, 0.8);
        assert_eq!(tuned.gyro_rad_s, crate::BiasLimit::DEFAULT_GYRO_RAD_S);
        // The still window's gyro mean is capped at the gyro bias limit: a window over it could
        // only be refused by the limit after it was found.
        let capped = cfg(", gyro_bias_limit: Some(0.01), static_init: Some((enabled: true))")
            .unwrap()
            .static_init
            .expect("enabled");
        assert_eq!(capped.gyro_mean_max, 0.01);
        let kept = cfg(", static_init: Some((enabled: true))")
            .unwrap()
            .static_init
            .expect("enabled");
        assert_eq!(kept.gyro_mean_max, 0.02, "under the default limit");
        for bad in ["gyro_bias_limit: Some(0.0)", "accel_bias_limit: Some(-1.0)"] {
            assert!(
                matches!(
                    cfg(&format!(", {bad}")),
                    Err(VioError::InertialParamInvalid { .. })
                ),
                "{bad} accepted"
            );
        }
    }

    /// `static_init`: absent and disabled are off; enabled takes the noise-derived defaults; an
    /// override replaces exactly its field; a typo or a non-positive value is refused.
    ///
    /// Mutation: ignoring `enabled` (always `Some`) turned this red.
    #[test]
    fn test_the_static_init_block() {
        let cfg = |extra: &str| inertial_ron(extra).unwrap().expect("present").to_config();

        assert!(cfg("").unwrap().static_init.is_none(), "absent is off");
        assert!(
            cfg(", static_init: (window_s: 2.0)")
                .unwrap()
                .static_init
                .is_none(),
            "enabled defaults to false"
        );

        let on = cfg(", static_init: (enabled: true)").unwrap();
        let derived = StaticInitConfig::from_noise(&on.calib, on.gates.nominal_rate);
        assert_eq!(on.static_init, Some(derived));

        let tuned = cfg(", static_init: (enabled: true, gyro_rms_max: 0.02, window_s: 1.5)")
            .unwrap()
            .static_init
            .unwrap();
        assert_eq!(tuned.gyro_rms_max, 0.02);
        assert_eq!(tuned.window.get::<cu29::units::si::time::second>(), 1.5);
        assert_eq!(tuned.accel_rms_max, derived.accel_rms_max);

        assert!(
            inertial_ron(", static_init: (enabled: true, gyro_rms_mx: 0.02)").is_err(),
            "unknown field accepted"
        );
        for bad in [
            "gravity_tolerance: 0.0",
            "window_s: -1.0",
            "accel_rms_max: -0.1",
        ] {
            assert!(
                matches!(
                    cfg(&format!(", static_init: (enabled: false, {bad})")),
                    Err(VioError::InertialParamInvalid { .. })
                ),
                "{bad} accepted"
            );
        }
    }

    /// The row-major -> column-major hand-off, checked on an asymmetric rotation so that a
    /// missing transpose changes the answer. A 90 degree yaw about y is symmetric under
    /// transpose in the wrong way to catch it; this is 90 degrees about z.
    #[test]
    fn test_rotation_is_read_row_major() {
        // Rows of R_BC: (0,-1,0), (1,0,0), (0,0,1).
        let cfg = ron([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            .to_config()
            .expect("a proper rotation");
        // R * (1,0,0) must be the FIRST COLUMN, which the rows above make (0,1,0).
        let mapped = cfg.imu_t_bc.rotation * Vec3F64::new(1.0, 0.0, 0.0);
        assert!(
            (mapped - Vec3F64::new(0.0, 1.0, 0.0)).length() < 1e-12,
            "got {mapped:?}; a transposed read would give (0,-1,0)"
        );
        assert_eq!(cfg.imu_t_bc.translation.y, -0.02);
    }

    #[test]
    fn test_non_rotation_extrinsic_is_rejected() {
        // Scaled by 2: still orthogonal, no longer orthonormal.
        let err = ron([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0])
            .to_config()
            .unwrap_err();
        assert!(
            matches!(err, VioError::InertialExtrinsicNotRigid { .. }),
            "{err:?}"
        );
        // A reflection: det = -1, every row still unit length.
        let err = ron([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            .to_config()
            .unwrap_err();
        assert!(
            matches!(err, VioError::InertialExtrinsicNotRigid { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn test_zero_noise_density_is_rejected() {
        let mut r = ron([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        r.gyro_noise = 0.0;
        let err = r.to_config().unwrap_err();
        match err {
            VioError::InertialParamInvalid { field, .. } => assert_eq!(field, "gyro_noise"),
            other => panic!("{other:?}"),
        }
    }

    /// `deny_unknown_keys` must REFUSE, not ignore. Without this nothing proves the function
    /// does anything — the key list alone is inert.
    #[test]
    fn test_typo_is_refused_rather_than_discarded() {
        // A TRANSPOSITION, not the singular `orb_keypoint`. The error echoes `known`, which
        // contains `orb_keypoints`, and the singular is a PREFIX of it — so `contains` would pass
        // on the echoed list alone, whether or not the message ever named the offending key. That
        // is the one property these assertions exist to prove.
        let err = deny_unknown_keys(Some(&cfg(&["orb_keypionts"])), "task", CONFIG_KEYS)
            .expect_err("an unknown key must be refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("orb_keypionts"),
            "must name the offender: {msg}"
        );
        assert!(msg.contains("orb_keypoints"), "must name the fix: {msg}");
        assert!(msg.contains("task"), "must name the task: {msg}");
    }

    /// The other half: every key the graphs actually ship is accepted. A checker that refuses
    /// everything would pass the test above.
    #[test]
    fn test_keys_the_graphs_ship_are_accepted() {
        deny_unknown_keys(Some(&cfg(CONFIG_KEYS)), "task", CONFIG_KEYS).unwrap();
        deny_unknown_keys(None, "task", CONFIG_KEYS).unwrap();
    }

    /// The `wheel_odom` block read the way a graph reads it: through a RON config file.
    fn wheel_ron(block: &str) -> CuResult<Option<WheelOdomRon>> {
        let text = format!(
            r#"( tasks: [ ( id: "vio", type: "t::T", config: {{ "wheel_odom": {block} }} ) ],
                 cnx: [] )"#
        );
        let cfg = cu29::config::read_configuration_str(text, None)?;
        let graph = cfg.get_graph(None)?;
        let id = graph.get_node_id_by_name("vio").expect("the node");
        let node = graph.get_node(id).expect("the node");
        optional_wheel_odom(node.get_instance_config())
    }

    const WHEEL_BLOCK: &str = "(enabled: true, \
        cam_in_base_rotation: [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0], \
        cam_in_base_translation: [0.1, 0.0, 0.9], translation_noise: 0.02, rotation_noise: 0.05)";

    #[test]
    fn test_a_wheel_odom_block_is_read_row_major_and_inverted_once() {
        let ron = wheel_ron(WHEEL_BLOCK).unwrap().expect("present");
        let cfg = ron.to_config().unwrap().expect("enabled");
        // The camera's +z (optical axis) is the base's +x: t_cb maps base +x onto camera +z.
        let fwd = cfg.t_cb.rotation * Vec3F64::new(1.0, 0.0, 0.0);
        assert!(
            (fwd - Vec3F64::new(0.0, 0.0, 1.0)).length() < 1e-12,
            "{fwd:?}"
        );
        assert_eq!(
            cfg.z_sigma,
            kornia_slam::mapping::wheel_odom::WheelOdomMeasurement::DEFAULT_Z_SIGMA
        );
        assert_eq!(cfg.huber_chi2, None);
    }

    #[test]
    fn test_a_wheel_odom_block_is_off_unless_enabled() {
        let text = WHEEL_BLOCK.replace("enabled: true, ", "");
        let ron = wheel_ron(&text).unwrap().expect("present");
        assert!(
            ron.to_config().unwrap().is_none(),
            "enabled defaults to false"
        );
        assert!(optional_wheel_odom(None).unwrap().is_none());
    }

    #[test]
    fn test_a_malformed_wheel_odom_block_is_refused() {
        // An unknown field: a misspelt optional must not silently keep its default.
        let typo = WHEEL_BLOCK.replace("rotation_noise", "rotation_noise: 0.05, z_sigmaa");
        assert!(wheel_ron(&typo).is_err(), "unknown field accepted");
        // A required field missing.
        let partial = WHEEL_BLOCK.replace(", translation_noise: 0.02", "");
        assert!(wheel_ron(&partial).is_err(), "missing noise accepted");
        // A non-positive sigma, even while disabled.
        let zero = WHEEL_BLOCK
            .replace("enabled: true", "enabled: false")
            .replace(")", ", z_sigma: 0.0)");
        let ron = wheel_ron(&zero).unwrap().expect("parses");
        assert!(matches!(
            ron.to_config(),
            Err(VioError::WheelParamInvalid {
                field: "z_sigma",
                ..
            })
        ));
    }

    /// The noise-model keys sit inside the `wheel_odom` block, read as given, defaulted when
    /// absent, and refused when negative even while the block is disabled. They are NOT
    /// task-level keys: `deny_unknown_fields` on the block is what catches a typo in them.
    #[test]
    fn test_the_wheel_noise_model_keys_are_read_defaulted_and_validated() {
        let defaults = wheel_ron(WHEEL_BLOCK)
            .unwrap()
            .expect("present")
            .to_config()
            .unwrap()
            .expect("enabled");
        assert_eq!(defaults.heading_noise_per_m, 0.0);
        assert_eq!(defaults.heading_scale_sigma_per_m, 0.0);
        assert_eq!(defaults.z_slope, 0.0);
        assert_eq!(
            defaults.clock_sigma.get::<cu29::units::si::time::second>(),
            crate::wheel::DEFAULT_CLOCK_SIGMA_S
        );
        // The floors stay where the factor shipped them unless a robot measures lower ones.
        assert_eq!(defaults.min_translation_sigma_m, 1e-3);
        assert_eq!(defaults.min_heading_sigma_rad, 1e-3);
        let set = WHEEL_BLOCK.replace(
            ")",
            ", heading_noise_per_m: 0.01, heading_scale_sigma_per_m: 0.002, z_slope: 0.003, \
             clock_sigma_s: 0.002, min_translation_sigma_m: 0.0001, min_heading_sigma_rad: 0.0002)",
        );
        let cfg = wheel_ron(&set)
            .unwrap()
            .expect("present")
            .to_config()
            .unwrap()
            .expect("enabled");
        assert_eq!(cfg.heading_noise_per_m, 0.01);
        assert_eq!(cfg.heading_scale_sigma_per_m, 0.002);
        assert_eq!(cfg.z_slope, 0.003);
        assert_eq!(
            cfg.clock_sigma.get::<cu29::units::si::time::second>(),
            0.002
        );
        assert_eq!(cfg.min_translation_sigma_m, 0.0001);
        assert_eq!(cfg.min_heading_sigma_rad, 0.0002);
        for key in [
            "heading_noise_per_m",
            "heading_scale_sigma_per_m",
            "z_slope",
            "clock_sigma_s",
            "min_translation_sigma_m",
            "min_heading_sigma_rad",
        ] {
            let negative = WHEEL_BLOCK
                .replace("enabled: true", "enabled: false")
                .replace(")", &format!(", {key}: -0.1)"));
            let err = wheel_ron(&negative).unwrap().expect("parses").to_config();
            assert!(
                matches!(err, Err(VioError::WheelParamInvalid { field, .. }) if field == key),
                "{key}: {err:?}"
            );
        }
        // A zero floor would hand a parked interval infinite information.
        for key in ["min_translation_sigma_m", "min_heading_sigma_rad"] {
            let zero = WHEEL_BLOCK.replace(")", &format!(", {key}: 0.0)"));
            let err = wheel_ron(&zero).unwrap().expect("parses").to_config();
            assert!(
                matches!(err, Err(VioError::WheelParamInvalid { field, .. }) if field == key),
                "{key}: {err:?}"
            );
        }
        let typo = WHEEL_BLOCK.replace(")", ", z_slop: 0.01)");
        assert!(wheel_ron(&typo).is_err(), "a misspelt noise key accepted");
        for key in ["heading_noise_per_m", "z_slope", "clock_sigma_s"] {
            assert!(!CONFIG_KEYS.contains(&key), "{key} is a wheel_odom field");
        }
    }

    #[test]
    fn test_the_parked_keyframe_interval_is_off_unless_set_and_positive() {
        let to = |block: &str| wheel_ron(block).unwrap().expect("parses").to_config();
        let off = to(WHEEL_BLOCK).unwrap().expect("enabled");
        assert_eq!(off.parked_keyframe_interval, None);
        let on = to(&WHEEL_BLOCK.replace(")", ", parked_keyframe_interval_s: 10.0)"))
            .unwrap()
            .expect("enabled");
        assert_eq!(
            on.parked_keyframe_interval
                .map(|t| t.get::<cu29::units::si::time::second>()),
            Some(10.0)
        );
        // Zero would suppress every parked keyframe for ever; negative is a typo.
        for v in ["0.0", "-1.0"] {
            let err = to(&WHEEL_BLOCK.replace(")", &format!(", parked_keyframe_interval_s: {v})")));
            assert!(
                matches!(err, Err(VioError::WheelParamInvalid { field, .. })
                    if field == "parked_keyframe_interval_s"),
                "{v}: {err:?}"
            );
        }
        assert!(!CONFIG_KEYS.contains(&"parked_keyframe_interval_s"));
    }

    #[test]
    fn test_wheel_odom_is_a_known_key() {
        assert!(CONFIG_KEYS.contains(&"wheel_odom"));
        deny_unknown_keys(Some(&cfg(&["wheel_odom"])), "task", CONFIG_KEYS).unwrap();
        assert!(deny_unknown_keys(Some(&cfg(&["wheel_odometry"])), "task", CONFIG_KEYS).is_err());
    }

    /// The `per_frame_fusion` block read the way a graph reads it.
    fn fusion_ron(block: &str) -> CuResult<Option<FusionConfig>> {
        let text = format!(
            r#"( tasks: [ ( id: "vio", type: "t::T", config: {{ "per_frame_fusion": {block} }} ) ],
                 cnx: [] )"#
        );
        let cfg = cu29::config::read_configuration_str(text, None)?;
        let graph = cfg.get_graph(None)?;
        let id = graph.get_node_id_by_name("vio").expect("the node");
        let node = graph.get_node(id).expect("the node");
        optional_fusion(node.get_instance_config())
    }

    /// Off unless enabled, the gate defaulted or read, and a bad block refused at startup, even
    /// while disabled.
    #[test]
    fn test_the_per_frame_fusion_block() {
        assert!(CONFIG_KEYS.contains(&"per_frame_fusion"));
        assert_eq!(optional_fusion(None).unwrap(), None);
        assert_eq!(
            fusion_ron("(chi2_gate: 30.0)").unwrap(),
            None,
            "enabled defaults to false"
        );
        assert_eq!(
            fusion_ron("(enabled: true)").unwrap(),
            Some(FusionConfig::default())
        );
        assert_eq!(
            fusion_ron("(enabled: true, chi2_gate: 30.0)")
                .unwrap()
                .map(|c| c.chi2_gate),
            Some(30.0)
        );
        assert!(fusion_ron("(enabled: false, chi2_gate: 0.0)").is_err());
        assert!(
            fusion_ron("(enabled: true, chi2: 30.0)").is_err(),
            "unknown field"
        );
        let c = fusion_ron("(enabled: true, extrinsic_sigma_m: 0.05, extrinsic_sigma_rad: 0.02)")
            .unwrap()
            .unwrap();
        assert_eq!((c.extrinsic_sigma_m, c.extrinsic_sigma_rad), (0.05, 0.02));
        assert!(fusion_ron("(enabled: true, extrinsic_sigma_m: -0.01)").is_err());
        assert!(fusion_ron("(enabled: true, extrinsic_sigma_rad: 0.0)").is_err());
    }

    /// A `vio` node's config carrying `entries` (RON map entries), read the way a graph reads it.
    fn node_config(entries: &str) -> ComponentConfig {
        let text = format!(
            r#"( tasks: [ ( id: "vio", type: "t::T", config: {{ {entries} }} ) ], cnx: [] )"#
        );
        let cfg = cu29::config::read_configuration_str(text, None).expect("a graph");
        let graph = cfg.get_graph(None).expect("the graph");
        let id = graph.get_node_id_by_name("vio").expect("the node");
        graph
            .get_node(id)
            .expect("the node")
            .get_instance_config()
            .cloned()
            .expect("a config")
    }

    fn insertion(block: &str) -> CuResult<KeyframeInsertion> {
        keyframe_insertion(Some(&node_config(&format!(
            r#""keyframe_policy": {block}"#
        ))))
    }

    /// Absent is kornia-slam's default policy and no time bound; each field replaces its own
    /// value; `motion_only` lifts the frame bound and needs parallax; nonsense is refused.
    ///
    /// Mutation: ignoring `motion_only` (keeping the frame bound) turned this red.
    #[test]
    fn test_the_keyframe_policy_block() {
        assert!(CONFIG_KEYS.contains(&"keyframe_policy"));
        let default = KeyframePolicy::default();
        let absent = keyframe_insertion(None).unwrap();
        assert_eq!(
            (
                absent.policy.min_frames_between,
                absent.policy.max_frames_between,
                absent.policy.ref_ratio,
                absent.policy.min_parallax_px
            ),
            (
                default.min_frames_between,
                default.max_frames_between,
                default.ref_ratio,
                None
            )
        );
        assert!(absent.max_interval.is_none());

        let tuned = insertion(
            "(min_frames_between: 2, max_frames_between: 12, ref_ratio: 0.5, \
             min_parallax_px: 10.0, max_interval_s: 0.8)",
        )
        .unwrap();
        assert_eq!(tuned.policy.min_frames_between, 2);
        assert_eq!(tuned.policy.max_frames_between, 12);
        assert_eq!(tuned.policy.ref_ratio, 0.5);
        assert_eq!(tuned.policy.min_parallax_px, Some(10.0));
        assert_eq!(tuned.max_interval.map(|t| t.get::<second>()), Some(0.8));

        let motion = insertion("(motion_only: true, min_parallax_px: 10.0)").unwrap();
        assert_eq!(motion.policy.max_frames_between, usize::MAX);

        for bad in [
            "(motion_only: true)",
            "(motion_only: true, min_parallax_px: 10.0, max_frames_between: 8)",
            "(max_frames_between: 0)",
            "(min_frames_between: 9)",
            "(ref_ratio: 0.0)",
            "(ref_ratio: 1.5)",
            "(min_parallax_px: -1.0)",
            "(max_interval_s: 0.0)",
            "(min_parallax: 10.0)",
        ] {
            assert!(insertion(bad).is_err(), "{bad} accepted");
        }
    }

    fn culling(
        entries: &str,
        max_keyframes: Option<usize>,
    ) -> CuResult<Option<KeyframeCullingConfig>> {
        keyframe_culling(
            Some(&node_config(&format!(r#""keyframe_culling": {entries}"#))),
            max_keyframes,
        )
    }

    /// Off unless enabled; enabled takes kornia-slam's defaults and each field replaces its own;
    /// it needs a `max_keyframes` above its `min_keyframes`; a bad value is refused even while
    /// disabled.
    ///
    /// Mutation: ignoring `enabled` (always `Some`) turned this red.
    #[test]
    fn test_the_keyframe_culling_block() {
        assert!(CONFIG_KEYS.contains(&"keyframe_culling"));
        assert!(keyframe_culling(None, Some(120)).unwrap().is_none());
        assert!(culling("(min_keyframes: 30)", Some(120)).unwrap().is_none());
        let on = culling("(enabled: true)", Some(120))
            .unwrap()
            .expect("enabled");
        let default = KeyframeCullingConfig::default();
        assert_eq!(on.min_keyframes, default.min_keyframes);
        assert_eq!(on.redundant_ratio, default.redundant_ratio);
        assert_eq!(on.max_removals, default.max_removals);

        let tuned = culling(
            "(enabled: true, redundant_ratio: 0.8, min_other_observers: 2, \
             scale_level_tolerance: 0, protect_newest: 5, min_keyframes: 40, \
             max_merged_imu_span_s: 2.0, max_removals: 7)",
            Some(120),
        )
        .unwrap()
        .expect("enabled");
        assert_eq!(tuned.redundant_ratio, 0.8);
        assert_eq!(tuned.min_other_observers, 2);
        assert_eq!(tuned.scale_level_tolerance, 0);
        assert_eq!(tuned.protect_newest, 5);
        assert_eq!(tuned.min_keyframes, 40);
        assert_eq!(tuned.max_merged_imu_span_sec, 2.0);
        assert_eq!(tuned.max_removals, 7);

        assert!(
            culling("(enabled: true)", None).is_err(),
            "no bound to act at"
        );
        assert!(
            culling("(enabled: true, min_keyframes: 120)", Some(120)).is_err(),
            "can never make room"
        );
        assert!(
            culling("(enabled: false)", None).unwrap().is_none(),
            "disabled needs no bound"
        );
        for bad in [
            "(enabled: false, redundant_ratio: 0.0)",
            "(enabled: false, protect_newest: 0)",
            "(enabled: false, min_other_observers: 0)",
            "(enabled: false, max_merged_imu_span_s: -1.0)",
            "(enabled: true, redundant: 0.9)",
        ] {
            assert!(culling(bad, Some(120)).is_err(), "{bad} accepted");
        }
    }

    fn stationary(entries: &str) -> CuResult<Option<StationaryEdgeConfig>> {
        optional_stationary_edges(Some(&node_config(&format!(
            r#""stationary_edges": {entries}"#
        ))))
    }

    /// Off unless enabled; enabled takes kornia-slam's defaults and each field replaces its own;
    /// a bad value is refused even while disabled, and so is an unknown field.
    ///
    /// Mutation: ignoring `enabled` (always `Some`) turned this red.
    #[test]
    fn test_the_stationary_edges_block() {
        assert!(CONFIG_KEYS.contains(&"stationary_edges"));
        deny_unknown_keys(Some(&cfg(&["stationary_edges"])), "task", CONFIG_KEYS).unwrap();
        assert!(optional_stationary_edges(None).unwrap().is_none());
        assert!(stationary("(min_span_s: 2.0)").unwrap().is_none());
        assert_eq!(
            stationary("(enabled: true)").unwrap(),
            Some(StationaryEdgeConfig::new())
        );
        let tuned = stationary(
            "(enabled: true, translation_sigma_m: 0.003, rotation_sigma_rad: 0.004, \
             velocity_sigma_m_s: 0.02, huber_chi2: 12.0, min_span_s: 2.0)",
        )
        .unwrap()
        .expect("enabled");
        assert_eq!(tuned.translation_sigma_m, 0.003);
        assert_eq!(tuned.rotation_sigma_rad, 0.004);
        assert_eq!(tuned.velocity_sigma_m_s, 0.02);
        assert_eq!(tuned.huber_chi2, Some(12.0));
        assert_eq!(tuned.min_span.get::<second>(), 2.0);
        for bad in [
            "(enabled: false, translation_sigma_m: 0.0)",
            "(enabled: false, min_span_s: 0.2)",
            "(enabled: true, huber_chi2: -1.0)",
            "(enabled: true, velocity_sigma_m_s: -0.01)",
            "(enabled: true, sigma: 1.0)",
        ] {
            assert!(stationary(bad).is_err(), "{bad} accepted");
        }
    }

    fn motion_veto(entries: &str) -> CuResult<Option<crate::veto::MotionVetoConfig>> {
        optional_motion_veto(Some(&node_config(entries)))
    }

    /// Absent is ON with the defaults; each key overrides; `enabled: false` is off; an
    /// out-of-range value and an unknown field are refused at startup.
    #[test]
    fn test_the_motion_veto_block() {
        let d = crate::veto::MotionVetoConfig::default();
        assert_eq!(optional_motion_veto(None).unwrap(), Some(d));
        assert_eq!(motion_veto("").unwrap(), Some(d));
        let c = motion_veto(
            r#""motion_veto": (rotation_deg: 20.0, translation_m: 0.5, rotation_deg_per_s: 1.0,
                translation_m_per_s: 0.05, motion_fraction: 0.1, recovery_frames: 8,
                recovery_timeout_s: 4.0, rotation_window_s: 8.0, translation_window_s: 1.0)"#,
        )
        .unwrap()
        .unwrap();
        assert!((c.rotation_rad - 20f64.to_radians()).abs() < 1e-12);
        assert_eq!(c.translation_m, 0.5);
        assert!((c.rotation_rad_per_s - 1f64.to_radians()).abs() < 1e-12);
        assert_eq!(c.translation_m_per_s, 0.05);
        assert_eq!(c.motion_fraction, 0.1);
        assert_eq!(c.recovery_frames, 8);
        assert_eq!(c.recovery_timeout_s, 4.0);
        assert_eq!(c.rotation_window_s, 8.0);
        assert_eq!(c.translation_window_s, 1.0);
        assert_eq!(
            motion_veto(r#""motion_veto": (enabled: false)"#).unwrap(),
            None
        );
        for bad in [
            r#""motion_veto": (rotation_deg: 0.0)"#,
            r#""motion_veto": (recovery_frames: 0)"#,
            r#""motion_veto": (translation_window_s: 0.0)"#,
            r#""motion_veto": (enabled: false, translation_m: -1.0)"#,
            r#""motion_veto": (rotation_degs: 10.0)"#,
        ] {
            assert!(motion_veto(bad).is_err(), "{bad}");
        }
        assert!(CONFIG_KEYS.contains(&MOTION_VETO));
    }

    fn cfg(keys: &[&str]) -> ComponentConfig {
        ComponentConfig(
            keys.iter()
                .map(|k| ((*k).to_string(), Value::from("1".to_string())))
                .collect(),
        )
    }
}
