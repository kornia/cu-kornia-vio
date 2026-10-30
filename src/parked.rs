//! Parked keyframes: stationary edges, and no time-based keyframes while parked.
//!
//! A parked robot used to insert a keyframe every `wheel_odom.parked_keyframe_interval_s` (5 s on
//! the robot), each joined to the last by an IMU edge. Keyframe culling merges IMU edges only up
//! to `keyframe_culling.max_merged_imu_span_s` (11 s): after one merge most parked edges spanned
//! ~10 s, two could not merge, `kept_for_edges` climbed and the map hit `max_keyframes` and was
//! dropped every ~20 minutes (robot, 2026-10-12, build 23a413bdf2e9: culling yields 61, 31, 15,
//! 7, 3, 2, 1, `kept_for_edges` 54 -> 113).
//!
//! With [`StationaryEdgeConfig`] on (`stationary_edges` in the RON), once the inertial path is
//! settled:
//!
//! * a keyframe a parked robot's policy asks for on its TIME bound (the frame-count bound,
//!   `keyframe_policy.max_interval_s`) is withheld while the robot stays parked, up to
//!   [`PARKED_REFRESH_NS`] (30 s) after the last keyframe; tracking-quality triggers (the tracked
//!   ratio dropping, parallax, an empty reference) still insert, so a changed scene is still
//!   mapped. The refresh keyframe is linked by a stationary factor, so the stop's gyro mean
//!   reaches VI-BA every 30 s and at most 30 s of it is lost when the stop ends in motion;
//! * a keyframe whose interval since the last one was verified still is joined to it by a
//!   kornia-slam `StationaryFactor` (zero relative pose, zero velocity, bias continuity and the
//!   mean gyro rate), not by an IMU edge. It has no span limit, and culling composes it across a
//!   removed keyframe indefinitely; a keyframe where it meets an IMU edge is culled too (the
//!   factor dropped, the IMU edge moved onto the still neighbour);
//! * a keyframe after a still stretch that ended in motion gets an IMU edge over the TAIL only
//!   (from [`STATIONARY_TAIL_MIN_S`] before the motion's start), its bias link widened for the
//!   elided time, which the edge records so a merge or a repropagation keeps it: given zero
//!   velocity and an unchanged pose up to the tail's start, that is exactly the edge from the
//!   keyframe;
//! * the IMU and wheel rings are pruned every frame to that anchor, so no keyframe for an hour
//!   holds a few hundred samples, not an hour of them.
//!
//! "Still" is the [`StationaryRun`]: from each keyframe on, the IMU samples are checked in 0.5 s
//! chunks against the stationary initialization's per-sample checks (accelerometer and gyroscope
//! spread, mean rate; the spread ceilings raised to [`RUN_ACCEL_RMS_MAX`] and
//! [`RUN_GYRO_RMS_MAX`] for the robot's own vibration), and two failing chunks in a row close the
//! run (one is a disturbance: a door, footsteps); the wheel path stays under the parked floors per 5 s window, as the interval gate
//! judged it, and the net displacement since the keyframe under the same floors. A hole in the
//! IMU samples or a discontinuity in the wheel readings closes the run for good; a gap in the
//! wheel readings (a stale head/body link) is bridged once the IMU has verified the stretch. The
//! next keyframe restarts the run. A keyframe also needs the camera within 1 cm / 0.5 deg of the
//! last one AND of the first keyframe of the stationary chain to be linked as still, so a slow
//! creep cannot end up under one composed zero-motion factor.
//!
//! Before the inertial path is initialized, and while a refinement of it is pending, nothing
//! changes: the still window and the refinements are built from IMU edges, and the parked
//! interval gate keeps those edges inside the merge cap.

use cu29::units::si::f64::Time;
use cu29::units::si::time::second;
use kornia_algebra::Vec3F64;
use kornia_sensors::imu::ImuMeasurement;
use kornia_slam::inertial::ImuWindowStats;
use kornia_slam::mapping::stationary::{StationaryFactor, StationaryGyro};

use crate::error::VioError;
use crate::imu::{ImuBuffer, ImuGates};
use crate::stationary::{NotStill, StaticInitConfig, check_imu_stats};
use crate::wheel::{PARKED_PATH_M, PARKED_PATH_RAD, WheelState, WheelWindowError};

/// IMU kept before a stationary anchor and the shortest tail edge, seconds: enough samples for
/// the ring's coverage gates. Also the floor on [`StationaryEdgeConfig::min_span`].
pub const STATIONARY_TAIL_MIN_S: f64 = 0.5;
pub(crate) const STATIONARY_TAIL_MIN_NS: u64 = 500_000_000;

/// Samples and readings kept before the anchor on top of the tail, nanoseconds.
pub(crate) const PRUNE_MARGIN_NS: u64 = 500_000_000;

/// The shortest IMU chunk the run checks at once, nanoseconds: 100 samples at 200 Hz. Two
/// failing chunks in a row close the run. A disturbance shorter than a chunk closes it only when
/// it straddles a chunk boundary strongly enough to fail both sides; a strong knock of a few
/// samples can, a weak one lasting under a chunk inside it cannot.
///
/// It was 250 ms; the owner asked for 500 ms. Parked on the robot (2026-10-12, build
/// 7b75117dd2da) the old check closed runs on the IMU 23 and 12 times per head run with the wheels
/// at 0, each closure letting a burst of interval keyframes in. The cause is bursts of 38-39 Hz
/// vibration, 2-4 s long, mostly on the gyro x axis (head pitch): peak 0.136 rad/s, net rotation
/// under 0.2 deg. Its spread does not shrink with the chunk (a steady vibration's never does), so
/// the chunk alone is not the fix (replayed over 10 recorded minutes at the static
/// initialization's ceilings: 8 closures at 250 ms, 5 at 500 ms, 2 at 1 s), the raised ceilings
/// [`RUN_GYRO_RMS_MAX`] and [`RUN_ACCEL_RMS_MAX`] are. With them the same replay closes nothing at
/// 250 ms or at 500 ms; with the vibration scaled up 1.3x, 500 ms still closes nothing and 250 ms
/// neither. What 500 ms buys is tolerance of short events (0.1-0.4 s ones were recorded too, too
/// weak to close a run at either length): a disturbance a few hundred milliseconds long that lies
/// inside a 500 ms chunk is a single failing chunk, where at 250 ms it always spans two. The cost
/// is later detection by the IMU: a 1.5 deg/s turn after 1.5 s instead of 0.9 s, a 0.5 m/s^2 x
/// 0.2 s shove in 1 of 30 phases instead of 30 (see [`RUN_ACCEL_RMS_MAX`]), a 2 deg head tilt
/// over 1 s in 19 of 30 phases instead of 30. Creep and gentle pushes never showed on the IMU at
/// any chunk length; the wheel floors and the camera's 1 cm / 0.5 deg floor catch them.
pub(crate) const RUN_CHUNK_NS: u64 = 500_000_000;

/// The run's gyroscope spread ceiling, rad/s, in place of the static initialization's when that
/// is lower (0.0118 on the robot's BNO086): 1.3x the worst 500 ms spread measured in the
/// vibration bursts above at any chunk phase (0.053; 0.048 on 500 ms chunks laid end to end). The
/// initialization keeps its own: it judges one window, once, and a vibration there only delays
/// it. A rotation is still caught by the mean-rate gate (a steady turn has no spread) and by the
/// wheels.
pub(crate) const RUN_GYRO_RMS_MAX: f64 = 0.07;

/// The run's accelerometer spread ceiling, m/s^2, in place of the static initialization's when
/// that is lower (0.186 on the robot's BNO086): 1.34x the worst 500 ms spread measured in the
/// vibration bursts at any chunk phase (0.179; 0.162 on chunks laid end to end), where 0.186 left
/// 1.04x. Replayed with the recorded bursts scaled up, 0.186 closed a run at 1.15x on this gate
/// alone, 0.24 nothing up to 1.3x. It costs the 0.5 m/s^2 x 0.2 s shove (12 of 30 phases to 1),
/// a 1 cm displacement the wheel floors (5 mm) and the camera floor see; nothing else of the
/// replayed motion set changed.
pub(crate) const RUN_ACCEL_RMS_MAX: f64 = 0.24;

/// Fewest samples in a checked chunk.
const RUN_CHUNK_MIN_SAMPLES: usize = 10;

/// The window the wheel path floors apply to, nanoseconds: the 5 s interval
/// `WheelState::parked_between` judged them over, so an hour of encoder jitter does not add up to
/// motion. Creep slower than the floors per window is caught by the net displacement since the
/// keyframe instead.
pub(crate) const RUN_WHEEL_WINDOW_NS: u64 = 5_000_000_000;

/// How far past the verified end of a still run that is still open a time-bound keyframe stays
/// withheld, nanoseconds, beyond the wheel `max_gap`: the wheel readings cross the head/body link,
/// which goes stale rather than applying backpressure, and a stall is not motion. Past it nothing
/// vouches for the robot and the keyframe is inserted.
pub(crate) const SENSOR_LAG_HOLD_NS: u64 = 2_000_000_000;

/// Longest a parked robot goes without a keyframe, seconds: past it a time-bound keyframe is
/// inserted (and linked by a stationary factor), so the gyro mean of a long stop reaches the
/// estimator every interval instead of being lost when the stop ends in motion, and VI-BA
/// refines the bias meanwhile. Culling composes these factors, so they cost no map room.
pub(crate) const PARKED_REFRESH_NS: u64 = 30_000_000_000;

/// Most stationary factors held. Culling composes them and a map drop clears them, so this is
/// never reached with `max_keyframes` set; without it it bounds a map that is never culled.
pub(crate) const MAX_STATIONARY_FACTORS: usize = 256;

/// The `stationary_edges` configuration. Build with [`StationaryEdgeConfig::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StationaryEdgeConfig {
    /// Sigma of the zero relative camera translation, metres.
    pub translation_sigma_m: f64,
    /// Sigma of the zero relative camera rotation, radians.
    pub rotation_sigma_rad: f64,
    /// Sigma of the zero velocity at each end, m/s.
    pub velocity_sigma_m_s: f64,
    /// Huber threshold on the factor's 12-row pose and velocity chi-square; `None` keeps it
    /// quadratic.
    pub huber_chi2: Option<f64>,
    /// The still time since the last keyframe under which a keyframe still gets the IMU edge of
    /// old: a robot that only paused does not change how it is linked.
    pub min_span: Time,
}

impl Default for StationaryEdgeConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl StationaryEdgeConfig {
    /// kornia-slam's `StationaryFactor` defaults (2 mm, 0.002 rad, 1 cm/s, Huber at the 95 %
    /// 12-DOF chi-square) and a 1 s minimum span.
    pub fn new() -> Self {
        Self {
            translation_sigma_m: StationaryFactor::DEFAULT_TRANSLATION_SIGMA_M,
            rotation_sigma_rad: StationaryFactor::DEFAULT_ROTATION_SIGMA_RAD,
            velocity_sigma_m_s: StationaryFactor::DEFAULT_VELOCITY_SIGMA_M_S,
            huber_chi2: Some(StationaryFactor::DEFAULT_HUBER_CHI2),
            min_span: Time::new::<second>(1.0),
        }
    }

    /// Every sigma finite and positive, a positive Huber threshold if any, and `min_span` at
    /// least [`STATIONARY_TAIL_MIN_S`].
    pub fn validate(&self) -> Result<(), VioError> {
        for (field, value) in [
            ("translation_sigma_m", self.translation_sigma_m),
            ("rotation_sigma_rad", self.rotation_sigma_rad),
            ("velocity_sigma_m_s", self.velocity_sigma_m_s),
            ("huber_chi2", self.huber_chi2.unwrap_or(1.0)),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(VioError::StationaryParamInvalid { field, value });
            }
        }
        let min_span = self.min_span.get::<second>();
        if !(min_span.is_finite() && min_span >= STATIONARY_TAIL_MIN_S) {
            return Err(VioError::StationaryParamInvalid {
                field: "min_span_s",
                value: min_span,
            });
        }
        Ok(())
    }

    pub(crate) fn min_span_ns(&self) -> u64 {
        (self.min_span.get::<second>() * 1e9) as u64
    }

    /// The factor between keyframes `from_idx` and `to_idx`.
    pub(crate) fn factor(
        &self,
        from_idx: usize,
        to_idx: usize,
        span_s: f64,
        gyro: Option<StationaryGyro>,
        calib: kornia_sensors::imu::ImuCalib,
    ) -> StationaryFactor {
        let mut factor = StationaryFactor::new(from_idx, to_idx, span_s, calib);
        factor.gyro = gyro;
        factor.translation_sigma_m = self.translation_sigma_m;
        factor.rotation_sigma_rad = self.rotation_sigma_rad;
        factor.velocity_sigma_m_s = self.velocity_sigma_m_s;
        factor.huber_chi2 = self.huber_chi2;
        factor
    }
}

/// Parked-keyframe counters. Diagnostics only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StationaryStats {
    /// Keyframes linked to the last one by a stationary factor, lifetime.
    pub edges_added: u64,
    /// Stationary factors composed across a culled keyframe, lifetime.
    pub edges_merged: u64,
    /// Stationary factors dropped with an evicted keyframe, lifetime.
    pub edges_evicted: u64,
    /// Keyframes culled at a boundary between a stationary factor and an IMU edge (the factor
    /// dropped, the IMU edge moved onto the still neighbour), lifetime.
    pub boundaries_removed: u64,
    /// Keyframes after a still stretch linked by an IMU edge over the tail only, lifetime.
    pub tail_edges: u64,
    /// Time-bound keyframes withheld while parked, once per keyframe interval, lifetime.
    pub keyframes_withheld: u64,
    /// Still runs closed by the IMU's stillness checks (two failing chunks in a row), lifetime.
    pub runs_closed_imu: u64,
    /// Still runs closed by a hole in the IMU samples, lifetime.
    pub runs_closed_imu_hole: u64,
    /// Still runs closed by the wheels (motion, or a discontinuity in the readings), lifetime.
    pub runs_closed_wheels: u64,
    /// Stationary factors held now.
    pub factors: usize,
    /// The robot is parked now: a still run from the last keyframe covers the last frame.
    pub parked: bool,
}

/// What closed a [`StationaryRun`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunClose {
    /// A chunk of IMU samples failed the stillness checks.
    Imu,
    /// The IMU samples have a hole (dropped, a gap, too few for the rate).
    ImuHole,
    /// The wheels' path since the keyframe passed the parked floors.
    Wheels,
    /// The wheel readings have a gap, a discontinuity, or do not cover the start.
    WheelGap,
}

/// The chunk length and spread ceilings [`StationaryRun::advance_imu`] checks with.
#[derive(Debug, Clone, Copy)]
struct RunImuCheck {
    chunk_ns: u64,
    accel_rms_max: f64,
    gyro_rms_max: f64,
}

/// The still stretch since one keyframe, verified as the samples and readings arrive.
#[derive(Debug, Clone)]
pub(crate) struct StationaryRun {
    /// `Frame::idx` of the keyframe the run starts at.
    pub kf_idx: usize,
    /// Its stamp.
    pub start_ns: u64,
    open: bool,
    /// The IMU is verified still up to here.
    imu_end_ns: u64,
    /// The wheels are verified still up to here.
    wheel_end_ns: u64,
    /// `sum(mean gyro * chunk span)` and the span, over the still chunks.
    gyro_dt_sum: Vec3F64,
    covered_s: f64,
    /// End of a chunk that failed the stillness checks, while the next one has not been judged:
    /// one failing chunk is a disturbance (a door, footsteps), two in a row close the run.
    suspect_end_ns: Option<u64>,
    /// The check that chunk failed, and its value.
    suspect_fail: Option<NotStill>,
    /// The checks the two failing chunks that closed the run failed, oldest first.
    pub imu_failures: Option<[NotStill; 2]>,
    /// Start of the current wheel window ([`RUN_WHEEL_WINDOW_NS`]) and the wheel path in it.
    window_start_ns: u64,
    path_m: f64,
    path_rad: f64,
    /// Net base motion since the keyframe, `(x, y, theta)` in the base at the keyframe.
    net: [f64; 3],
    /// Why it closed, once it did.
    pub closed_by: Option<RunClose>,
    /// The newest anchor the rings were pruned to; never decreases. Once past the keyframe the
    /// next edge can no longer start at the keyframe.
    pub committed_ns: u64,
}

impl StationaryRun {
    /// A run starting at keyframe `kf_idx`, stamped `start_ns`.
    pub fn new(kf_idx: usize, start_ns: u64) -> Self {
        Self {
            kf_idx,
            start_ns,
            open: true,
            imu_end_ns: start_ns,
            wheel_end_ns: start_ns,
            gyro_dt_sum: Vec3F64::ZERO,
            covered_s: 0.0,
            suspect_end_ns: None,
            suspect_fail: None,
            imu_failures: None,
            window_start_ns: start_ns,
            path_m: 0.0,
            path_rad: 0.0,
            net: [0.0; 3],
            closed_by: None,
            committed_ns: start_ns,
        }
    }

    /// The newest stamp both sensors vouch for. Frozen once the run closes.
    pub fn end_ns(&self) -> u64 {
        self.imu_end_ns.min(self.wheel_end_ns)
    }

    /// Whether the run is still open.
    #[cfg(test)]
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Whether the run vouches for `stamp_ns`: open, the stamp no further past the wheels'
    /// verified end than a wheel reading gap may be (the hold `WheelState::parked_between`
    /// allows), and no further past the IMU's than that plus one [`RUN_CHUNK_NS`]: the IMU end
    /// moves a chunk at a time, so on clean data it trails the frame by up to a chunk (475 ms
    /// measured with 500 ms chunks, against a 500 ms `max_gap`), and IMU arriving a little after
    /// the frame must not make a parked robot flicker out of `parked`. The chunk is not granted
    /// while a failing chunk awaits the next one: the samples past the IMU end were checked and
    /// showed a disturbance, so nothing past `max_gap` of it is vouched for.
    pub fn covers(&self, stamp_ns: u64, max_gap_ns: u64) -> bool {
        let chunk_lag_ns = if self.suspect_end_ns.is_some() {
            0
        } else {
            RUN_CHUNK_NS
        };
        self.open
            && stamp_ns <= self.wheel_end_ns.saturating_add(max_gap_ns)
            && stamp_ns
                <= self
                    .imu_end_ns
                    .saturating_add(max_gap_ns)
                    .saturating_add(chunk_lag_ns)
    }

    /// Whether a time-bound keyframe at `stamp_ns` stays withheld: the run is open (nothing has
    /// shown motion) and the stamp is within the wheel gap plus [`SENSOR_LAG_HOLD_NS`] of the
    /// verified end. Looser than [`Self::covers`], which decides what the keyframe is linked by.
    pub fn holds(&self, stamp_ns: u64, max_gap_ns: u64) -> bool {
        self.open
            && stamp_ns
                <= self
                    .end_ns()
                    .saturating_add(max_gap_ns)
                    .saturating_add(SENSOR_LAG_HOLD_NS)
    }

    /// The mean raw gyro rate over the still chunks, if any were checked.
    pub fn gyro(&self) -> Option<StationaryGyro> {
        (self.covered_s > 0.0)
            .then(|| StationaryGyro::new(self.gyro_dt_sum / self.covered_s, self.covered_s))
    }

    fn close(&mut self, why: RunClose) -> Option<RunClose> {
        self.open = false;
        self.closed_by = Some(why);
        Some(why)
    }

    /// Checks the IMU samples that arrived since the verified end (or since a suspect chunk),
    /// once at least [`RUN_CHUNK_NS`] of them are buffered, against the static initialization's
    /// checks with the spread ceilings raised to [`RUN_ACCEL_RMS_MAX`] and [`RUN_GYRO_RMS_MAX`].
    /// A chunk failing them makes the run suspect without moving its end; the next chunk passing
    /// clears it (the disturbance is left out of the gyro mean), failing too closes the run at the
    /// end before the first, recording the two failed checks in [`Self::imu_failures`]. A hole
    /// closes it at once. Returns the reason when this call closed the run.
    pub fn advance_imu(
        &mut self,
        buffer: &ImuBuffer,
        gates: &ImuGates,
        cfg: &StaticInitConfig,
        epoch_ns: u64,
    ) -> Option<RunClose> {
        let check = RunImuCheck {
            chunk_ns: RUN_CHUNK_NS,
            accel_rms_max: RUN_ACCEL_RMS_MAX,
            gyro_rms_max: RUN_GYRO_RMS_MAX,
        };
        self.advance_imu_with(buffer, gates, cfg, epoch_ns, check)
    }

    /// [`Self::advance_imu`] with the chunk length and spread ceilings given (the tests replay the
    /// old ones).
    fn advance_imu_with(
        &mut self,
        buffer: &ImuBuffer,
        gates: &ImuGates,
        cfg: &StaticInitConfig,
        epoch_ns: u64,
        check: RunImuCheck,
    ) -> Option<RunClose> {
        if !self.open {
            return None;
        }
        let from = self.suspect_end_ns.unwrap_or(self.imu_end_ns);
        let newest = buffer.newest_stamp_ns()?;
        if newest < from.saturating_add(check.chunk_ns) {
            return None;
        }
        let Ok(window) = buffer.window(from, newest, gates) else {
            return self.close(RunClose::ImuHole);
        };
        let samples: Vec<ImuMeasurement> = window.map(|s| s.to_measurement(epoch_ns)).collect();
        if samples.len() < RUN_CHUNK_MIN_SAMPLES {
            return self.close(RunClose::ImuHole);
        }
        let Some(stats) = ImuWindowStats::from_samples(&samples) else {
            return self.close(RunClose::ImuHole);
        };
        let cfg = StaticInitConfig {
            accel_rms_max: cfg.accel_rms_max.max(check.accel_rms_max),
            gyro_rms_max: cfg.gyro_rms_max.max(check.gyro_rms_max),
            ..*cfg
        };
        if let Err(fail) = check_imu_stats(&stats, &cfg) {
            if let Some(first) = self.suspect_fail {
                self.imu_failures = Some([first, fail]);
                return self.close(RunClose::Imu);
            }
            self.suspect_end_ns = Some(newest);
            self.suspect_fail = Some(fail);
            return None;
        }
        let span_s = (newest - from) as f64 * 1e-9;
        self.gyro_dt_sum += stats.mean_gyro * span_s;
        self.covered_s += span_s;
        self.suspect_end_ns = None;
        self.suspect_fail = None;
        self.imu_end_ns = newest;
        None
    }

    /// Adds the wheel motion since the verified end. The path floors apply per
    /// [`RUN_WHEEL_WINDOW_NS`] window, the same floors to the net displacement since the keyframe.
    /// A gap in the readings is bridged once the IMU has verified the stretch still (until then
    /// the end waits); a discontinuity closes the run. Returns the reason when this call closed
    /// it.
    pub fn advance_wheels(&mut self, wheels: &WheelState) -> Option<RunClose> {
        if !self.open {
            return None;
        }
        let newest = wheels.newest_stamp_ns()?;
        if newest <= self.wheel_end_ns {
            return None;
        }
        let d = match wheels.delta(self.wheel_end_ns, newest, 0) {
            Ok(d) => d,
            Err(WheelWindowError::Gap) if self.imu_end_ns < newest => return None,
            Err(WheelWindowError::Gap) => match wheels.delta_across_gaps(self.wheel_end_ns, newest)
            {
                Ok(d) => d,
                Err(_) => return self.close(RunClose::WheelGap),
            },
            Err(_) => return self.close(RunClose::WheelGap),
        };
        self.path_m += d.path_m;
        self.path_rad += d.path_rad;
        let [x, y, theta] = self.net;
        let (sin, cos) = theta.sin_cos();
        let [dx, dy, dtheta] = d.delta;
        self.net = [
            x + cos * dx - sin * dy,
            y + sin * dx + cos * dy,
            theta + dtheta,
        ];
        let net_m = self.net[0].hypot(self.net[1]);
        // A NaN path is not parked.
        if !(self.path_m <= PARKED_PATH_M
            && self.path_rad <= PARKED_PATH_RAD
            && net_m <= PARKED_PATH_M
            && self.net[2].abs() <= PARKED_PATH_RAD)
        {
            return self.close(RunClose::Wheels);
        }
        self.wheel_end_ns = newest;
        if newest - self.window_start_ns >= RUN_WHEEL_WINDOW_NS {
            self.window_start_ns = newest;
            self.path_m = 0.0;
            self.path_rad = 0.0;
        }
        None
    }
}

/// Where the edge into the next keyframe starts, and what it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum EdgeAnchor {
    /// An IMU edge from the last keyframe's stamp: what every keyframe got before.
    Keyframe(u64),
    /// A stationary factor; the wheel interval (zero motion) starts at `anchor_ns`.
    Stationary {
        anchor_ns: u64,
        span_s: f64,
        gyro: Option<StationaryGyro>,
    },
    /// An IMU edge from `anchor_ns`; the robot was still from the keyframe to there.
    Tail { anchor_ns: u64, elided_s: f64 },
}

impl EdgeAnchor {
    /// The stamp the IMU and wheel intervals of the edge start at.
    pub fn start_ns(&self) -> u64 {
        match *self {
            Self::Keyframe(t) => t,
            Self::Stationary { anchor_ns, .. } | Self::Tail { anchor_ns, .. } => anchor_ns,
        }
    }
}

/// The anchor of the edge from keyframe `(kf_idx, kf_ns)` to a keyframe at `stamp_ns`, given the
/// run, whether stationary mode is active, and whether the camera is within the parked floors of
/// the keyframe.
///
/// `Keyframe` unless stationary mode is active, the run starts at that keyframe and the still
/// part (to `stamp_ns` while the run covers it, to its frozen end after) is at least `min_span`.
/// Then `Stationary` while the run covers `stamp_ns` and the camera is still, `Tail` otherwise,
/// both anchored `STATIONARY_TAIL_MIN` before `stamp_ns`, clamped into `[kf_ns, end]` and never
/// earlier than an anchor the rings were already pruned to. A run whose rings were pruned past
/// the keyframe never goes back to `Keyframe`: those samples are gone.
#[allow(clippy::too_many_arguments)]
pub(crate) fn edge_anchor(
    run: Option<&StationaryRun>,
    kf_idx: usize,
    kf_ns: u64,
    stamp_ns: u64,
    active: bool,
    camera_still: bool,
    min_span_ns: u64,
    max_gap_ns: u64,
) -> EdgeAnchor {
    let Some(run) = run.filter(|r| r.kf_idx == kf_idx && r.start_ns == kf_ns) else {
        return EdgeAnchor::Keyframe(kf_ns);
    };
    let elided = |anchor_ns: u64| (anchor_ns - kf_ns) as f64 * 1e-9;
    let fallback = if run.committed_ns > kf_ns {
        EdgeAnchor::Tail {
            anchor_ns: run.committed_ns,
            elided_s: elided(run.committed_ns),
        }
    } else {
        EdgeAnchor::Keyframe(kf_ns)
    };
    if !active {
        return fallback;
    }
    let covered = run.covers(stamp_ns, max_gap_ns);
    let still_to = if covered { stamp_ns } else { run.end_ns() };
    if still_to.saturating_sub(kf_ns) < min_span_ns {
        return fallback;
    }
    let anchor_ns = stamp_ns
        .saturating_sub(STATIONARY_TAIL_MIN_NS)
        .clamp(kf_ns, run.end_ns().max(kf_ns))
        .max(run.committed_ns);
    if covered && camera_still {
        EdgeAnchor::Stationary {
            anchor_ns,
            span_s: stamp_ns.saturating_sub(kf_ns) as f64 * 1e-9,
            gyro: run.gyro(),
        }
    } else {
        EdgeAnchor::Tail {
            anchor_ns,
            elided_s: elided(anchor_ns),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::RawImuSample;
    use crate::wheel::{WheelOdomConfig, WheelSample};
    use cu29::units::si::f64::Frequency;
    use cu29::units::si::frequency::hertz;
    use kornia_3d::pose::Pose3d;
    use kornia_sensors::imu::ImuCalib;

    const MS: u64 = 1_000_000;
    const T0: u64 = 1_000_000_000_000;
    const BIAS: [f64; 3] = [0.003, -0.002, 0.001];

    fn calib() -> ImuCalib {
        ImuCalib {
            gyro_noise: 0.00012,
            accel_noise: 0.0019,
            gyro_bias_noise: 0.000002,
            accel_bias_noise: 0.00004,
        }
    }

    fn still_cfg() -> StaticInitConfig {
        StaticInitConfig::from_noise(&calib(), Frequency::new::<hertz>(200.0))
    }

    fn gates() -> ImuGates {
        ImuGates::new(Frequency::new::<hertz>(200.0))
    }

    /// 200 Hz samples over `[from, to)` ms after `T0`: gravity on z, the gyro bias, small
    /// deterministic noise, plus `extra(k)` (accel, gyro).
    fn imu(
        from_ms: u64,
        to_ms: u64,
        extra: impl Fn(u64) -> ([f64; 3], [f64; 3]),
    ) -> Vec<RawImuSample> {
        (from_ms / 5..to_ms / 5)
            .map(|k| {
                let j = k as f64;
                let n = [(1.7 * j).sin(), (2.9 * j).sin(), (4.3 * j).sin()];
                let (ea, eg) = extra(k);
                RawImuSample {
                    stamp_ns: T0 + k * 5 * MS,
                    gyro: [0, 1, 2].map(|i| BIAS[i] + 0.002 * n[i] + eg[i]),
                    accel: [0, 1, 2].map(|i| [0.0, 0.0, 9.81][i] + 0.02 * n[i] + ea[i]),
                }
            })
            .collect()
    }

    fn still(_: u64) -> ([f64; 3], [f64; 3]) {
        ([0.0; 3], [0.0; 3])
    }

    fn wheels(readings: &[(u64, f64)]) -> WheelState {
        let mut w =
            WheelState::new(WheelOdomConfig::new(Pose3d::IDENTITY, 0.01, 0.01).expect("valid"));
        w.push(readings.iter().map(|&(ms, x)| WheelSample {
            stamp_ns: T0 + ms * MS,
            x,
            y: 0.0,
            theta: 0.0,
        }));
        w
    }

    /// Feeds `samples` in 100 ms batches, advancing after each; the closure, if any.
    fn run_imu(run: &mut StationaryRun, samples: Vec<RawImuSample>) -> Option<RunClose> {
        run_imu_with(run, samples, NEW).map(|(why, _)| why)
    }

    /// The shipped check, and the one before 2026-10-12 (250 ms chunks, the static
    /// initialization's spread ceilings).
    const NEW: RunImuCheck = RunImuCheck {
        chunk_ns: RUN_CHUNK_NS,
        accel_rms_max: RUN_ACCEL_RMS_MAX,
        gyro_rms_max: RUN_GYRO_RMS_MAX,
    };
    const OLD: RunImuCheck = RunImuCheck {
        chunk_ns: 250 * MS,
        accel_rms_max: 0.0,
        gyro_rms_max: 0.0,
    };

    /// [`run_imu`] with `check`; the closure and the newest sample's time when it closed, ms
    /// after `T0`.
    fn run_imu_with(
        run: &mut StationaryRun,
        samples: Vec<RawImuSample>,
        check: RunImuCheck,
    ) -> Option<(RunClose, u64)> {
        let mut b = ImuBuffer::with_capacity(samples.len().max(1));
        for chunk in samples.chunks(20) {
            b.push(chunk.iter().copied(), 0);
            if let Some(why) = run.advance_imu_with(&b, &gates(), &still_cfg(), T0, check) {
                return Some((why, (b.newest_stamp_ns()? - T0) / MS));
            }
        }
        None
    }

    /// Still samples keep the run open to the newest one, with the gyro mean the bias and the
    /// covered time the chunks'; a gyro spread, a slow turn and an accelerometer spread over two
    /// chunks each close it, frozen before the first; one disturbed chunk does not; a hole
    /// closes it at once.
    ///
    /// Mutations (each turned this red): accumulating the chunk mean without its span
    /// (`covered_s += 1`), the covered-time assert; closing on the first failing chunk, the
    /// knock; recording the second failure twice, the spin's gates.
    #[test]
    fn the_imu_side_closes_on_each_kind_of_motion_and_on_a_hole() {
        let mut run = StationaryRun::new(1, T0);
        assert_eq!(run_imu(&mut run, imu(0, 10_005, still)), None);
        assert!(run.is_open());
        // Checked in chunks of at least 500 ms, fed in 100 ms batches: the end is within one
        // chunk and a batch of the newest sample.
        assert!(run.imu_end_ns > T0 + 9_400 * MS && run.imu_end_ns <= T0 + 10_000 * MS);
        let g = run.gyro().expect("chunks checked");
        let span = (run.imu_end_ns - T0) as f64 * 1e-9;
        assert!(
            (g.covered_s - span).abs() < 1e-9,
            "{} vs {span}",
            g.covered_s
        );
        for (i, b) in BIAS.iter().enumerate() {
            let m = [g.mean.x, g.mean.y, g.mean.z][i];
            assert!((m - b).abs() < 2e-4, "axis {i}: {m} vs {b}");
        }

        // One second of motion: two chunks of it, whatever the phase.
        let at = |k0: u64| move |k: u64| k >= k0 && k < k0 + 200;
        let spin = at(1000);
        for (extra, gate) in [
            (
                Box::new(move |k| ([0.0; 3], [0.0, 0.0, if spin(k) { 0.2 } else { 0.0 }]))
                    as Box<dyn Fn(u64) -> ([f64; 3], [f64; 3])>,
                ["GyroSpread", "Turning"],
            ),
            (
                Box::new(|k| ([0.0; 3], [0.0, if k >= 1000 { 0.03 } else { 0.0 }, 0.0])),
                ["Turning", "Turning"],
            ),
            (
                Box::new(move |k| {
                    (
                        [if spin(k) { 0.5 * (k as f64).sin() } else { 0.0 }, 0.0, 0.0],
                        [0.0; 3],
                    )
                }),
                ["AccelSpread", "AccelSpread"],
            ),
        ] {
            let mut run = StationaryRun::new(1, T0);
            assert_eq!(
                run_imu(&mut run, imu(0, 10_005, extra)),
                Some(RunClose::Imu)
            );
            assert!(!run.is_open());
            // The failed checks are kept for the log, each with its value: the spin's onset
            // chunk fails on the spread, the steady spin on the rate.
            let fails = run.imu_failures.expect("recorded");
            assert!(
                (0..2).all(|i| format!("{:?}", fails[i]).starts_with(gate[i])),
                "{gate:?}: {fails:?}"
            );
            // The IMU side alone (no wheels here): `end_ns` is the sooner of the two.
            let end = run.imu_end_ns;
            // The 4.6-5.1 s chunk holds the motion's first 95 ms, which passes unless it is
            // the spin.
            assert!(
                (T0 + 4_500 * MS..=T0 + 5_100 * MS).contains(&end),
                "frozen at the chunk before the motion or the one it starts in: {}",
                (end - T0) / MS
            );
            assert!(!run.covers(T0 + 5_000 * MS, 0));
        }

        // One disturbed chunk (a 50 ms knock inside the 5.4-6.0 s chunk) is not motion: the run
        // stays open, its end passes the knock, and the knock stays out of the gyro mean.
        let knock = |k: u64| {
            let on = (1_100..1_110).contains(&k);
            (
                [if on { 2.0 } else { 0.0 }, 0.0, 0.0],
                [0.0, 0.0, if on { 0.5 } else { 0.0 }],
            )
        };
        let mut run = StationaryRun::new(1, T0);
        assert_eq!(run_imu(&mut run, imu(0, 10_005, knock)), None);
        assert!(run.is_open());
        assert!(run.imu_end_ns > T0 + 9_400 * MS);
        let g = run.gyro().expect("chunks checked");
        assert!((g.mean.z - BIAS[2]).abs() < 2e-4, "{}", g.mean.z);

        // A 100 ms hole in the samples.
        let mut holed = imu(0, 4_000, still);
        holed.extend(imu(4_100, 8_000, still));
        let mut run = StationaryRun::new(1, T0);
        assert_eq!(run_imu(&mut run, holed), Some(RunClose::ImuHole));
    }

    /// The robot's own vibration while parked, as recorded on 2026-10-12: 38.5 Hz on the gyro
    /// x axis (head pitch) at 0.075 rad/s and on the accelerometer at 0.18 m/s^2 on two axes,
    /// which puts a chunk's spreads at the worst measured over 500 ms at any phase (0.053 rad/s,
    /// 0.179 m/s^2), in bursts of 2.4, 3.6 and 4 s; `scale` times that.
    fn vibration(scale: f64, on: impl Fn(u64) -> bool) -> impl Fn(u64) -> ([f64; 3], [f64; 3]) {
        move |k| {
            if !on(k) {
                return ([0.0; 3], [0.0; 3]);
            }
            let s = (2.0 * std::f64::consts::PI * 38.5 * k as f64 * 0.005).sin();
            let s = scale * s;
            ([0.0, 0.18 * s, 0.18 * s], [0.075 * s, 0.0, 0.0])
        }
    }

    /// One burst every 30 s for ten minutes, cycling through the recorded durations.
    fn bursty(k: u64) -> bool {
        let t_ms = k * 5;
        let len_ms = [2_400, 3_600, 4_000][(t_ms / 30_000 % 3) as usize];
        (10_000..10_000 + len_ms).contains(&(t_ms % 30_000))
    }

    /// Ten minutes parked with a vibration burst every 30 s stay open on the shipped check. The
    /// old check (250 ms chunks, the initialization's ceilings) closes on the first burst, as on
    /// the robot, and so do 500 ms chunks alone: a steady vibration's spread does not shrink with
    /// the window, so the chunk is not the fix by itself. Nor does it pin the chunk: a steady
    /// burst passes 250 ms chunks under the raised ceilings too, as it did in the replay of the
    /// recorded ten minutes; [`a_short_disturbance_inside_a_chunk_is_not_motion`] pins it.
    ///
    /// Mutations (each turned this red): `RUN_GYRO_RMS_MAX` at 0.04, under the burst; the run's
    /// check keeping the static accelerometer ceiling (the stronger bursts).
    #[test]
    fn the_robots_own_vibration_does_not_close_the_run() {
        let samples = imu(0, 600_005, vibration(1.0, bursty));
        // The bursts are at the measured level: over the static gyro ceiling, the accelerometer
        // spread at the static one, and both under the run's with the margin claimed.
        let chunk: Vec<ImuMeasurement> = samples[2_000..2_100]
            .iter()
            .map(|s| s.to_measurement(T0))
            .collect();
        let stats = ImuWindowStats::from_samples(&chunk).expect("chunk");
        let cfg = still_cfg();
        assert!(
            (0.05..0.056).contains(&stats.gyro_rms_dev),
            "{}",
            stats.gyro_rms_dev
        );
        assert!(stats.gyro_rms_dev > 3.0 * cfg.gyro_rms_max);
        assert!(RUN_GYRO_RMS_MAX > 1.25 * stats.gyro_rms_dev);
        assert!(
            (0.9 * cfg.accel_rms_max..0.19).contains(&stats.accel_rms_dev),
            "{}",
            stats.accel_rms_dev
        );
        assert!(RUN_ACCEL_RMS_MAX > 1.25 * stats.accel_rms_dev);

        let mut run = StationaryRun::new(1, T0);
        assert_eq!(run_imu_with(&mut run, samples.clone(), NEW), None);
        assert!(run.is_open());
        assert!(run.imu_end_ns > T0 + 599_400 * MS);
        // The bursts are left out of the gyro mean or average out of it.
        let g = run.gyro().expect("chunks checked");
        assert!((g.mean.x - BIAS[0]).abs() < 5e-4, "{}", g.mean.x);
        // Bursts 1.2x stronger than any recorded (the ones of the first run on 2026-10-12 were
        // not), the accelerometer spread over the static ceiling: still open.
        let mut run = StationaryRun::new(1, T0);
        let stronger = imu(0, 600_005, vibration(1.2, bursty));
        assert_eq!(run_imu_with(&mut run, stronger, NEW), None);

        for (check, name) in [
            (OLD, "250 ms, static ceiling"),
            (
                RunImuCheck {
                    chunk_ns: RUN_CHUNK_NS,
                    accel_rms_max: 0.0,
                    gyro_rms_max: 0.0,
                },
                "500 ms, static ceiling",
            ),
        ] {
            let mut run = StationaryRun::new(1, T0);
            let closed = run_imu_with(&mut run, samples.clone(), check);
            assert!(
                matches!(closed, Some((RunClose::Imu, t)) if (10_000..12_000).contains(&t)),
                "{name}: {closed:?}"
            );
        }
    }

    /// The 500 ms chunk the owner asked for: a zero-mean disturbance 400 ms long (a knock, a burst
    /// of footsteps, well over the raised gyro ceiling) is a single failing chunk whenever it
    /// lies inside one, and closes the run only when it straddles a boundary enough to fail both
    /// sides. At 250 ms it spans two chunks at every phase and closes the run every time.
    ///
    /// Mutation: `RUN_CHUNK_NS` back at 250 ms turned this red.
    #[test]
    fn a_short_disturbance_inside_a_chunk_is_not_motion() {
        let knock = |onset: u64| {
            move |k: u64| {
                let on = (onset..onset + 80).contains(&k);
                let s = (2.0 * std::f64::consts::PI * 38.5 * k as f64 * 0.005).sin();
                ([0.0; 3], [if on { 0.3 * s } else { 0.0 }, 0.0, 0.0])
            }
        };
        let short = RunImuCheck {
            chunk_ns: 250 * MS,
            ..NEW
        };
        let phases: Vec<u64> = (0..600).step_by(25).collect();
        let closed = |check: RunImuCheck| {
            phases
                .iter()
                .filter(|&&phase_ms| {
                    let mut run = StationaryRun::new(1, T0);
                    let onset = (5_000 + phase_ms) / 5;
                    run_imu_with(&mut run, imu(0, 8_005, knock(onset)), check).is_some()
                })
                .count()
        };
        let (at_500, at_250) = (closed(NEW), closed(short));
        assert_eq!(
            at_250,
            phases.len(),
            "250 ms closed {at_250} of {}",
            phases.len()
        );
        assert!(
            at_500 * 3 <= phases.len() * 2,
            "500 ms closed {at_500} of {}",
            phases.len()
        );
        // Not a blind spot: a phase that straddles a boundary still closes it.
        assert!(at_500 > 0, "500 ms closed none");
    }

    /// Real slow motion still closes the run within a bounded time, at every chunk phase and in
    /// the middle of a vibration burst: a 1.5 deg/s turn (just over the mean-rate gate) within
    /// 2.5 s of its onset, a 9 cm shove (1 m/s^2 for 0.3 s, then braking) within 1.5 s. A 1 cm/s
    /// creep never shows on the IMU at any chunk length; the wheels close on it within 0.75 s.
    ///
    /// Mutation: dropping the mean-rate gate from the run's check turned the turn red.
    #[test]
    fn slow_motion_still_closes_the_run_in_bounded_time() {
        let turn = |onset: u64| {
            move |k: u64| {
                let t = (k as f64 - onset as f64) * 0.005;
                let w = 1.5_f64.to_radians() * (t / 0.5).clamp(0.0, 1.0);
                ([0.0; 3], [0.0, 0.0, w])
            }
        };
        let shove = |onset: u64| {
            move |k: u64| {
                let a = match k.checked_sub(onset) {
                    Some(d) if d < 60 => 1.0,
                    Some(d) if d < 120 => -1.0,
                    _ => 0.0,
                };
                ([a, 0.0, 0.0], [0.0; 3])
            }
        };
        let buzz = vibration(1.0, |k| (1_000..1_800).contains(&k));
        for phase_ms in (0..600).step_by(50) {
            let onset_ms = 5_000 + phase_ms;
            let onset = onset_ms / 5;
            for (name, motion, bound_ms) in [
                (
                    "turn",
                    Box::new(turn(onset)) as Box<dyn Fn(u64) -> ([f64; 3], [f64; 3])>,
                    2_500,
                ),
                ("shove", Box::new(shove(onset)), 1_500),
            ] {
                for vibrating in [false, true] {
                    let extra = |k| {
                        let (ma, mg) = motion(k);
                        let (va, vg) = if vibrating {
                            buzz(k)
                        } else {
                            ([0.0; 3], [0.0; 3])
                        };
                        (
                            [0, 1, 2].map(|i| ma[i] + va[i]),
                            [0, 1, 2].map(|i| mg[i] + vg[i]),
                        )
                    };
                    let mut run = StationaryRun::new(1, T0);
                    let closed = run_imu_with(&mut run, imu(0, 15_005, extra), NEW);
                    assert!(
                        matches!(closed, Some((RunClose::Imu, t))
                            if t > onset_ms && t <= onset_ms + bound_ms),
                        "{name} at {onset_ms} ms, vibrating {vibrating}: {closed:?}"
                    );
                    // Frozen at most a chunk past the onset (a ramp under the gates is still).
                    assert!(run.imu_end_ns <= T0 + onset_ms * MS + RUN_CHUNK_NS + 100 * MS);
                }
            }
        }

        // A 1 cm/s creep, read every 50 ms like the wheel feed.
        let mut run = StationaryRun::new(1, T0);
        let mut w = wheels(&[]);
        let mut closed_at = None;
        for i in 0..100u64 {
            w.push([WheelSample {
                stamp_ns: T0 + i * 50 * MS,
                x: 0.01 * (i * 50) as f64 * 1e-3,
                y: 0.0,
                theta: 0.0,
            }]);
            if run.advance_wheels(&w).is_some() {
                closed_at = Some(i * 50);
                break;
            }
        }
        assert!(matches!(closed_at, Some(t) if t <= 750), "{closed_at:?}");
        assert_eq!(run.closed_by, Some(RunClose::Wheels));
    }

    /// `covers` allows the IMU end a chunk more lag than the wheels': it moves a chunk at a time;
    /// but not while a failing chunk is pending.
    ///
    /// Mutations (each turned this red): dropping the chunk from the IMU side, the trailing-IMU
    /// case; granting it while suspect, the suspect case.
    #[test]
    fn covers_allows_the_imu_one_chunk_of_lag() {
        let gap = 500 * MS;
        let mut run = open_run(10_000);
        run.imu_end_ns = T0 + 9_000 * MS;
        assert!(run.covers(T0 + 10_000 * MS, gap));
        assert!(!run.covers(T0 + 10_001 * MS, gap));
        run.suspect_end_ns = Some(T0 + 9_500 * MS);
        assert!(!run.covers(T0 + 9_600 * MS, gap), "suspect: no chunk");
        assert!(run.covers(T0 + 9_500 * MS, gap));
        run.suspect_end_ns = None;
        run.imu_end_ns = T0 + 10_000 * MS;
        run.wheel_end_ns = T0 + 9_400 * MS;
        assert!(
            !run.covers(T0 + 10_000 * MS, gap),
            "the wheels get no chunk"
        );
        assert!(run.covers(T0 + 9_900 * MS, gap));
        run.close(RunClose::Imu);
        assert!(!run.covers(T0 + 9_500 * MS, gap));
    }

    /// The wheel side: jitter-free parked readings keep it open; a creep past the parked floor
    /// within one 5 s window closes it; so does a creep too slow for any window whose NET
    /// displacement passes the floor; encoder jitter (one tick back and forth on every reading)
    /// for ten minutes does not, though its summed path is metres.
    ///
    /// Mutations (each turned this red): dropping the net check (the slow creep stayed open);
    /// never resetting the window path (the jitter closed the run, as the run-long sum did).
    #[test]
    fn the_wheel_side_judges_the_path_per_window_and_the_net_since_the_keyframe() {
        let mut run = StationaryRun::new(1, T0);
        let parked: Vec<(u64, f64)> = (0..40).map(|i| (i * 250, 0.0)).collect();
        let w = wheels(&parked);
        assert_eq!(run.advance_wheels(&w), None);
        assert_eq!(run.wheel_end_ns, T0 + 39 * 250 * MS);

        // Fed one reading at a time, like the tracker; the closure, if any.
        let feed = |x: &dyn Fn(u64) -> f64, n: u64| {
            let mut run = StationaryRun::new(1, T0);
            let mut w = wheels(&[]);
            let mut closed = None;
            for i in 0..n {
                w.push([WheelSample {
                    stamp_ns: T0 + i * 250 * MS,
                    x: x(i),
                    y: 0.0,
                    theta: 0.0,
                }]);
                closed = closed.or(run.advance_wheels(&w));
            }
            closed
        };
        // 2 mm/s: 10 mm in a 5 s window.
        assert_eq!(feed(&|i| 0.0005 * i as f64, 40), Some(RunClose::Wheels));
        // 0.4 mm/s: 2 mm per window, 6 mm net after 15 s.
        assert_eq!(feed(&|i| 0.0001 * i as f64, 80), Some(RunClose::Wheels));
        // 0.1 mm back and forth every 250 ms: 2 mm per window, 0 net, 240 mm summed.
        assert_eq!(
            feed(&|i| if i % 2 == 0 { 0.0 } else { 0.0001 }, 2_400),
            None
        );
    }

    /// A gap in the wheel readings (a stale head/body link) waits for the IMU and is bridged once
    /// the IMU has verified the stretch still; motion across the gap still closes the run.
    ///
    /// Mutation: closing on every gap (the old `WheelGap`) turned the bridged case red.
    #[test]
    fn a_wheel_gap_is_bridged_once_the_imu_vouches_for_it() {
        let gap = wheels(&[(0, 0.0), (250, 0.0), (2_000, 0.0)]);
        let mut run = StationaryRun::new(1, T0);
        run.imu_end_ns = T0 + 1_000 * MS;
        assert_eq!(run.advance_wheels(&gap), None);
        assert!(run.is_open());
        assert_eq!(run.wheel_end_ns, T0, "the end waits for the IMU");
        run.imu_end_ns = T0 + 2_100 * MS;
        assert_eq!(run.advance_wheels(&gap), None);
        assert!(run.is_open());
        assert_eq!(run.wheel_end_ns, T0 + 2_000 * MS, "bridged");

        let moved = wheels(&[(0, 0.0), (250, 0.0), (2_000, 0.02)]);
        let mut run = StationaryRun::new(1, T0);
        run.imu_end_ns = T0 + 2_100 * MS;
        assert_eq!(run.advance_wheels(&moved), Some(RunClose::Wheels));
    }

    fn open_run(end_ms: u64) -> StationaryRun {
        let mut run = StationaryRun::new(3, T0);
        run.imu_end_ns = T0 + end_ms * MS;
        run.wheel_end_ns = T0 + end_ms * MS;
        run.gyro_dt_sum = Vec3F64::new(0.003, 0.0, 0.0) * (end_ms as f64 * 1e-3);
        run.covered_s = end_ms as f64 * 1e-3;
        run
    }

    /// The anchor: `Keyframe` when stationary mode is off, the run is another keyframe's or the
    /// still part is under `min_span`; `Stationary` while covered with the camera still; `Tail`
    /// after the run closed or the camera moved; clamped into `[kf, end]`, monotone through
    /// `committed_ns`, and never `Keyframe` again once the rings were pruned past the keyframe.
    #[test]
    fn the_edge_anchor_follows_the_run() {
        let (min_span, gap) = (1_000 * MS, 500 * MS);
        let at = |ms: u64| T0 + ms * MS;
        let run = open_run(9_800);
        let anchor = |run: &StationaryRun, kf: usize, stamp: u64, active: bool, cam: bool| {
            edge_anchor(Some(run), kf, T0, stamp, active, cam, min_span, gap)
        };
        assert_eq!(
            anchor(&run, 3, at(10_000), false, true),
            EdgeAnchor::Keyframe(T0)
        );
        assert_eq!(
            anchor(&run, 4, at(10_000), true, true),
            EdgeAnchor::Keyframe(T0)
        );
        assert_eq!(
            edge_anchor(None, 3, T0, at(10_000), true, true, min_span, gap),
            EdgeAnchor::Keyframe(T0)
        );
        // 0.9 s still: under the minimum span.
        assert_eq!(
            anchor(&open_run(900), 3, at(900), true, true),
            EdgeAnchor::Keyframe(T0)
        );

        let EdgeAnchor::Stationary {
            anchor_ns,
            span_s,
            gyro,
        } = anchor(&run, 3, at(10_000), true, true)
        else {
            panic!("covered and still");
        };
        assert_eq!(anchor_ns, at(9_500));
        assert!((span_s - 10.0).abs() < 1e-9);
        assert!((gyro.unwrap().mean.x - 0.003).abs() < 1e-12);

        assert!(matches!(
            anchor(&run, 3, at(10_000), true, false),
            EdgeAnchor::Tail { anchor_ns, .. } if anchor_ns == at(9_500)
        ));

        // Closed at 6 s: the anchor freezes at the end, whatever the stamp.
        let mut closed = run.clone();
        closed.close(RunClose::Imu);
        closed.imu_end_ns = at(6_000);
        closed.wheel_end_ns = at(6_000);
        let EdgeAnchor::Tail {
            anchor_ns,
            elided_s,
        } = anchor(&closed, 3, at(20_000), true, true)
        else {
            panic!("closed");
        };
        assert_eq!(anchor_ns, at(6_000));
        assert!((elided_s - 6.0).abs() < 1e-9);

        // Pruned to 9.5 s: off or short, it stays a tail from there.
        let mut committed = open_run(900);
        committed.committed_ns = at(400);
        assert!(matches!(
            anchor(&committed, 3, at(900), false, true),
            EdgeAnchor::Tail { anchor_ns, .. } if anchor_ns == at(400)
        ));
        let mut ahead = run.clone();
        ahead.committed_ns = at(9_700);
        assert_eq!(
            anchor(&ahead, 3, at(10_000), true, true).start_ns(),
            at(9_700)
        );
    }

    /// Startup validation: the defaults pass; a non-positive sigma or Huber threshold and a
    /// minimum span under the tail are refused, naming the field.
    #[test]
    fn the_config_validates_each_field() {
        StationaryEdgeConfig::new().validate().unwrap();
        let bad = |f: fn(&mut StationaryEdgeConfig)| {
            let mut c = StationaryEdgeConfig::new();
            f(&mut c);
            c.validate()
        };
        for (f, field) in [
            (
                (|c: &mut StationaryEdgeConfig| c.translation_sigma_m = 0.0)
                    as fn(&mut StationaryEdgeConfig),
                "translation_sigma_m",
            ),
            (|c| c.rotation_sigma_rad = f64::NAN, "rotation_sigma_rad"),
            (|c| c.velocity_sigma_m_s = -1.0, "velocity_sigma_m_s"),
            (|c| c.huber_chi2 = Some(0.0), "huber_chi2"),
            (|c| c.min_span = Time::new::<second>(0.4), "min_span_s"),
        ] {
            match bad(f) {
                Err(VioError::StationaryParamInvalid { field: got, .. }) => assert_eq!(got, field),
                other => panic!("{field}: {other:?}"),
            }
        }
        let mut quad = StationaryEdgeConfig::new();
        quad.huber_chi2 = None;
        quad.validate().unwrap();
        let f = quad.factor(1, 2, 30.0, None, calib());
        assert!(f.is_valid());
        assert_eq!(f.huber_chi2, None);
        assert_eq!((f.from_idx, f.to_idx), (1, 2));
    }
}
