//! Copper tasks wrapping [kornia-slam](https://github.com/kornia/kornia-slam) stereo VIO.
//!
//! Deliberately separable layers:
//!
//! * [`track`]: the composition root over kornia-slam. kornia-slam ships stereo matching, the
//!   map, map-projection PnP and bundle adjustment, but its own orchestration
//!   (`kornia_slam::SlamSystem`) keeps its tracking state private and cannot be driven piece by
//!   piece; this is an orchestration over the same parts, written for a live camera. It uses
//!   no copper runtime or graph types (no `CuMsg`, no tasks), only the cu29 unit types, `CuTime`
//!   and the logging macros, so it runs without a graph.
//! * [`task`]: the `CuTask` shell that drives it from a copper graph. It owns only the
//!   payload-to-image conversion and the world-to-camera inversion, and must add no estimation
//!   logic of its own: anything it computed would be a second implementation to keep in step
//!   with any offline replay of [`track`].
//! * [`imu`]: the inertial ingest: the stamp-indexed sample ring, the coverage gates every
//!   interval must pass before it may be preintegrated, and the configuration the inertial
//!   path refuses to run without. Likewise free of copper runtime and graph types.
//! * [`stationary`]: the stillness detector that lets a parked robot initialize the inertial
//!   path from a still window instead of waiting to drive.
//! * [`parked`]: what a parked robot inserts once initialized: no keyframes on the time bound,
//!   stationary edges instead of IMU edges between parked keyframes, and bounded IMU and wheel
//!   rings while no keyframe comes. Off unless the `stationary_edges` block enables it.
//! * [`bus`], [`feed`], [`imu_channel`], [`reset`]: what reaches a BACKGROUNDED tracker without
//!   a second input edge. A [`VioBus`] resource bundle holds an [`ImuQueue`] that the inline
//!   [`ImuFeed`] sink fills from the graph's IMU edge, and a [`ResetEpoch`] any bound task can
//!   bump. One bus per estimator, so two estimators in one process stay independent.
//! * [`tap`], [`diag`]: what the tracker saw and concluded per frame, out through the bus to an
//!   inline [`VioTapSource`]: a status record, the stereo pair it solved and its keypoints.
//! * [`wheel`], [`wheel_channel`], [`wheel_feed`]: optional planar wheel odometry, the same seam
//!   as the IMU. It constrains local BA, replaces the constant-velocity prior, carries the pose
//!   through a vision loss, and keeps the world continuous across a map drop. Off unless the
//!   `wheel_odom` block enables it.
//! * [`veto`]: the hard motion veto, the last gate before a pose is published: a pose the gyro
//!   or the wheels contradict past hard bounds since the last published one is never published
//!   and never touches the map; the wheel and gyro dead reckoning is published instead.
//! * [`fusion`]: optional per-frame fused tracking: each frame's pose optimized against its map
//!   matches, the IMU and the wheels together, with a Mahalanobis gate that publishes the motion
//!   sensors' prediction when vision disagrees with them. Off unless `per_frame_fusion` enables it.
//!
//! That split is the reason this crate can be reviewed as a copper component rather than as a
//! slice of an application: the estimation is testable without a graph, and the graph binding
//! is small enough to read in one sitting.

#![deny(missing_docs)]

pub mod bus;
pub mod config;
pub mod consistency;
pub mod covariance;
pub mod diag;
pub mod error;
pub mod feed;
pub mod fusion;
pub mod imu;
pub mod imu_channel;
pub mod parked;
pub mod pose;
pub mod reset;
pub mod stationary;
mod stats;
pub mod tap;
pub mod task;
pub mod task_error;
pub mod track;
pub mod veto;
pub mod wheel;
pub mod wheel_channel;
pub mod wheel_feed;

pub use bus::VioBus;
pub use config::{AtKeyframeBound, OnTrackingLoss};
pub use consistency::{ConsistencyStats, MotionVerdict};
pub use error::VioError;
pub use feed::ImuFeed;
pub use fusion::{FusionConfig, FusionStats};
pub use imu::{BiasLimit, ImuGates, ImuWindowError, InertialConfig, InertialStats};
pub use imu_channel::ImuQueue;
pub use parked::{StationaryEdgeConfig, StationaryStats};
pub use reset::ResetEpoch;
pub use stationary::{NotStill, StaticInitConfig};
pub use tap::{MAX_TAP_FEATURES, VioTap, VioTapSource};
pub use task::{DEFAULT_LANDMARK_BUFFERS, MAX_LANDMARK_BUFFERS, MAX_LANDMARKS, StereoVio};
pub use task_error::{Eye, TaskError};
pub use track::{
    FrameReport, SystemMode, TrackStatus, TrackedPose, Tracker, TrackerConfig, TrackerStats,
};
pub use veto::{MotionVetoConfig, VetoStats};
pub use wheel::{WheelOdomConfig, WheelSample, WheelStats};
pub use wheel_channel::WheelQueue;
pub use wheel_feed::{MountFeed, WheelFeed};
