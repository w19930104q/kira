/*!
Modifies audio signals.

Any type that implements [`EffectBuilder`] can be added to a mixer track by
using [`TrackBuilder::add_effect`](crate::track::TrackBuilder::add_effect). Kira
comes with a number of commonly used effects.

If needed, you can create custom effects by implementing the [`EffectBuilder`]
and [`Effect`] traits.
*/

pub mod compressor;
pub mod delay;
pub mod distortion;
pub mod eq_filter;
pub mod filter;
pub mod panning_control;
pub mod reverb;
pub mod volume_control;

use std::time::Duration;

use crate::{frame::Frame, info::Info};

/// Configures an effect.
pub trait EffectBuilder {
	/// Allows the user to control the effect from gameplay code.
	type Handle;

	/// Creates the effect and a handle to the effect.
	#[must_use]
	fn build(self) -> (Box<dyn Effect>, Self::Handle);
}

/// Receives input audio from a mixer track and outputs modified audio.
///
/// For performance reasons, avoid allocating and deallocating memory in any methods
/// of this trait besides [`on_change_sample_rate`](Effect::on_change_sample_rate).
#[allow(unused_variables)]
pub trait Effect: Send {
	/// Called when the effect is first sent to the renderer.
	fn init(&mut self, sample_rate: u32, internal_buffer_size: usize) {}

	/// Called when the sample rate of the renderer is changed.
	fn on_change_sample_rate(&mut self, sample_rate: u32) {}

	/// Called whenever a new batch of audio samples is requested by the backend.
	///
	/// This is a good place to put code that needs to run fairly frequently,
	/// but not for every single audio sample.
	fn on_start_processing(&mut self) {}

	/// Transforms a slice of input [`Frame`]s.
	///
	/// `dt` is the time between each frame (in seconds).
	fn process(&mut self, input: &mut [Frame], dt: f64, info: &Info);

	/// How long the effect can continue producing audio after all of its
	/// inputs have gone silent.
	///
	/// When a mixer track is removed, its sounds are stopped immediately and
	/// the effects are kept connected for this long so their tail rings out
	/// before the effect resources are dropped. Returning `None` means the
	/// effect has no tail and does not need extra time (beyond one processing
	/// chunk) to flush.
	///
	/// This is called on the audio thread when the track's removal begins, so
	/// the returned duration may reflect the effect's current settings. If a
	/// fixed or longer tail is needed, it can be configured explicitly through
	/// `TrackBuilder::removal_tail_duration`, which takes precedence over this
	/// value.
	fn tail_duration(&self) -> Option<Duration> {
		None
	}
}
