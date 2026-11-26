//! Tests for the coordinated teardown of mixer tracks.
//!
//! When a track's handle is dropped while sounds are playing:
//! - the sounds must be stopped immediately,
//! - the effects must stay connected until their tails have flushed,
//! - sounds and effects must be reclaimed together in a single commit.
//!
//! The sample track has three effects (filter, delay, reverb), two of which
//! carry stateful tails, plus a tracking effect that observes the chain.

use std::sync::{
	Arc,
	atomic::{AtomicU64, AtomicUsize, Ordering},
};

use kira::{
	AudioManager, AudioManagerSettings, Frame,
	backend::mock::{MockBackend, MockBackendSettings},
	effect::{
		Effect, EffectBuilder, delay::DelayBuilder, filter::FilterBuilder, reverb::ReverbBuilder,
	},
	info::Info,
	sound::PlaybackState,
	sound::static_sound::{StaticSoundData, StaticSoundSettings},
	track::TrackBuilder,
};
use rtrb::{Consumer, Producer, RingBuffer};

/// An effect that records what happens to it while it's alive and reports
/// when it is dropped, so the test can observe the effect chain's lifetime
/// from the user's side of the audio-thread boundary.
struct TrackingEffect {
	events: Producer<&'static str>,
	process_count: Arc<AtomicU64>,
	alive_count: Arc<AtomicUsize>,
}

impl Effect for TrackingEffect {
	fn on_start_processing(&mut self) {
		let _ = self.events.push("start");
	}

	fn process(&mut self, input: &mut [Frame], _dt: f64, _info: &Info) {
		self.process_count.fetch_add(1, Ordering::SeqCst);
		let _ = self
			.events
			.push(if input.iter().all(|frame| *frame == Frame::ZERO) {
				"silent"
			} else {
				"signal"
			});
	}
}

impl Drop for TrackingEffect {
	fn drop(&mut self) {
		self.alive_count.fetch_sub(1, Ordering::SeqCst);
		let _ = self.events.push("dropped");
	}
}

struct TrackingEffectHandle {
	events: Consumer<&'static str>,
	process_count: Arc<AtomicU64>,
}

struct TrackingEffectBuilder {
	alive_count: Arc<AtomicUsize>,
}

impl EffectBuilder for TrackingEffectBuilder {
	type Handle = TrackingEffectHandle;

	fn build(self) -> (Box<dyn Effect>, Self::Handle) {
		let (producer, consumer) = RingBuffer::new(1024);
		let process_count = Arc::new(AtomicU64::new(0));
		self.alive_count.fetch_add(1, Ordering::SeqCst);
		(
			Box::new(TrackingEffect {
				events: producer,
				process_count: process_count.clone(),
				alive_count: self.alive_count,
			}),
			TrackingEffectHandle {
				events: consumer,
				process_count,
			},
		)
	}
}

fn sound_data(num_frames: usize) -> StaticSoundData {
	StaticSoundData {
		sample_rate: 44100,
		frames: (0..num_frames)
			.map(|i| Frame::from_mono(if i % 2 == 0 { 0.5 } else { -0.5 }))
			.collect(),
		settings: StaticSoundSettings::default(),
		slice: None,
	}
}

fn drain(consumer: &mut Consumer<&'static str>) -> Vec<&'static str> {
	let mut events = Vec::new();
	while let Ok(event) = consumer.pop() {
		events.push(event);
	}
	events
}

/// The sample track: three built-in effects, two of which (delay and reverb)
/// keep audio after the sound source stops, with the tracking effect last in
/// the chain. `tail` overrides how long the effects are given to ring out
/// when the track is removed.
fn sample_track_builder(alive_count: &Arc<AtomicUsize>, tail: std::time::Duration) -> TrackBuilder {
	let mut builder = TrackBuilder::new().removal_tail_duration(tail);
	// layer 1: a stateless-tail filter
	builder.add_effect(FilterBuilder::new());
	// layer 2: a delay whose echo line carries sound past the source's end
	builder.add_effect(DelayBuilder::new().delay_time(std::time::Duration::from_millis(50)));
	// layer 3: a reverb with its own decaying tail
	builder.add_effect(ReverbBuilder::new());
	// observer at the end of the chain
	builder.add_effect(TrackingEffectBuilder {
		alive_count: alive_count.clone(),
	});
	builder
}

#[test]
fn removing_a_playing_track_coordinates_sound_stop_tail_and_reclamation() {
	let mut manager = AudioManager::<MockBackend>::new(AudioManagerSettings {
		backend_settings: MockBackendSettings { sample_rate: 1000 },
		internal_buffer_size: 100,
		..Default::default()
	})
	.unwrap();
	let alive_count = Arc::new(AtomicUsize::new(0));
	// the sample track has three effects; the reverb alone reports a ~2.4s
	// tail, which outlasts the 50ms builder override, so the flush is driven
	// by the effects' own tail durations
	let mut track = manager
		.add_sub_track(sample_track_builder(
			&alive_count,
			std::time::Duration::from_millis(50),
		))
		.unwrap();
	// a long source that will not finish on its own during the test, so
	// without explicit teardown it would keep playing forever
	let sound_handle = track.play(sound_data(1000)).unwrap();
	let backend = manager.backend_mut();

	// let the sound play and feed the effects' tails
	for _ in 0..3 {
		backend.on_start_processing();
		backend.process();
	}
	assert_eq!(alive_count.load(Ordering::SeqCst), 1);

	// remove the track while the sound is still playing
	drop(track);

	// the first processing pass after the removal request begins the
	// coordinated teardown: the sound is stopped immediately, but the
	// effects stay connected so the already-emitted tail has a destination
	backend.on_start_processing();
	backend.process();
	assert_eq!(
		sound_handle.state(),
		PlaybackState::Stopped,
		"the sound must stop in the same commit the teardown begins"
	);
	assert_eq!(
		alive_count.load(Ordering::SeqCst),
		1,
		"the effects must stay connected while the tail is flushing"
	);

	// keep processing until the flush completes; the effects keep ringing
	// out with silent input and are then reclaimed together with the track
	// in one commit
	let mut flush_passes = 0;
	while alive_count.load(Ordering::SeqCst) == 1 {
		backend.on_start_processing();
		backend.process();
		flush_passes += 1;
		assert!(
			flush_passes < 100,
			"the track should be reclaimed after the tail flushes, not kept alive forever"
		);
	}
	assert_eq!(alive_count.load(Ordering::SeqCst), 0);
	assert!(flush_passes > 1, "the tail should take more than one pass");

	// the sound handle must never have been usable as a live source again:
	// its state remains Stopped after the track is gone
	assert_eq!(sound_handle.state(), PlaybackState::Stopped);
}

#[test]
fn effects_process_silent_input_while_flushing_and_drop_once() {
	// a low sample rate keeps the reverb's ~2.4s tail to a manageable number
	// of processing passes, so the tracking effect's event ring never fills up
	let mut manager = AudioManager::<MockBackend>::new(AudioManagerSettings {
		backend_settings: MockBackendSettings { sample_rate: 1000 },
		internal_buffer_size: 100,
		..Default::default()
	})
	.unwrap();
	let alive_count = Arc::new(AtomicUsize::new(0));
	let mut effect_handle;
	let mut track = {
		let mut builder = TrackBuilder::new();
		// the observer sits first in the chain so it sees the track's raw
		// input: signal while the sound plays, silence once it is stopped.
		// the stateful tails live downstream and still get reclaimed with it
		effect_handle = builder.add_effect(TrackingEffectBuilder {
			alive_count: alive_count.clone(),
		});
		builder.add_effect(DelayBuilder::new().delay_time(std::time::Duration::from_millis(50)));
		builder.add_effect(ReverbBuilder::new());
		manager.add_sub_track(builder).unwrap()
	};
	track.play(sound_data(1000)).unwrap();
	let backend = manager.backend_mut();

	for _ in 0..3 {
		backend.on_start_processing();
		backend.process();
	}
	let before = effect_handle.process_count.load(Ordering::SeqCst);
	assert!(before > 0);
	assert!(drain(&mut effect_handle.events).contains(&"signal"));

	drop(track);
	let mut flush_passes = 0;
	while alive_count.load(Ordering::SeqCst) == 1 {
		backend.on_start_processing();
		backend.process();
		flush_passes += 1;
		assert!(flush_passes < 100, "the tail flush never completed");
	}

	let events = drain(&mut effect_handle.events);
	assert!(
		events.contains(&"silent"),
		"the effects must keep processing the silent tail input: {events:?}"
	);
	assert_eq!(
		events.iter().filter(|event| **event == "dropped").count(),
		1,
		"effect resources must be reclaimed exactly once: {events:?}"
	);
	assert_eq!(alive_count.load(Ordering::SeqCst), 0);
}

#[test]
fn paused_track_still_flushes_its_tail() {
	let mut manager = AudioManager::<MockBackend>::new(AudioManagerSettings {
		backend_settings: MockBackendSettings { sample_rate: 1000 },
		internal_buffer_size: 100,
		..Default::default()
	})
	.unwrap();
	let alive_count = Arc::new(AtomicUsize::new(0));
	let mut track = manager
		.add_sub_track(sample_track_builder(
			&alive_count,
			std::time::Duration::from_millis(50),
		))
		.unwrap();
	track.play(sound_data(1000)).unwrap();
	let backend = manager.backend_mut();
	backend.on_start_processing();
	backend.process();

	// pause the track, then remove it while paused: the flush must not get
	// stuck behind the paused playback state
	track.pause(kira::Tween::default());
	drop(track);
	for _ in 0..3 {
		backend.on_start_processing();
		backend.process();
	}

	let mut flush_passes = 0;
	while alive_count.load(Ordering::SeqCst) == 1 {
		backend.on_start_processing();
		backend.process();
		flush_passes += 1;
		assert!(
			flush_passes < 100,
			"the paused track's flush never completed"
		);
	}
	assert_eq!(alive_count.load(Ordering::SeqCst), 0);
}

#[test]
fn zero_tail_is_reclaimed_after_one_final_chunk() {
	let mut manager = AudioManager::<MockBackend>::new(AudioManagerSettings {
		backend_settings: MockBackendSettings { sample_rate: 44100 },
		internal_buffer_size: 10,
		..Default::default()
	})
	.unwrap();
	let alive_count = Arc::new(AtomicUsize::new(0));
	let mut builder = TrackBuilder::new()
		.with_effect(FilterBuilder::new())
		.removal_tail_duration(std::time::Duration::ZERO);
	builder.add_effect(TrackingEffectBuilder {
		alive_count: alive_count.clone(),
	});
	let track = manager.add_sub_track(builder).unwrap();
	let backend = manager.backend_mut();
	backend.on_start_processing();
	backend.process();
	assert_eq!(alive_count.load(Ordering::SeqCst), 1);

	drop(track);

	// the first pass starts the flush and gives the effects their final
	// chunk of processing; the track is not eligible for removal yet.
	backend.on_start_processing();
	backend.process();
	assert_eq!(alive_count.load(Ordering::SeqCst), 1);
	// the next pass's mixer cleanup removes the track and drops its sounds
	// and effects together, all within that one processing commit.
	backend.on_start_processing();
	backend.process();
	assert_eq!(alive_count.load(Ordering::SeqCst), 0);
}
