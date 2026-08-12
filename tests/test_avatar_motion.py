from __future__ import annotations

import json

import numpy as np
import pytest

from signsync.avatar import (
    Animation,
    FaceChannel,
    Pose,
    Rig,
    animation_to_dict,
    default_rig,
    export_animation,
    quat_from_axis_angle,
    quat_identity,
    quat_multiply,
    quat_slerp,
    quat_to_matrix,
    rig_to_dict,
)
from signsync.avatar.rig import Joint, quat_between
from signsync.errors import SignSyncError
from signsync.motion import (
    MotionConfig,
    MotionGenerator,
    ProceduralLibrary,
    RecordedLibrary,
    SignClip,
    ease_in_out,
    hand_travel,
    solve_two_bone,
    transition_frames,
)
from signsync.motion.ik import reach_wrist
from signsync.translation import EnglishToSign


@pytest.fixture(scope="module")
def rig() -> Rig:
    return default_rig()


@pytest.fixture(scope="module")
def generator(rig):
    return MotionGenerator(ProceduralLibrary(rig=rig), rig)


# --------------------------------------------------------------------------- quaternions


def test_slerp_endpoints_and_midpoint():
    a = quat_identity()
    b = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(quat_slerp(a, b, 0.0), a, atol=1e-6)
    np.testing.assert_allclose(quat_slerp(a, b, 1.0), b, atol=1e-6)

    mid = quat_slerp(a, b, 0.5)
    expected = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 4)
    np.testing.assert_allclose(mid, expected, atol=1e-5)


def test_slerp_takes_the_short_way_round():
    """q and -q are the same rotation; blending the long way spins the joint."""
    a = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.1)
    b = -quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.2)
    mid = quat_slerp(a, b, 0.5)
    angle = 2 * np.arccos(np.clip(abs(float(np.dot(mid, a))), -1, 1))
    assert angle < 0.3, "slerp went the long way around"


def test_slerp_moves_at_a_constant_angular_rate():
    """The property linear interpolation lacks, and the cause of "robotic" motion."""
    a = quat_identity()
    b = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 2.0)
    angles = [
        2 * np.arccos(np.clip(abs(float(np.dot(quat_slerp(a, b, t), a))), -1, 1))
        for t in np.linspace(0, 1, 11)
    ]
    steps = np.diff(angles)
    assert steps.std() < 0.02, f"angular rate is uneven: {steps}"


def test_quat_between_rotates_one_vector_onto_another():
    source = np.array([0.0, 1.0, 0.0])
    target = np.array([1.0, 0.0, 0.0])
    rotated = quat_to_matrix(quat_between(source, target)) @ source
    np.testing.assert_allclose(rotated, target, atol=1e-5)


def test_quat_between_handles_opposite_vectors():
    result = quat_to_matrix(quat_between(np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])))
    np.testing.assert_allclose(result @ np.array([0.0, 1.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-5)


def test_quat_multiply_composes_rotations():
    half = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    full = quat_multiply(half, half)
    rotated = quat_to_matrix(full) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [-1.0, 0.0, 0.0], atol=1e-5)


# --------------------------------------------------------------------------- rig


def test_default_rig_has_expressive_hands_and_a_face(rig):
    """Plan §8.8: without these the avatar cannot render intelligible USL."""
    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for segment in (1, 2, 3):
                assert f"{side}_{finger}_{segment}" in rig.names
    assert len(FaceChannel.ALL) >= 8


def test_forward_kinematics_places_joints_below_the_root(rig):
    positions = rig.forward_kinematics(rig.rest_pose())
    assert positions.shape == (len(rig), 3)
    assert np.isfinite(positions).all()
    head = positions[rig.index("head")]
    hips = positions[rig.index("hips")]
    assert head[1] < hips[1], "head should sit above the hips in image coordinates"


def test_rotating_a_parent_moves_its_children(rig):
    pose = rig.rest_pose()
    before = rig.forward_kinematics(pose)[rig.index("right_index_3")]
    pose.set(rig, "right_upper_arm", quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.8))
    after = rig.forward_kinematics(pose)[rig.index("right_index_3")]
    assert np.linalg.norm(after - before) > 0.1


def test_rig_rejects_a_child_declared_before_its_parent():
    with pytest.raises(SignSyncError, match="before its parent"):
        Rig((Joint("child", "parent", (0, 0, 0)), Joint("parent", None, (0, 0, 0))))


def test_rig_rejects_unknown_parent_and_duplicate_names():
    with pytest.raises(SignSyncError, match="unknown parent"):
        Rig((Joint("a", "ghost", (0, 0, 0)),))
    with pytest.raises(SignSyncError, match="duplicate"):
        Rig((Joint("a", None, (0, 0, 0)), Joint("a", None, (0, 0, 0))))


def test_unknown_joint_is_named_in_the_error(rig):
    with pytest.raises(SignSyncError, match="elbow"):
        rig.index("elbow")


def test_pose_rejects_malformed_rotations():
    with pytest.raises(SignSyncError, match="quaternions"):
        Pose(rotations=np.zeros((5, 3)))


# --------------------------------------------------------------------------- IK


def test_ik_puts_the_wrist_on_a_reachable_target(rig):
    pose = rig.rest_pose()
    target = rig.forward_kinematics(pose)[rig.index("right_upper_arm")] + np.array(
        [-0.3, 0.5, 0.4], dtype=np.float32
    )
    result = reach_wrist(rig, pose, "right", target)
    achieved = rig.forward_kinematics(pose)[rig.index("right_wrist")]

    assert result.reached
    assert np.linalg.norm(achieved - target) < 0.05, f"wrist landed at {achieved}, wanted {target}"


def test_ik_reports_an_unreachable_target_instead_of_pretending(rig):
    """A silently straightened arm looks like a different sign (plan §8.7)."""
    pose = rig.rest_pose()
    result = reach_wrist(rig, pose, "right", np.array([9.0, 9.0, 9.0], dtype=np.float32))
    assert not result.reached
    assert result.error > 0


def test_ik_is_deterministic(rig):
    target = np.array([-0.3, 0.4, 0.3], dtype=np.float32)
    a, b = rig.rest_pose(), rig.rest_pose()
    reach_wrist(rig, a, "right", target)
    reach_wrist(rig, b, "right", target)
    np.testing.assert_allclose(a.rotations, b.rotations)


def test_ik_validates_its_inputs(rig):
    with pytest.raises(SignSyncError, match="bone lengths"):
        solve_two_bone(np.zeros(3), np.ones(3), 0.0, 1.0)
    with pytest.raises(SignSyncError, match="side"):
        reach_wrist(rig, rig.rest_pose(), "middle", np.zeros(3))


# --------------------------------------------------------------------------- blending


def test_easing_is_smooth_at_the_endpoints():
    assert ease_in_out(0.0) == 0.0
    assert ease_in_out(1.0) == 1.0
    assert ease_in_out(0.5) == pytest.approx(0.5)
    # Slow at the ends is the whole point.
    assert ease_in_out(0.1) < 0.1
    assert ease_in_out(0.9) > 0.9


def test_transition_length_scales_with_distance(rig):
    near, far = rig.rest_pose(), rig.rest_pose()
    reach_wrist(rig, far, "right", np.array([-0.4, 0.3, 0.5], dtype=np.float32))

    short = transition_frames(rig, near, near.copy())
    long = transition_frames(rig, near, far)
    assert long > short, "a bigger movement must take longer, or it looks teleported"


def test_transition_length_is_bounded(rig):
    a, b = rig.rest_pose(), rig.rest_pose()
    reach_wrist(rig, b, "right", np.array([-0.5, 0.4, 0.6], dtype=np.float32))
    assert 2 <= transition_frames(rig, a, b, minimum=2, maximum=5) <= 5


def test_hand_travel_is_zero_for_identical_poses(rig):
    assert hand_travel(rig, rig.rest_pose(), rig.rest_pose()) == pytest.approx(0.0)


def test_transition_config_is_validated(rig):
    with pytest.raises(SignSyncError, match="minimum"):
        transition_frames(rig, rig.rest_pose(), rig.rest_pose(), minimum=5, maximum=2)


# --------------------------------------------------------------------------- generation


def test_generation_produces_a_continuous_animation(generator):
    motion = generator.generate(EnglishToSign().translate("Where is the hospital?"))
    animation = motion.animation

    assert animation.glosses == ("HOSPITAL", "WHERE")
    assert len(animation) > 0
    assert animation.duration > 0
    assert all(np.isfinite(p.rotations).all() for p in animation.poses)


def test_signs_are_separated_by_transitions_not_concatenated(generator, rig):
    """Playing clips back to back is the "robotic" failure plan §8.7 names."""
    motion = generator.generate(["HELLO", "HOSPITAL"])
    segments = motion.animation.segments
    assert len(segments) == 2
    gap = segments[1][0] - segments[0][1]
    assert gap > 0, "no transition inserted between signs"


def test_motion_is_continuous_frame_to_frame(generator, rig):
    """No teleporting hands: consecutive frames must not jump."""
    animation = generator.generate(["HELLO", "WATER", "HELP"]).animation
    positions = np.stack(
        [rig.forward_kinematics(p)[rig.index("right_wrist")] for p in animation.poses]
    )
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert steps.max() < 0.35, f"largest single-frame hand jump was {steps.max():.3f}"


def test_markers_are_applied_over_their_scope_only(generator):
    """A brow raise over a clause is a question; over one sign it is a topic."""
    sequence = EnglishToSign().translate("I do not understand.")
    animation = generator.generate(sequence).animation

    shake = [p.face.get(FaceChannel.HEAD_SHAKE, 0.0) for p in animation.poses]
    assert max(shake) > 0.5, "negation produced no head shake"
    assert min(shake) == 0.0, "the marker was applied to the whole animation regardless of scope"


def test_marker_channels_ramp_rather_than_switch(generator):
    sequence = EnglishToSign().translate("Where is the hospital?")
    animation = generator.generate(sequence).animation
    values = [p.face.get(FaceChannel.BROW_FURROW, 0.0) for p in animation.poses]
    active = [v for v in values if v > 0]
    assert len(set(np.round(active, 2))) > 1, "facial channel switched instantly"


def test_question_and_negation_use_different_channels(generator):
    question = generator.generate(EnglishToSign().translate("Where is the hospital?")).animation
    negation = generator.generate(EnglishToSign().translate("I do not understand.")).animation

    assert max(p.face.get(FaceChannel.BROW_FURROW, 0) for p in question.poses) > 0
    assert max(p.face.get(FaceChannel.HEAD_SHAKE, 0) for p in negation.poses) > 0


def test_missing_glosses_are_reported_not_approximated(rig):
    """Plan §8.7: never silently render motion for a sign we do not have."""
    empty = RecordedLibrary(rig=rig)
    motion = MotionGenerator(empty, rig).generate(["HELLO", "HOSPITAL"])

    assert motion.missing == ("HELLO", "HOSPITAL")
    assert not motion.is_complete
    assert len(motion.animation) == 0


def test_procedural_motion_is_labelled_as_generated(generator):
    """A Deaf evaluator reviewing avatar quality must know what is real."""
    motion = generator.generate(["HELLO"])
    assert motion.procedural == ("HELLO",)
    assert not motion.is_fully_recorded
    assert motion.is_complete


def test_recorded_clips_are_marked_as_recorded(rig):
    clip = SignClip(gloss="HELLO", poses=[rig.rest_pose()] * 5, source="signer-07")
    library = RecordedLibrary(rig=rig)
    library.add(clip)

    motion = MotionGenerator(library, rig).generate(["HELLO"])
    assert motion.is_fully_recorded
    assert clip.is_recorded


def test_library_coverage_reports_what_is_missing(rig):
    library = RecordedLibrary(rig=rig)
    library.add(SignClip("HELLO", [rig.rest_pose()], source="signer-01"))
    have, missing = library.coverage(["HELLO", "WATER", "HELP"])
    assert have == ["HELLO"]
    assert missing == ["WATER", "HELP"]


def test_recorded_library_roundtrips(tmp_path, rig):
    library = RecordedLibrary(rig=rig)
    library.add(SignClip("HELLO", [rig.rest_pose(), rig.rest_pose()], source="signer-02"))
    restored = RecordedLibrary.load(library.save(tmp_path / "clips.json"), rig)

    assert "HELLO" in restored
    assert restored.get("HELLO").source == "signer-02"
    assert len(restored.get("HELLO")) == 2


def test_empty_sequence_generates_an_empty_animation(generator):
    assert len(generator.generate([]).animation) == 0


def test_motion_config_is_validated():
    with pytest.raises(SignSyncError, match="fps"):
        MotionConfig(fps=0)
    with pytest.raises(SignSyncError, match="transition"):
        MotionConfig(min_transition_frames=10, max_transition_frames=2)


# --------------------------------------------------------------------------- export


def test_export_carries_the_rig_and_every_frame(generator, rig):
    animation = generator.generate(["HELLO", "WATER"]).animation
    payload = animation_to_dict(animation, rig)

    assert payload["version"] == 1
    assert len(payload["frames"]) == len(animation)
    assert len(payload["rig"]["joints"]) == len(rig)
    assert payload["rig"]["joints"][0]["parent"] is None
    assert [s["gloss"] for s in payload["segments"]] == ["HELLO", "WATER"]


def test_export_is_json_serialisable_and_finite(tmp_path, generator, rig):
    animation = generator.generate(["HELP"]).animation
    path = export_animation(animation, tmp_path / "a.json", rig)
    restored = json.loads(path.read_text())

    rotations = np.array(restored["frames"][0]["rotations"])
    assert np.isfinite(rotations).all()
    assert rotations.shape == (len(rig), 4)


def test_rig_export_lists_face_channels(rig):
    assert set(rig_to_dict(rig)["faceChannels"]) == set(FaceChannel.ALL)


def test_animation_gloss_lookup_by_time(generator):
    animation = generator.generate(["HELLO", "WATER"]).animation
    start, end, gloss = animation.segments[0]
    assert animation.gloss_at((start + end) / 2) == gloss
    assert animation.gloss_at(9_999) is None


def test_animations_with_different_frame_rates_cannot_be_joined():
    with pytest.raises(SignSyncError, match="fps"):
        Animation([], fps=30).concat(Animation([], fps=25))
