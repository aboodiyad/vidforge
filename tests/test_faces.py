"""Face detection decoding, tracking, alignment, blending and estimates (no models needed)."""

import numpy as np
import pytest

from vidforge.core import media
from vidforge.features.face import align, detect, enhancer, estimate, service, track
from vidforge.features.face.detect import Face


# --- detection --------------------------------------------------------------------------

def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[10, 10, 50, 50], [12, 12, 50, 50], [200, 200, 40, 40]], float)
    scores = np.array([0.8, 0.95, 0.7])
    assert sorted(detect.nms(boxes, scores, 0.3)) == [1, 2]


def test_decode_one_face_cell():
    """One confident cell at row 10, col 20 of the stride-8 grid."""
    sizes = [6400, 1600, 400]
    outs = ([np.zeros((1, n, 1), np.float32) for n in sizes] * 2
            + [np.zeros((1, n, 4), np.float32) for n in sizes]
            + [np.zeros((1, n, 10), np.float32) for n in sizes])
    outs = [o.copy() for o in outs]
    i = 10 * 80 + 20
    outs[0][0, i, 0] = 0.9  # cls
    outs[3][0, i, 0] = 0.9  # obj
    outs[6][0, i] = [0.5, 0.5, np.log(4), np.log(5)]  # centre in the middle of the cell, 32x40 px
    outs[9][0, i] = [0.2, 0.3, 0.8, 0.3, 0.5, 0.5, 0.3, 0.8, 0.7, 0.8]
    boxes, lms, scores = detect.decode(outs, 0.6)
    assert len(scores) == 1 and scores[0] == pytest.approx(0.9, abs=1e-6)
    cx, cy = (20 + 0.5) * 8, (10 + 0.5) * 8
    np.testing.assert_allclose(boxes[0], [cx - 16, cy - 20, 32, 40], atol=1e-4)
    np.testing.assert_allclose(lms[0][0], [(20 + 0.2) * 8, (10 + 0.3) * 8], atol=1e-4)


def _face(x, y, size=100.0, jitter=(0.0, 0.0)):
    lm = align.FFHQ_512 / 512 * size + (x, y) + np.array(jitter)
    return Face(box=np.array([x, y, size, size], float), landmarks=lm.astype(float), score=0.9)


# --- tracking ---------------------------------------------------------------------------

def test_one_euro_holds_still_and_follows_moves():
    f = track.OneEuro()
    rng = np.random.default_rng(0)
    still = [f(np.array([5.0]) + rng.normal(0, 0.05, 1), 1 / 25)[0] for _ in range(100)]
    assert np.std(still[20:]) < 0.05 * 0.5  # noise is damped
    f = track.OneEuro()
    out = [f(np.array([0.0 if i < 10 else 10.0]), 1 / 25)[0] for i in range(40)]
    assert out[-1] == pytest.approx(10.0, abs=0.3)  # a real move is followed


def test_tracker_keeps_identity_and_expires():
    tr = track.FaceTracker(max_missed=2)
    a = tr.update([_face(10, 10), _face(300, 10)], 1 / 25)
    ids = {round(t.face.box[0]): t.id for t in a}
    b = tr.update([_face(14, 12), _face(304, 11)], 1 / 25)
    assert {round(t.face.box[0]): t.id for t in b} == {14: ids[10], 304: ids[300]}
    for _ in range(3):
        tr.update([_face(16, 12)], 1 / 25)
    assert len(tr.tracks) == 1


def test_smoothing_reduces_jitter_of_a_still_face():
    """Regression: dividing by each frame's (wobbly) face size and multiplying back used to
    turn tiny scale wobble into big position jumps, so smoothing made faces shake MORE."""
    tr = track.FaceTracker()
    rng = np.random.default_rng(1)
    raw, smooth = [], []
    for _ in range(80):
        f = _face(400, 300)
        centre = f.landmarks.mean(axis=0)
        f.landmarks = centre + (f.landmarks - centre) * (1 + rng.normal(0, 0.005))  # detector scale wobble
        raw.append(f.landmarks.copy())
        smooth.append(tr.update([f], 1 / 25)[0].landmarks.copy())
    step = lambda seq: np.abs(np.diff(np.array(seq[10:]), axis=0)).mean()  # noqa: E731
    assert step(smooth) < step(raw) * 0.6


def test_scene_change():
    a = np.full((36, 64), 50, np.uint8)
    assert not track.scene_changed(None, a)
    assert not track.scene_changed(a, a + 5)
    assert track.scene_changed(a, a + 80)


# --- alignment and blending ---------------------------------------------------------------

def test_face_transform_maps_landmarks_to_template():
    lm = _face(123, 45, 180).landmarks
    m = align.face_transform(lm, 512)
    mapped = np.c_[lm, np.ones(5)] @ m.T
    np.testing.assert_allclose(mapped, align.FFHQ_512, atol=0.05)


def test_soft_mask_shape():
    m = align.soft_mask(512)
    assert m[300, 256] == pytest.approx(1.0) and m[5, 5] == 0.0 and m[300, 5] < 0.05


def test_paste_face_only_touches_face_region():
    frame = np.full((400, 600, 3), 40, np.uint8)
    lm = _face(250, 120, 120).landmarks
    m = align.face_transform(lm, 512)
    white = np.full((512, 512, 3), 240, np.uint8)
    align.paste_face(frame, white, m, align.soft_mask(512))
    eyes_mid = lm[:2].mean(axis=0).astype(int)
    assert frame[eyes_mid[1] + 15, eyes_mid[0]].mean() > 200  # face area replaced
    assert frame[10, 10].mean() == 40 and frame[390, 590].mean() == 40  # elsewhere untouched


def test_match_colors_moves_mean_only():
    orig = np.full((64, 64, 3), (180, 120, 100), np.uint8)
    rest = np.full((64, 64, 3), (120, 120, 160), np.uint8)
    rest[::2] += 10  # detail
    out = align.match_colors(rest, orig, np.ones((64, 64), np.float32))
    assert np.abs(out.reshape(-1, 3).mean(axis=0) - (180, 120, 100)).max() < 6
    assert out[0].mean() != out[1].mean()  # detail kept


# --- the per-frame enhancer with stand-in models --------------------------------------------

class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, frame):
        return [Face(f.box.copy(), f.landmarks.copy(), f.score) for f in self.faces]


class FakeRestorer:
    calls = 0

    def restore(self, crops, fidelity=0.7):
        FakeRestorer.calls += 1
        return [np.full_like(c, 230) for c in crops]


def test_enhancer_pastes_into_bigger_output_at_scaled_position():
    src = np.full((200, 300, 3), 60, np.uint8)
    face = _face(100, 40, 80)
    enh = enhancer.FaceEnhancer(FakeDetector([face]), FakeRestorer(),
                                enhancer.EnhanceSettings(color_match=False, temporal=0), 25)
    out = enh.process(src, np.full((400, 600, 3), 60, np.uint8))
    nose = (face.landmarks[2] * 2).astype(int)
    assert out[nose[1], nose[0]].mean() > 200  # face pasted where it is in the 2x frame
    assert out[390, 10].mean() == 60
    assert enh.stats.faces == 1 and enh.stats.frames == 1


def test_enhancer_skips_tiny_faces():
    src = np.full((200, 300, 3), 60, np.uint8)
    enh = enhancer.FaceEnhancer(FakeDetector([_face(10, 10, 8)]), FakeRestorer(), enhancer.EnhanceSettings(), 25)
    out = enh.process(src, src.copy())
    assert enh.stats.faces == 0 and np.array_equal(out, src)


def test_steady_blends_only_still_faces():
    enh = enhancer.FaceEnhancer(FakeDetector([]), FakeRestorer(), enhancer.EnhanceSettings(temporal=0.5), 25)
    crop = np.full((512, 512, 3), 100, np.uint8)
    mem = {}
    a = enh._steady(mem, crop, np.full_like(crop, 100))
    b = enh._steady(mem, crop, np.full_like(crop, 200))  # same input: carry over half
    assert a.mean() == 100 and 140 < b.mean() < 160
    moved = np.full_like(crop, 160)
    c = enh._steady(mem, moved, np.full_like(crop, 50))  # input changed a lot: no carry-over
    assert c.mean() == 50


# --- service and estimates ------------------------------------------------------------------

def _info(w=720, h=480, sar="10:11", dur=20.0):
    return media.parse_ffprobe({
        "format": {"filename": "x.mp4", "format_name": "mp4", "duration": str(dur)},
        "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264", "width": w, "height": h,
                     "avg_frame_rate": "25/1", "pix_fmt": "yuv420p", "sample_aspect_ratio": sar,
                     "bit_rate": "2000000"}],
    })


def test_keep_size_uses_square_pixels():
    assert service.keep_size(_info()) == (654, 480)


def test_face_estimates_order(isolated_home):
    info = _info()
    kw = dict(esrgan_model="general", codec="hevc", quality="balanced", container="mp4", stats={})
    g = estimate.estimate(info, 654, 480, background="keep", model="gfpgan", **kw)
    c = estimate.estimate(info, 654, 480, background="keep", model="codeformer", **kw)
    q = estimate.estimate(info, 1308, 960, background="quality", model="gfpgan", **kw)
    assert g.seconds < c.seconds  # Strong repair is slower
    assert g.seconds < q.seconds  # making it bigger adds time
    assert g.bytes < q.bytes
    assert g.download_mb > 300  # models not installed in the test data folder
