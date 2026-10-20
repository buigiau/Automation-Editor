from pathlib import Path
import numpy as np
from PIL import Image

from autoedit.ai.appearance import ReferenceAppearance


class People:
    def inspect(self, rgb, **kwargs):
        return {'people': 1, 'faces': [{'face': [20, 4, 30, 25]}], 'body_boxes': [[0, 0, 80, 80]]}


def costume(path, head=(220, 140, 100), clothes=(240, 110, 10)):
    rgb = np.full((80, 80, 3), clothes, dtype=np.uint8)
    rgb[4:29, 20:50] = head
    Image.fromarray(rgb).save(path)
    return rgb


def test_reference_profile_excludes_faces_and_compares_wardrobe_only(tmp_path):
    path = tmp_path/'reference.png'
    original = costume(path)
    before = path.read_bytes()
    appearance = ReferenceAppearance([{'id': 'r', 'path': str(path)}], People(), tmp_path/'cache')
    assert appearance.human_references
    changed = original.copy()
    changed[4:29, 20:50] = [10, 30, 250]
    data = People().inspect(changed)
    assert np.array_equal(appearance.profile(original, data), appearance.profile(changed, data))
    assert appearance.preview_score(changed, data) > .99
    unrelated = costume(tmp_path/'other.png', clothes=(10, 150, 240))
    assert appearance.preview_score(unrelated, data) < .25
    with Image.open(appearance.references[0]['path']) as masked:
        assert max(abs(c-128) for c in masked.getpixel((35, 15))) < 3
    assert path.read_bytes() == before


def test_target_images_are_masked_and_original_files_are_retained(tmp_path):
    reference = tmp_path/'reference.png'
    target = tmp_path/'target.png'
    costume(reference)
    costume(target)
    before = target.read_bytes()
    appearance = ReferenceAppearance([{'id': 'r', 'path': str(reference)}], People(), tmp_path/'cache')
    masked = appearance.masked(target)
    assert Path(masked).is_file() and Path(masked) != target
    assert target.read_bytes() == before
    with Image.open(masked) as image:
        assert max(abs(c-128) for c in image.getpixel((35, 15))) < 3


def test_group_screen_considers_outfits_beyond_the_largest_body(tmp_path):
    path = tmp_path/'reference.png'
    reference = costume(path)
    appearance = ReferenceAppearance([{'id': 'r', 'path': str(path)}], People(), tmp_path/'cache')
    group = np.full((120, 200, 3), [0, 80, 255], dtype=np.uint8)
    group[:80, 120:200] = reference
    data = {'people': 2, 'faces': [{'face': [140, 4, 30, 25]}],
            'body_boxes': [[0, 0, 110, 120], [120, 0, 80, 80]]}
    assert appearance.preview_score(group, data) > .99
    mask = appearance.face_mask(group, data)
    assert mask[20, 30] == 0 and mask[15, 155] == 0


def test_matching_one_accessory_colour_cannot_replace_a_two_colour_costume():
    target = np.zeros((16, 4), np.float32)
    target[1, 3], target[11, 3] = .6, .4  # Orange garment, purple accessories.
    wrong = np.zeros_like(target)
    wrong[8, 3], wrong[11, 3] = .8, .2  # Blue/white outfit, purple accessory.
    assert not ReferenceAppearance.palette_compatible(wrong, target)
    correct = np.zeros_like(target)
    correct[1, 3], correct[11, 3], correct[8, 3] = .3, .2, .5
    assert ReferenceAppearance.palette_compatible(correct, target)


def test_only_substantial_reference_costume_colours_are_supported():
    appearance = object.__new__(ReferenceAppearance)
    profile = np.zeros((16,4),np.float32)
    profile[1,3], profile[11,3], profile[8,3] = .6,.36,.04
    appearance.profiles = [profile]
    appearance.profile_reference_ids = ['r']
    assert appearance.supported_outfit_colors('r') == {'orange','purple'}
    assert appearance.supported_outfit_colors('other') == set()
