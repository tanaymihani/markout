import pytest

from markout import plotting as mp


def test_render_writes_light_and_dark(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "FIGURES", tmp_path)
    seen = []

    def draw():
        seen.append(mp.surface())
        fig, ax = mp.subplots()
        ax.plot([0, 1], [0, 1], color=mp.series(0), label="a")
        ax.plot([0, 1], [1, 0], color=mp.series(1), label="b")
        mp.legend(ax)
        mp.title(ax, "t", "s")
        return fig

    paths = mp.render("x", draw)
    assert [p.name for p in paths] == ["x.png", "x_dark.png"]
    assert all(p.stat().st_size > 1000 for p in paths)
    assert seen == [mp.LIGHT["surface"], mp.DARK["surface"]]  # each theme drawn with its own colors
    assert mp.surface() == mp.LIGHT["surface"]  # theme restored afterwards


def test_categorical_slots_are_capped_at_eight():
    assert mp.series(7) == mp.LIGHT["series"][7]
    with pytest.raises(ValueError):
        mp.series(8)


def test_picture_snippet_points_at_both_variants():
    s = mp.picture("a_b", "alt text")
    assert 'srcset="figures/a_b_dark.png"' in s and 'src="figures/a_b.png"' in s
