import pytest

from threed.topology import Grid, parse_grid

GRIDS = [Grid(1, 1, 1), Grid(2, 2, 2), Grid(8, 4, 2), Grid(4, 2, 8), Grid(8, 8, 16)]


@pytest.mark.parametrize("g", GRIDS)
def test_coords_roundtrip_and_ranges(g):
    seen = set()
    for r in range(g.world):
        pp, dp, tp = g.coords(r)
        assert 0 <= pp < g.pp and 0 <= dp < g.dp and 0 <= tp < g.tp
        assert g.rank(pp, dp, tp) == r
        seen.add((pp, dp, tp))
    assert len(seen) == g.world


@pytest.mark.parametrize("g", GRIDS)
def test_groups_partition_the_world(g):
    for kind, size in (("tp", g.tp), ("dp", g.dp), ("pp", g.pp)):
        groups = g.all_groups(kind)
        assert all(len(x) == size for x in groups)
        flat = sorted(r for x in groups for r in x)
        assert flat == list(range(g.world))               # every rank in exactly one group of each kind


def test_groups_are_orthogonal():
    g = Grid(4, 2, 8)
    for r in range(g.world):
        assert set(g.tp_group(r)) & set(g.dp_group(r)) == {r}
        assert set(g.tp_group(r)) & set(g.pp_group(r)) == {r}
        assert set(g.dp_group(r)) & set(g.pp_group(r)) == {r}


def test_tensor_parallel_stays_inside_a_node():
    d = Grid(8, 4, 2).describe(8)
    assert d["tp"]["inside_one_node"] == d["tp"]["groups"] == 8
    assert d["pp"]["inside_one_node"] == 0 and d["dp"]["inside_one_node"] == 0


def test_parse_and_bad_rank():
    assert parse_grid("8x4x2") == Grid(8, 4, 2) == parse_grid("tp=8,pp=4,dp=2")
    with pytest.raises(ValueError):
        Grid(2, 2, 2).coords(8)
