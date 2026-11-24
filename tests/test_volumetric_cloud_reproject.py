import pytest
import graphics.volumetric_cloud_reproject as mod

CASES = [
    ("pixel_hash", (1,)),
    ("motion_reproject", (1,)),
    ("clear_history", (1,)),
    ("upsample_first", (1,)),
    ("weather_reset", (1,)),
    ("converge", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("raymarch", (1,)),
    ("metrics", (1,)),
    ("density", (1,)),
    ("composite", (1,)),
    ("accumulate", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
