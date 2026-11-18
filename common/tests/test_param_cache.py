import pytest

from openpilot.common.params import Params
from openpilot.common.param_cache import ParamCache


@pytest.fixture
def params(tmp_path):
  return Params(str(tmp_path / "params"))


def test_get_returns_cached_value_without_rereading_storage(params, tmp_path):
  params.put("PlanplusControl", 0.5, block=True)

  cache = ParamCache(params, ttl=60.0)
  assert cache.get("PlanplusControl") == pytest.approx(0.5)

  # Change the underlying param behind the cache's back.
  writer = Params(str(tmp_path / "params"))
  writer.put("PlanplusControl", 0.25, block=True)

  # A fresh ParamCache sees the new value, the cached one must still serve RAM.
  assert ParamCache(params, ttl=60.0).get("PlanplusControl") == pytest.approx(0.25)
  assert cache.get("PlanplusControl") == pytest.approx(0.5)

  cache.invalidate()
  assert cache.get("PlanplusControl") == pytest.approx(0.25)


def test_get_matches_params_for_unset_key_and_default(params):
  cache = ParamCache(params, ttl=60.0)

  # Unset key: same as Params.get() without return_default -> None.
  assert cache.get("CameraOffset") is None
  # return_default=True must resolve the declared default, like Params does.
  assert cache.get("CameraOffset", return_default=True) == params.get("CameraOffset", return_default=True)


def test_get_bool_matches_params_semantics(params):
  cache = ParamCache(params, ttl=60.0)

  params.put_bool("Blendful", True, block=True)
  assert cache.get_bool("Blendful") is True
  assert cache.get_bool("Blendful") == params.get_bool("Blendful")

  cache.invalidate()
  params.put_bool("Blendful", False, block=True)
  assert cache.get_bool("Blendful") is False
  assert cache.get_bool("Blendful") == params.get_bool("Blendful")


def test_missing_key_is_false_for_get_bool(params):
  # An unset boolean is "0" in Params terms, so get_bool must not raise.
  cache = ParamCache(params, ttl=60.0)
  assert cache.get_bool("IsMetric") is False


def test_expired_ttl_refreshes(params, tmp_path):
  params.put("PlanplusControl", 0.5, block=True)
  cache = ParamCache(params, ttl=0.01)

  assert cache.get("PlanplusControl") == pytest.approx(0.5)
  Params(str(tmp_path / "params")).put("PlanplusControl", 0.75, block=True)

  import time
  time.sleep(0.02)
  assert cache.get("PlanplusControl") == pytest.approx(0.75)


def test_non_positive_ttl_rejected(params):
  with pytest.raises(ValueError):
    ParamCache(params, ttl=0.0)
