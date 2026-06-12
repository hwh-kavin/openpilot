import datetime
import subprocess
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

MIN_DATE = datetime.datetime(year=2025, month=2, day=21)
MAX_DATE = datetime.datetime(year=2035, month=1, day=1)

NETWORK_TIME_URLS = (
  'http://www.baidu.com',
  'http://connectivitycheck.gstatic.com/generate_204',
)

def min_date():
  # on systemd systems, the default time is the systemd build time
  systemd_path = Path("/lib/systemd/systemd")
  if systemd_path.exists():
    d = datetime.datetime.fromtimestamp(systemd_path.stat().st_mtime)
    return max(MIN_DATE, d + datetime.timedelta(days=1))
  return MIN_DATE

def system_time_valid():
  return min_date() < datetime.datetime.now() < MAX_DATE

def set_system_time(new_time: datetime.datetime) -> bool:
  from openpilot.common.swaglog import cloudlog

  diff = datetime.datetime.now() - new_time
  if abs(diff) < datetime.timedelta(seconds=10):
    cloudlog.debug(f"Time diff too small: {diff}")
    return False

  cloudlog.info(f"Setting system time to {new_time} UTC")
  try:
    subprocess.run(f"TZ=UTC date -s '{new_time}'", shell=True, check=True)
    return True
  except subprocess.CalledProcessError:
    cloudlog.exception("time_helpers.failed_setting_time")
    return False

def fetch_network_time() -> Optional[datetime.datetime]:
  from openpilot.common.swaglog import cloudlog

  for url in NETWORK_TIME_URLS:
    try:
      req = Request(url, method='HEAD')
      with urlopen(req, timeout=8) as r:
        date_hdr = r.headers.get('Date')
        if not date_hdr:
          continue
        dt = parsedate_to_datetime(date_hdr)
        return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    except Exception:
      cloudlog.debug(f"Failed to fetch time from {url}", exc_info=True)
  return None

def sync_time_from_network() -> bool:
  new_time = fetch_network_time()
  if new_time is None:
    return False

  set_system_time(new_time)
  return system_time_valid()
