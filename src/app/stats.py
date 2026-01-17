import time
from dataclasses import dataclass, field

@dataclass
class StreamStats:
    started_at: float = field(default_factory=time.time)
    frames_ok: int = 0
    frames_drop: int = 0
    last_frame_ts: float | None=None

    def uptime_s(self) -> float:
        return time.time() - self.started_at
    
    def fps_estimate(self) -> float:
        uptime = self.uptime_s()
        return (self.frames_ok/uptime) if uptime > 0 else 0.0