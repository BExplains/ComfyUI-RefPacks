"""Shared sampling timing wrappers; never modify numerical inputs."""
import time


def log(message):
    print(f"[BExplains RefPacks] {message}", flush=True)




class SamplingTimings:
    def __init__(self):
        self.loop_seconds = None

    def outer(self, executor, *args, **kwargs):
        self.loop_seconds = None
        started = time.perf_counter()
        succeeded = False
        log("Sampling entered: model preparation/loading, conditioning processing and denoising follow.")
        try:
            result = executor(*args, **kwargs)
            succeeded = True
            return result
        finally:
            elapsed = time.perf_counter() - started
            status = "finished" if succeeded else "interrupted/failed"
            log(f"Sampling {status}: {elapsed:.3f}s total wall time")
            if self.loop_seconds is not None:
                log(f"Breakdown: sampler loop {self.loop_seconds:.3f}s; other work inside sampling {max(0.0, elapsed-self.loop_seconds):.3f}s")

    def loop(self, executor, *args, **kwargs):
        started = time.perf_counter()
        log("Sampler loop starting.")
        try:
            return executor(*args, **kwargs)
        finally:
            self.loop_seconds = time.perf_counter() - started
            log(f"Sampler loop ended: {self.loop_seconds:.3f}s wall time")


def profile_model(model):
    import comfy.patcher_extension as ext
    timings = SamplingTimings()
    result = model.clone()
    ext.add_wrapper_with_key(ext.WrappersMP.OUTER_SAMPLE, "BExplains.RefPacks.Timing", timings.outer, result.model_options, is_model_options=True)
    ext.add_wrapper_with_key(ext.WrappersMP.SAMPLER_SAMPLE, "BExplains.RefPacks.Timing", timings.loop, result.model_options, is_model_options=True)
    return result
