REGISTRY = {}


from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner


from .paralle_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner