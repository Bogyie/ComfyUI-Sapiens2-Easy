class NodeProgress:
    def __init__(self, total: int):
        self._progress = None
        self._total = max(0, int(total))
        if self._total <= 0:
            return
        try:
            import comfy.utils

            self._progress = comfy.utils.ProgressBar(self._total)
        except Exception:
            self._progress = None

    def update(self, amount: int = 1) -> None:
        if self._progress is None:
            return
        try:
            self._progress.update(max(0, int(amount)))
        except Exception:
            self._progress = None
