from abc import ABC, abstractmethod

class LRInterface(ABC):

    @abstractmethod
    def get_lr(self, it: int) -> float:
        pass
