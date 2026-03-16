from abc import ABC, abstractmethod

class MPAccelHAL(ABC):
    @abstractmethod
    async def write(self, addr: int, data: int):
        """Writes a 32-bit word to the hardware."""
        pass

    @abstractmethod
    async def read(self, addr: int) -> int:
        """Reads a 32-bit word from the hardware."""
        pass

    @abstractmethod
    async def wait_for_irq(self, timeout: int = 100000):
        """Waits for the hardware interrupt."""
        pass
