from .hal import MPAccelHAL
import asyncio
import time

class PynqHALAdapter(MPAccelHAL):
    def __init__(self, ip_handle):
        """
        Args:
            ip_handle: The IP object from pynq.Overlay (e.g., overlay.mpaccel_top_0)
        """
        self.ip = ip_handle
        # Assuming PYNQ handles MMIO under the hood

    async def write(self, addr: int, data: int):
        # Pynq MMIO write is typically synchronous
        self.ip.write(addr, data)

    async def read(self, addr: int) -> int:
        return self.ip.read(addr)

    async def wait_for_irq(self, timeout: int = 1000000):
        """
        Wait for hardware interrupt. 
        PYNQ IP objects usually have a .interrupt property if configured in Vivado.
        """
        if hasattr(self.ip, 'interrupt'):
            await self.ip.interrupt.wait()
        else:
            # Fallback to polling if interrupt not wired in Overlay
            start_time = time.time()
            while True:
                status = await self.read(0x0004) # ADDR_STATUS
                if status & 0x1: # Assuming bit 0 is 'done' or similar
                    break
                await asyncio.sleep(0.001)
                if (time.time() - start_time) * 1000000 > timeout:
                    raise TimeoutError("Hardware timeout during polling")
