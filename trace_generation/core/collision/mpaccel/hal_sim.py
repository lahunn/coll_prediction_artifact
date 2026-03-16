from .hal import MPAccelHAL
import cocotb
from cocotb.triggers import RisingEdge

class CocotbHALAdapter(MPAccelHAL):
    def __init__(self, axi_master):
        self.axi = axi_master

    async def write(self, addr: int, data: int):
        # cocotbext-axi write expects bytes
        await self.axi.write(addr, data.to_bytes(4, "little"))

    async def read(self, addr: int) -> int:
        data = await self.axi.read(addr, 4)
        return int.from_bytes(data.data, "little")

    async def wait_for_irq(self, timeout: int = 1000000):
        # We need access to the dut's irq_o port.
        # This adapter assumes the axi_master is tied to the dut and we can find the port.
        # Usually in cocotb tests, the irq is part of the dut object.
        dut = self.axi.bus.reset_n._parent # Hacky way to get DUT? 
        # Better: let user pass dut or irq signal
        if hasattr(dut, 'irq_o'):
            for _ in range(timeout):
                await RisingEdge(dut.clk)
                if dut.irq_o.value:
                    return
            raise TimeoutError("Simulation timeout waiting for IRQ")
        else:
             raise RuntimeError("DUT does not have irq_o port")
