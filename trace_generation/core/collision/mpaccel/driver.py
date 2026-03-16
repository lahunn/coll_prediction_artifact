import numpy as np
import math
import asyncio
from typing import List, Dict, Any, Optional
from .config import *
from .hal import MPAccelHAL

class MPAccelDriver:
    def __init__(self, hal: MPAccelHAL):
        self.hal = hal
        self.queued_tasks = []
        # 硬件互斥锁：确保 AXI 总线和硬件核心在配置与运行期间不被干扰
        self.hw_lock = asyncio.Lock()

    async def init_robot(self, robot_config: Dict[str, Any]):
        """
        Configures the robot geometry (SGU and LLUT).
        """
        async with self.hw_lock:
            links = robot_config.get("links", [])
            current_sgu_geom_row = 0

            for i, link in enumerate(links):
                spheres = link.get("spheres", [])
                cnt = len(spheres)
                is_last = (i == len(links) - 1)
                
                # 1. LLUT Entry
                llut_val = pack_llut_entry(cnt, current_sgu_geom_row)
                await self.hal.write(ADDR_SGU_CONFIG + i * 4, llut_val)

                # 2. Link Parameters
                w0, w1 = pack_link_params(
                    rot_id=link.get("rot_id", 0),
                    tx=link["t"][0], ty=link["t"][1], tz=link["t"][2],
                    active=link.get("active", True),
                    is_last=is_last,
                    axis_type=link.get("axis_type", 0),
                    joint_idx=link.get("joint_idx", 0),
                    invert_axis=link.get("invert_axis", 0)
                )
                link_param_base = ADDR_SGU_CONFIG + 0x100 + i * 8
                await self.hal.write(link_param_base, w0)
                await self.hal.write(link_param_base + 4, w1)

                # 3. Sphere Geometries
                for si, s in enumerate(spheres):
                    row = current_sgu_geom_row + (si // 4)
                    lane = si % 4
                    sw0, sw1 = pack_sphere_geom(s[0], s[1], s[2], s[3])
                    addr = ADDR_SGU_GEOM + row * 32 + lane * 8
                    await self.hal.write(addr, sw0)
                    await self.hal.write(addr + 4, sw1)
                
                current_sgu_geom_row += (cnt + 3) // 4

    async def update_environment(self, obstacles: List[Any]):
        """
        Configures the Octree Environment (DEM).
        """
        async with self.hw_lock:
            octree_data = generate_octree_data(obstacles)
            for i, val in enumerate(octree_data):
                await self.hal.write(ADDR_OCTREE_BASE + i * 4, val)

    def add_edge_task(self, state_start: np.ndarray, state_end: np.ndarray, eps: float = 0.05):
        """
        Adds an edge feasibility task to the Python-side queue.
        This is a lightweight O(1) operation, intended for explicit batching in algorithms like BIT*.
        """
        diff = state_end - state_start
        dist = np.linalg.norm(diff)
        steps = max(1, int(dist / eps))
        delta = diff / steps
        
        task = {
            "start": state_start,
            "delta": delta,
            "count": steps
        }
        self.queued_tasks.append(task)

    async def execute_batch(self) -> List[bool]:
        """
        Explicitly triggers hardware execution for the current task queue.
        Protected by hardware mutex.
        """
        async with self.hw_lock:
            return await self._execute_batch_internal()

    async def _execute_batch_internal(self) -> List[bool]:
        """
        Internal implementation of batch execution. Assumes hw_lock is already held.
        """
        if not self.queued_tasks:
            return []

        num_tasks = min(len(self.queued_tasks), 16)
        tasks_to_run = self.queued_tasks[:num_tasks]
        self.queued_tasks = self.queued_tasks[num_tasks:]

        # 1. Write Tasks to SAS RAM
        for i, task in enumerate(tasks_to_run):
            packed_config = pack_sas_motion_config(
                motion_id=i,
                pose_count=task["count"],
                start_pose=task["start"],
                step_delta=task["delta"]
            )
            base_addr = ADDR_SAS_RAM_BASE + i * 32
            for word_idx in range(8):
                word_val = (packed_config >> (word_idx * 32)) & 0xFFFFFFFF
                await self.hal.write(base_addr + word_idx * 4, word_val)

        # 2. Start Hardware
        await self.hal.write(ADDR_CSR_CTRL, 1)

        # 3. Wait for IRQ
        await self.hal.wait_for_irq()

        # 4. Read Results
        collision_map = await self.hal.read(ADDR_COLLISION_MAP)
        
        results = []
        for i in range(num_tasks):
            collided = (collision_map >> i) & 0x1
            results.append(not bool(collided)) 
            
        # Reset CTRL
        await self.hal.write(ADDR_CSR_CTRL, 0)
        return results

    async def _edge_fp(self, state_start: np.ndarray, state_end: np.ndarray, eps: float = 0.05) -> bool:
        """
        Atomic sequential edge check. 
        Intended for algorithms that need immediate results (like serial RRT).
        """
        async with self.hw_lock:
            # Note: This will also flush any previously 'orphaned' tasks in the queue
            idx = len(self.queued_tasks)
            self.add_edge_task(state_start, state_end, eps)
            
            # Flush queue until we reach our task
            while idx >= 16:
                await self._execute_batch_internal()
                idx -= 16
                
            results = await self._execute_batch_internal()
            return results[idx]

    async def _state_fp(self, state: np.ndarray) -> bool:
        """
        Atomic sequential state check.
        """
        return await self._edge_fp(state, state, eps=1.0)
