import airsim
import time
# Connect to the AirSim client

class controlDrone:
    def __init__(self, clock_speed = 150):
        self.client = airsim.MultirotorClient()
        self.clock_speed = clock_speed
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.start_state = self.client.getMultirotorState()

        self.start_pos = self.start_state.kinematics_estimated.position
        self.z_val = self.start_pos.z_val
        self.client.moveToPositionAsync(0,0,self.z_val, 1/self.clock_speed).join()

    def move(self, vx, vy, vz, duration):
        # vx = vx / self.clock_speed
        # vy = vy / self.clock_speed
        # vz = vz / self.clock_speed
        duration = duration/ self.clock_speed
        self.client.moveByVelocityAsync(vx, vy, vz, duration).join()
        time.sleep(duration)
        self.client.moveByVelocityAsync(0,0,0, duration).join()


    def _reset(self):
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()