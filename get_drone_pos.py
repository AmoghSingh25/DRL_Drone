import airsim

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Get the drone's state information
state = client.getMultirotorState()

# Get the position of the drone in X, Y, Z coordinates
position_x = state.kinematics_estimated.position.x_val
position_y = state.kinematics_estimated.position.y_val
position_z = state.kinematics_estimated.position.z_val

# Print the drone's position
print("Drone Position (X, Y, Z):", position_x, position_y, position_z)
