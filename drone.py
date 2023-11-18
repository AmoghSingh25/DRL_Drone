
import airsim
import time
from time import sleep
import numpy as np
import math
import cv2


def savePointCloud(image, fileName):
    f = open(fileName, "w")
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            pt = image[x, y]
            if (math.isinf(pt[0]) or math.isnan(pt[0])):
                # skip it
                None
            else:
                f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, rgb))
    f.close()


projectionMatrix = np.array([[-0.501202762, 0.000000000, 0.000000000, 0.000000000],
                             [0.000000000, -0.501202762, 0.000000000, 0.000000000],
                             [0.000000000, 0.000000000, 10.00000000, 100.00000000],
                             [0.000000000, 0.000000000, -10.0000000, 0.000000000]])

outputFile = "cloud.asc"
color = (0, 255, 0)
rgb = "%d %d %d" % color
client = airsim.MultirotorClient()
# client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()


print("Took off")
client.moveToPositionAsync(0, 0, 1, 3).join()
client.moveByVelocityBodyFrameAsync(0, 0, 10, 1).join()

# client.moveToPositionAsync(10, 0, 0, 1).join()
# client.moveToPositionAsync(0, 10, 0, 1).join()
client.hoverAsync().join()

print("Moved")
time.sleep(10)
# client.moveToPositionAsync(0, 0, -10, 3).join()
# client.hoverAsync().join()
# rawImage = client.simGetImage("0", airsim.ImageType.DepthPerspective)
# png = cv2.imdecode(np.frombuffer(rawImage, np.uint8), cv2.IMREAD_UNCHANGED)
# gray = cv2.cvtColor(png, cv2.COLOR_BGR2GRAY)
# Image3D = cv2.reprojectImageTo3D(gray, projectionMatrix)
# savePointCloud(Image3D, outputFile)
# print("saved " + outputFile)

# client.moveToPositionAsync(0, 0, -10, 5).join()
# client.hoverAsync().join()

# client.moveToPositionAsync(0, 0, -10, 5).join()
# client.hoverAsync().join()

# client.moveToPositionAsync(0, 0, -10, 5).join()
# client.hoverAsync().join()

# print(client.simGetCollisionInfo())

# client.landAsync().join()

# Get camera image from drone


# while True:

#     responses = client.simGetImages([
#         airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPerspective, True),
#         # airsim.ImageRequest("1", airsim.ImageType.Scene, False, False),
#         # airsim.ImageRequest("2", airsim.ImageType.Scene, False, False),
#         # airsim.ImageRequest("3", airsim.ImageType.Scene, False, False),
#         # airsim.ImageRequest("4", airsim.ImageType.Scene, False, False),

#     ])
#     # print(responses)
#     # response = responses[0]
#     # View depth image from AirSim camera front_center_custom
#     # print(responses[0].image_data_float)
#     print(responses[0])
#     img= np.array(responses[0].image_data_float)
#     print(max(img))
#     print(img.shape)
#     # print(responses[0].image_data_uint8)
#     # img1d = np.fromstring(responses[0].image_data_float, dtype=np.uint8)
#     img_rgb = img.reshape(responses[0].height, responses[0].width, 1)
#     cv2.imshow('image', img_rgb)
#     cv2.waitKey(1)
#     break

    