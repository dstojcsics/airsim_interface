# This message contains camera data with odomerty
# https://docs.ros2.org/latest/api/std_msgs/msg/Header.html

# https://docs.ros2.org/latest/api/nav_msgs/msg/Odometry.html
nav_msgs/Odometry odomerty
# odomerty.header.stamp
# odomerty.pose.pose.position
# odomerty.pose.pose.orientation

# https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html
sensor_msgs/Image segmentation
# rgb.data (image_data_uint8)
# rgb.header (identical to Odom header)
# rgb.width
# rgb.height

# https://docs.ros2.org/latest/api/sensor_msgs/msg/Image.html
sensor_msgs/Image depth
# segmentation.data (image_data_uint8)
# segmentation.header (identical to Odom header)
# segmentation.width
# segmentation.height

# https://docs.ros2.org/latest/api/std_msgs/msg/Float32.html
std_msgs/Float32 difference
# Time difference between rgb and segmentation image